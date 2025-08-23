# pylint: disable=E1101
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class create_classifier(nn.Module):
    def __init__(self, latent_dim, nhidden=16, N=2):
        super(create_classifier, self).__init__()
        self.gru_rnn = nn.GRU(latent_dim, nhidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, N),
        )

    def forward(self, z):
        _, out = self.gru_rnn(z)
        return self.classifier(out.squeeze(0))


class multiTimeAttention(nn.Module):
    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_time, embed_time),
                nn.Linear(embed_time, embed_time),
                nn.Linear(input_dim * num_heads, nhidden),
            ]
        )

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, _, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [
            l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class enc_mtan_rnn(nn.Module):
    def __init__(
        self,
        input_dim,
        query,
        latent_dim=2,
        nhidden=16,
        embed_time=16,
        num_heads=1,
        learn_emb=False,
        device="cuda",
    ):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.learn_emb = learn_emb
        # FIX: keep query as a buffer so it moves with .to(device)
        self.register_buffer("query", torch.as_tensor(query, dtype=torch.float32))

        self.att = multiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, latent_dim * 2)
        )
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def fixed_time_embedding(self, ref):
        """
        Build sinusoidal PE on the same device/dtype as `ref`.
        Returns: [B, L, self.embed_time]
        """
        device = ref.device
        dtype = ref.dtype if ref.is_floating_point() else torch.float32

        # infer B, L
        if ref.dim() >= 2:
            B, L = ref.size(0), ref.size(1)
        else:
            B, L = 1, int(ref.numel())

        E = self.embed_time
        pe = torch.zeros(B, L, E, device=device, dtype=dtype)

        position = torch.arange(L, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
        div_term = torch.exp(
            torch.arange(0, E, 2, device=device, dtype=dtype) * (-math.log(10000.0) / E)
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, time_steps):
        # FIX: do NOT move time_steps to CPU
        mask = x[:, :, self.dim :]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.unsqueeze(0))
        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out


class dec_mtan_rnn(nn.Module):
    def __init__(
        self,
        input_dim,
        query,
        latent_dim=2,
        nhidden=16,
        embed_time=16,
        num_heads=1,
        learn_emb=False,
        device="cuda",
    ):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.learn_emb = learn_emb
        # FIX: buffer for query
        self.register_buffer("query", torch.as_tensor(query, dtype=torch.float32))

        self.att = multiTimeAttention(2 * nhidden, 2 * nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, input_dim)
        )
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def fixed_time_embedding(self, x):
        """
        returns: [B, L, self.embed_time] on x.device/x.dtype
        """
        device = x.device
        dtype = x.dtype if x.is_floating_point() else torch.float32
        B, L = x.size(0), x.size(1)
        E = self.embed_time
        pe = torch.zeros(B, L, E, device=device, dtype=dtype)
        position = torch.arange(L, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
        div_term = torch.exp(
            torch.arange(0, E, 2, device=device, dtype=dtype) * (-math.log(10000.0) / E)
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        # FIX: do NOT move time_steps to CPU
        if self.learn_emb:
            query = self.learn_time_embedding(time_steps)
            key = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            query = self.fixed_time_embedding(time_steps)
            key = self.fixed_time_embedding(self.query.unsqueeze(0))
        out = self.att(query, key, out)
        out = self.z0_to_obs(out)
        return out


class enc_mtan_classif(nn.Module):
    def __init__(
        self,
        input_dim,
        query,
        nhidden=16,
        embed_time=16,
        num_heads=1,
        learn_emb=True,
        freq=10.0,
        device="cuda",
    ):
        super(enc_mtan_classif, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = float(freq)
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        # FIX: buffer for query
        self.register_buffer("query", torch.as_tensor(query, dtype=torch.float32))

        self.att = multiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 2),
        )
        self.enc = nn.GRU(nhidden, nhidden)
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    # FIX: device-aware time embedding
    def time_embedding(self, pos, d_model):
        device = pos.device
        dtype = pos.dtype if pos.is_floating_point() else torch.float32
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=device, dtype=dtype)
        position = (48.0 * pos).unsqueeze(2).to(device=device, dtype=dtype)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=dtype)
            * -(torch.log(torch.tensor(self.freq, device=device, dtype=dtype)) / d_model)
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, time_steps):
        # FIX: no .cpu()
        mask = x[:, :, self.dim :]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key = self.time_embedding(time_steps, self.embed_time)
            query = self.time_embedding(self.query.unsqueeze(0), self.embed_time)

        out = self.att(query, key, x, mask)
        out = out.permute(1, 0, 2)
        _, out = self.enc(out)
        return self.classifier(out.squeeze(0))


class enc_mtan_classif_activity(nn.Module):
    def __init__(
        self,
        input_dim,
        nhidden=16,
        embed_time=16,
        num_heads=1,
        learn_emb=True,
        freq=10.0,
        device="cuda",
    ):
        super(enc_mtan_classif_activity, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = float(freq)
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.att = multiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        self.gru = nn.GRU(nhidden, nhidden, batch_first=True)
        self.classifier = nn.Linear(nhidden, 11)
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    # FIX: device-aware time embedding
    def time_embedding(self, pos, d_model):
        device = pos.device
        dtype = pos.dtype if pos.is_floating_point() else torch.float32
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=device, dtype=dtype)
        position = (48.0 * pos).unsqueeze(2).to(device=device, dtype=dtype)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=dtype)
            * -(torch.log(torch.tensor(self.freq, device=device, dtype=dtype)) / d_model)
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, time_steps):
        mask = x[:, :, self.dim :]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
        else:
            key = self.time_embedding(time_steps, self.embed_time)
        out = self.att(key, key, x, mask)
        out, _ = self.gru(out)
        out = self.classifier(out)
        return out


class enc_interp(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, device="cuda"):
        super(enc_interp, self).__init__()
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        # FIX: buffer for query
        self.register_buffer("query", torch.as_tensor(query, dtype=torch.float32))

        self.cross = nn.Linear(2 * input_dim, 2 * input_dim)
        self.bandwidth = nn.Linear(1, 2 * input_dim, bias=False)
        self.gru_rnn = nn.GRU(
            2 * input_dim, nhidden, bidirectional=True, batch_first=True
        )
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, latent_dim * 2)
        )

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        query, key = query.to(self.device), key.to(self.device)
        batch, _, dim = value.size()
        scores = -((query.unsqueeze(-1) - key.unsqueeze(-2)) ** 2)
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        bandwidth = torch.log(
            1 + torch.exp(self.bandwidth(torch.ones(1, 1, 1, 1, device=self.device)))
        )
        scores = scores * bandwidth
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(1), -2), p_attn

    def forward(self, x, time_steps):
        mask = x[:, :, self.dim :]
        mask = torch.cat((mask, mask), 2)
        out, _ = self.attention(self.query.unsqueeze(0), time_steps, x, mask)
        out = self.cross(out)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out


class dec_interp(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, device="cuda"):
        super(dec_interp, self).__init__()
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        # FIX: buffer for query
        self.register_buffer("query", torch.as_tensor(query, dtype=torch.float32))

        self.bandwidth = nn.Linear(1, 2 * nhidden, bias=False)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, input_dim)
        )

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        query, key = query.to(self.device), key.to(self.device)
        batch, _, dim = value.size()
        scores = -((query.unsqueeze(-1) - key.unsqueeze(-2)) ** 2)
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        bandwidth = torch.log(
            1 + torch.exp(self.bandwidth(torch.ones(1, 1, 1, 1, device=self.device))))
        scores = scores * bandwidth
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(1), -2), p_attn

    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        out, _ = self.attention(time_steps, self.query.unsqueeze(0), out)
        out = self.z0_to_obs(out)
        return out


class enc_rnn3(nn.Module):
    def __init__(
        self,
        input_dim,
        query,
        latent_dim=2,
        nhidden=16,
        embed_time=16,
        use_classif=False,
        learn_emb=False,
        device="cuda",
    ):
        super(enc_rnn3, self).__init__()
        self.use_classif = use_classif
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.learn_emb = learn_emb
        # FIX: buffer for query
        self.register_buffer("query", torch.as_tensor(query, dtype=torch.float32))

        self.cross = nn.Linear(2 * input_dim, nhidden)
        if use_classif:
            self.gru_rnn = nn.GRU(nhidden, nhidden, batch_first=True)
        else:
            self.gru_rnn = nn.GRU(
                nhidden, nhidden, bidirectional=True, batch_first=True
            )
        self.linears = nn.ModuleList(
            [nn.Linear(embed_time, embed_time) for _ in range(2)]
        )
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)
        if use_classif:
            self.classifier = nn.Sequential(
                nn.Linear(nhidden, 300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, 2),
            )
        else:
            self.hiddens_to_z0 = nn.Sequential(
                nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, latent_dim * 2)
            )

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    # FIX: device-aware sinusoid
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        device = pos.device
        dtype = pos.dtype if pos.is_floating_point() else torch.float32
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=device, dtype=dtype)
        position = (48.0 * pos).unsqueeze(2).to(device=device, dtype=dtype)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=dtype)
            * -(torch.log(torch.tensor(10.0, device=device, dtype=dtype)) / d_model)
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, _, dim = value.size()
        d_k = query.size(-1)
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(1), -2), p_attn

    def forward(self, x, time_steps):
        mask = x[:, :, self.dim :]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.unsqueeze(0))
        out, _ = self.attention(query, key, x, mask)
        out = self.cross(out)
        if not self.use_classif:
            out, _ = self.gru_rnn(out)
            out = self.hiddens_to_z0(out)
        else:
            _, h = self.gru_rnn(out)
            out = self.classifier(h.squeeze(0))
        return out


class dec_rnn3(nn.Module):
    def __init__(
        self,
        input_dim,
        query,
        latent_dim=2,
        nhidden=16,
        embed_time=16,
        learn_emb=False,
        device="cuda",
    ):
        super(dec_rnn3, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.learn_emb = learn_emb
        # FIX: buffer for query
        self.register_buffer("query", torch.as_tensor(query, dtype=torch.float32))

        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_time, embed_time),
                nn.Linear(embed_time, embed_time),
                nn.Linear(2 * nhidden, 2 * nhidden),
            ]
        )
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, input_dim)
        )

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    # FIX: device-aware sinusoid
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        device = pos.device
        dtype = pos.dtype if pos.is_floating_point() else torch.float32
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=device, dtype=dtype)
        position = (48.0 * pos).unsqueeze(2).to(device=device, dtype=dtype)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=dtype)
            * -(torch.log(torch.tensor(10.0, device=device, dtype=dtype)) / d_model)
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, _, dim = value.size()
        d_k = query.size(-1)
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(1), -2), p_attn

    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        if self.learn_emb:
            query = self.learn_time_embedding(time_steps)
            key = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            query = self.fixed_time_embedding(time_steps)
            key = self.fixed_time_embedding(self.query.unsqueeze(0))
        out, _ = self.attention(query, key, out)
        out = self.z0_to_obs(out)
        return out


class mTANDFull(nn.Module):
    def __init__(
        self,
        device,
        dim,
        num_ref_points=64,
        latent_dim=16,
        rec_hidden=32,
        gen_hidden=50,
        enc_num_heads=1,
        dec_num_heads=1,
        embed_time=128,
        learn_emb=True,
        k_iwae=5,
    ):
        super(mTANDFull, self).__init__()
        ref = torch.linspace(0, 1.0, num_ref_points)  # stays CPU; modules register as buffers
        self.enc = enc_mtan_rnn(
            dim,
            ref,
            latent_dim,
            rec_hidden,
            embed_time=embed_time,
            learn_emb=learn_emb,
            num_heads=enc_num_heads,
            device=device,
        ).to(device)
        self.dec = dec_mtan_rnn(
            dim,
            ref,
            latent_dim,
            gen_hidden,
            embed_time=embed_time,
            learn_emb=learn_emb,
            num_heads=dec_num_heads,
            device=device,
        ).to(device)
        self.latent_dim = latent_dim
        self.device = device
        self.k_iwae = k_iwae

    def forward(self, timesteps, X, M, y_time_steps, test=False):
        out = self.enc(torch.cat((X, M), 2), timesteps)
        qz0_mean = out[:, :, : self.latent_dim]
        qz0_logvar = out[:, :, self.latent_dim :]
        epsilon = torch.randn(
            self.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2], device=self.device
        )
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
        pred_x = self.dec(
            z0,
            y_time_steps[None, :, :].repeat(self.k_iwae, 1, 1).view(-1, y_time_steps.shape[1]),
        )
        pred_x = pred_x.view(self.k_iwae, X.shape[0], pred_x.shape[1], pred_x.shape[2])
        if test:
            pred_x = pred_x.mean(0)
        return pred_x, qz0_mean, qz0_logvar
