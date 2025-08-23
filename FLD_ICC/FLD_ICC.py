# FLD_ICC/FLD_ICC.py\
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Multi-head attention over flattened (time x channel) memory ----------
class FLDAttention(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int, num_heads: int = 2):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.h = num_heads
        self.k = embed_dim // num_heads
        self.E = embed_dim

        self.Wq = nn.Linear(self.E, self.E)
        self.Wk = nn.Linear(self.E, self.E)
        self.Wv = nn.Linear(self.E, self.E)
        self.Wo = nn.Linear(self.E, out_dim)

    def forward(self, Q, K, V, mask):
        """
        Q: [B, P, E]       (P = #basis)
        K: [B, S, E]       (S = T*C flattened)
        V: [B, S, E]
        mask: [B, S] bool  (True = valid)
        -> [B, P, out_dim]
        """
        B, P, E = Q.shape
        S = K.shape[1]

        Qp = self.Wq(Q).view(B, P, self.h, self.k).permute(0, 2, 1, 3)  # [B,h,P,k]
        Kp = self.Wk(K).view(B, S, self.h, self.k).permute(0, 2, 1, 3)  # [B,h,S,k]
        Vp = self.Wv(V).view(B, S, self.h, self.k).permute(0, 2, 1, 3)  # [B,h,S,k]

        scores = torch.einsum("bhpk,bhsk->bhps", Qp, Kp) / math.sqrt(self.k)  # [B,h,P,S]
        m = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,S]
        scores = scores.masked_fill(~m, float("-inf"))
        A = F.softmax(scores, dim=-1)  # [B,h,P,S]

        C = torch.einsum("bhps,bhsk->bhpk", A, Vp)  # [B,h,P,k]
        C = C.permute(0, 2, 1, 3).contiguous().view(B, P, self.E)  # [B,P,E]
        return self.Wo(C)  # [B,P,out_dim]


# ---------- Inter-Channel FLD (ICC): attention over all (time,channel) + basis decoder ----------
class IC_FLD(nn.Module):
    """
    Predicts y(t) = sum_p c_p * phi_p(t) per channel.
    Attention reads the entire history over all channels (flattened T*C memory)
    to produce per-basis coefficients for each channel.

    Args:
        input_dim (int): #channels C
        latent_dim (int): attention output size per basis before projecting to channels
        num_heads (int): attention heads
        embed_dim (int): embedding size for Q/K/V
        function (str): basis: 'C' (const), 'L' (linear), 'Q' (quadratic), 'S' (sin/cos, 2*harmonics)
        residual_cycle (bool): if True, subtract/add periodic baseline (requires physical time units)
        cycle_length (int): cycle period in hours (or your unit)
        sinusoid_harmonics (int): #harmonics for sinusoidal basis (used when function='S')
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_heads: int,
        embed_dim: int,
        function: str,
        residual_cycle: bool = False,
        cycle_length: int = 24,
        sinusoid_harmonics: int = 2,
    ):
        super().__init__()
        self.C = input_dim
        self.E = embed_dim
        self.residual_cycle = residual_cycle
        self.cycle_length = cycle_length
        self.F = function.upper()
        self.H = sinusoid_harmonics

        # number of basis terms P
        if self.F == "C":   self.P = 1
        elif self.F == "L": self.P = 2
        elif self.F == "Q": self.P = 3
        elif self.F == "S": self.P = 2 * self.H  # sin & cos pairs
        else: raise ValueError(f"Unknown function {function}")

        # embeddings
        self.time_linear   = nn.Linear(1, self.E)
        self.channel_embed = nn.Parameter(torch.randn(self.C, self.E))
        self.value_proj    = nn.Linear(1, self.E)  # project observed values into the memory

        # learned per-basis queries
        self.q_basis = nn.Parameter(torch.randn(self.P, self.E))  # [P,E]

        # attention over flattened memory
        self.attn = FLDAttention(self.E, latent_dim, num_heads)

        # map per-basis latent to per-channel coefficient
        self.coeff_proj = nn.Linear(latent_dim, self.C)  # -> [B,P,C]

    # ---- helpers ----
    def _time_embed(self, tt):  # tt: [B,T] normalized to [0,1]
        e = self.time_linear(tt.unsqueeze(-1))  # [B,T,E]
        return e + torch.sin(e)

    def _basis(self, y_times):  # y_times: [B,Ty] normalized to [0,1]
        B, Ty = y_times.shape
        t = y_times
        if self.F == "C":
            Phi = torch.ones(B, Ty, 1, device=y_times.device, dtype=y_times.dtype)
        elif self.F == "L":
            Phi = torch.stack([torch.ones_like(t), t], dim=-1)
        elif self.F == "Q":
            Phi = torch.stack([torch.ones_like(t), t, t*t], dim=-1)
        else:
            cols = []
            for k in range(1, self.H+1):
                cols += [torch.sin(2*math.pi*k*t), torch.cos(2*math.pi*k*t)]
            Phi = torch.stack(cols, dim=-1)  # [B,Ty,2H]
        return Phi  # [B,Ty,P]

    def _cycle_baseline(self, timesteps_phys, X, M):
        """ timesteps_phys must be in physical units (e.g., hours). """
        B, T, C = X.shape
        phases = torch.floor(timesteps_phys).long().to(X.device) % self.cycle_length
        base = torch.zeros(B, self.cycle_length, C, device=X.device)
        for p in range(self.cycle_length):
            mask_p = (phases == p).unsqueeze(-1) & M.bool()
            sum_p  = (X * mask_p).sum(dim=1)
            cnt_p  = mask_p.sum(dim=1).clamp(min=1)
            base[:, p] = sum_p / cnt_p
        c_in = base.gather(1, phases.unsqueeze(-1).expand(-1, -1, C))
        return c_in, base

    # ---- forward ----
    def forward(self, timesteps, X, M, y_times, denorm_time_max: Optional[float] = None):        
        """
        timesteps: [B,T]   normalized to [0,1]
        X:         [B,T,C]
        M:         [B,T,C] (0/1)
        y_times:   [B,Ty]  normalized to [0,1]
        denorm_time_max: if not None and residual_cycle=True, used to rescale times to physical units
        returns: [B,Ty,C]
        """
        B, T, C = X.shape
        assert C == self.C

        # (optional) periodic residual on physical timeline
        if self.residual_cycle:
            if denorm_time_max is None:
                # If you enable residual_cycle, pass denorm_time_max from the trainer (e.g., 48 for PhysioNet).
                raise ValueError("residual_cycle=True requires denorm_time_max (e.g., 48 for PhysioNet).")
            tt_phys = timesteps * denorm_time_max
            c_in, c_base = self._cycle_baseline(tt_phys, X, M)
            X = X - c_in

        # memory over all (time, channel) slots
        Et = self._time_embed(timesteps).unsqueeze(2).expand(-1, -1, C, -1)         # [B,T,C,E]
        Ec = self.channel_embed.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)       # [B,T,C,E]
        Ex = self.value_proj(X.unsqueeze(-1))                                        # [B,T,C,E]
        KV = Et + Ec + Ex

        K = KV.view(B, T*C, self.E)                                                 # [B,S,E]
        V = K
        mask_flat = M.reshape(B, T*C).bool()                                        # [B,S]

        # queries: one per basis (shared across batch)
        Q = self.q_basis.unsqueeze(0).expand(B, -1, -1)                             # [B,P,E]

        # attention -> [B,P,latent_dim]
        Z = self.attn(Q, K, V, mask_flat)

        # per-basis, per-channel coefficients
        coeffs = self.coeff_proj(Z)                                                 # [B,P,C]

        # basis at target times -> time-varying predictions
        Phi = self._basis(y_times)                                                  # [B,Ty,P]
        Y   = torch.einsum("btp,bpc->btc", Phi, coeffs)                             # [B,Ty,C]

        if self.residual_cycle:
            yt_phys = y_times * denorm_time_max
            phases_out = torch.floor(yt_phys).long().to(X.device) % self.cycle_length
            c_fut = c_base.gather(1, phases_out.unsqueeze(-1).expand(-1, -1, C))
            Y = Y + c_fut

        return Y
