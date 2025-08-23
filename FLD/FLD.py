from __future__ import annotations


import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FLDAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        parameters: int,
        latent_dim: int = 16,
        embed_dim: int = 16,
        num_heads: int = 2,
        shared_out: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_time = embed_dim
        self.embed_time_k = embed_dim // num_heads
        self.h = num_heads
        self.nhidden = latent_dim
        if shared_out:
            self.out = nn.Sequential(nn.Linear(input_dim * num_heads, latent_dim))
        else:
            self.out = nn.Parameter(
                torch.randn(1, parameters, input_dim * num_heads, latent_dim)
            )
            self.out_bias = nn.Parameter(torch.zeros(1, parameters, latent_dim))
        self.shared = shared_out

        self.query_map = nn.Linear(embed_dim, embed_dim)
        self.key_map = nn.Linear(embed_dim, embed_dim)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,P,T]
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)          # [B,H,P,T,D]
        if mask is not None:
            # mask: [B,T,D] -> broadcast to [B,H,P,T,D]
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)                                    # over T
        return torch.sum(p_attn * value.unsqueeze(1).unsqueeze(1), -2), p_attn

    def forward(
        self,
        query: torch.Tensor,  # [B,P,E]
        key: torch.Tensor,    # [B,T,E]
        value: torch.Tensor,  # [B,T,D]
        mask: torch.Tensor | None = None,  # [B,T,D]
    ) -> torch.Tensor:
        batch, _, dim = value.size()
        query = self.query_map(query)  # [B,P,E]
        key = self.key_map(key)        # [B,T,E]
        query, key = [
            x.view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for x in (query, key)
        ]  # query:[B,H,P,d_k], key:[B,H,T,d_k]

        x, _ = self.attention(query, key, value, mask)  # x:[B,H,P,D]
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)  # [B,P,H*D]
        if self.shared:
            return self.out(x)  # [B,P,L]
        else:
            x = x.unsqueeze(-2) @ self.out  # [B,P,1,L]
            x = x.squeeze(-2) + self.out_bias
            return x  # [B,P,L]


class FLD(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        embed_dim_per_head: int,
        num_heads: int,
        function: str,
        device: torch.device,
        depth: int = 1,
        hidden_dim: int | None = None,
        shared_out_for_attn: bool = True,
    ) -> None:
        super().__init__()
        if function == "C":
            P = 1
        elif function == "L":
            P = 2
        elif function == "Q":
            P = 3
        elif function == "S":
            P = 4
        else:
            raise ValueError("function must be one of {'C','L','Q','S'}")
        self.F = function
        embed_dim = embed_dim_per_head * num_heads
        self.attn = FLDAttention(
            input_dim=2 * input_dim,   # concat X and M
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            parameters=P,
            shared_out=shared_out_for_attn,
        )
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.time_embedding = nn.Linear(1, embed_dim)
        self.query = nn.Parameter(torch.randn(1, P, embed_dim))
        if hidden_dim is None:
            hidden_dim = latent_dim
        if depth > 0:
            decoder = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
            for _ in range(depth - 1):
                decoder += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            decoder.append(nn.Linear(hidden_dim, input_dim))
        else:
            decoder = [nn.Linear(latent_dim, input_dim)]
        self.out = nn.Sequential(*decoder)
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim

    def learn_time_embedding(self, tt: torch.Tensor) -> torch.Tensor:
        tt = tt.to(self.device).unsqueeze(-1)  # [B,L,1]
        out = self.time_embedding(tt)          # [B,L,E]
        inds = list(range(0, self.embed_dim, self.num_heads))
        out[:, :, inds] = torch.sin(out[:, :, inds])
        return out

    @staticmethod
    def _time_last(x: torch.Tensor, D: int) -> torch.Tensor:
        """Ensure [B,L,D]. Accepts [B,D,L] and permutes."""
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D tensor, got {x.shape}")
        if x.shape[-1] == D:
            return x
        if x.shape[1] == D:
            return x.transpose(1, 2).contiguous()
        raise ValueError(f"Cannot infer feature axis from shape {x.shape} with D={D}")

    def forward(
        self,
        timesteps: torch.Tensor,  # [B,L]
        X: torch.Tensor,          # [B,L,D] or [B,D,L]
        M: torch.Tensor,          # [B,L,D] or [B,D,L]
        y_time_steps: torch.Tensor,  # [B,Ty]
    ) -> torch.Tensor:
        X = self._time_last(X, self.input_dim)
        M = self._time_last(M, self.input_dim)

        key = self.learn_time_embedding(timesteps)  # [B,L,E]
        Xcat = torch.cat((X, M), -1)               # [B,L,2D]
        Mcat = torch.cat((M, M), -1)               # [B,L,2D]

        B = X.shape[0]
        query = self.query.expand(B, -1, -1)       # [B,P,E]
        coeffs = self.attn(query, key, Xcat, Mcat) # [B,P,L*]

        if self.F == "C":
            x = coeffs[:, 0, :].unsqueeze(-2).repeat_interleave(y_time_steps.size(1), -2)
        elif self.F == "L":
            x = coeffs[:, 0, :].unsqueeze(-2) + (
                y_time_steps.unsqueeze(-1) @ coeffs[:, 1, :].unsqueeze(-2)
            )
        elif self.F == "Q":
            x = (
                coeffs[:, 0, :].unsqueeze(-2)
                + (y_time_steps.unsqueeze(-1) @ coeffs[:, 1, :].unsqueeze(-2))
                + ((y_time_steps.unsqueeze(-1) ** 2) @ coeffs[:, 2, :].unsqueeze(-2))
            )
        elif self.F == "S":
            x = (
                coeffs[:, 0].unsqueeze(-2)
                * torch.sin(
                    (coeffs[:, 1].unsqueeze(-2) * y_time_steps.unsqueeze(-1))
                    + coeffs[:, 2].unsqueeze(-2)
                )
            ) + coeffs[:, 3].unsqueeze(-2)
        else:
            raise RuntimeError("Invalid function type.")
        return self.out(x)  # [B,Ty,D]
