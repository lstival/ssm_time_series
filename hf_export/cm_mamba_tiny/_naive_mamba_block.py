"""Pure-PyTorch Mamba block with mamba_ssm-compatible parameter layout.

Weight keys are identical to mamba_ssm.modules.mamba_simple.Mamba so that
checkpoints trained with the CUDA kernel load without any conversion:

    blocks.N.norm.{weight,bias}
    blocks.N.mamba.in_proj.weight
    blocks.N.mamba.conv1d.{weight,bias}
    blocks.N.mamba.x_proj.weight
    blocks.N.mamba.dt_proj.{weight,bias}
    blocks.N.mamba.A_log
    blocks.N.mamba.D
    blocks.N.mamba.out_proj.weight

Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MambaCore(nn.Module):
    """Selective SSM with parameter layout matching mamba_ssm.Mamba."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        # Projections — names must match mamba_ssm exactly
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            bias=True, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # SSM parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)  →  (B, L, d_model)."""
        B, L, _ = x.shape
        d_state = self.d_state

        # Expand and split
        xz = self.in_proj(x)                           # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                  # each (B, L, d_inner)

        # Causal conv1d
        x_conv = x_in.permute(0, 2, 1)                 # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :L]           # trim padding
        x_conv = F.silu(x_conv).permute(0, 2, 1)       # (B, L, d_inner)

        # Compute dt, B, C
        xbc = self.x_proj(x_conv)                      # (B, L, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = torch.split(
            xbc, [self.dt_rank, d_state, d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))               # (B, L, d_inner)

        # Discretise A and B
        A = -torch.exp(self.A_log.float())              # (d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A)            # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)     # (B, L, d_inner, d_state)

        # Sequential scan
        h = x_in.new_zeros(B, self.d_inner, d_state)
        ys = []
        for t in range(L):
            u_t = x_conv[:, t, :].unsqueeze(-1)         # (B, d_inner, 1)
            h = dA[:, t] * h + dB[:, t] * u_t
            y_t = (h * C_ssm[:, t].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                     # (B, L, d_inner)
        y = y + x_conv * self.D                         # skip connection
        y = y * F.silu(z)                               # gating
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Residual Mamba block — matches mamba_ssm key layout.

    state dict keys per block:
        norm.weight, norm.bias
        mamba.in_proj.weight
        mamba.conv1d.weight, mamba.conv1d.bias
        mamba.x_proj.weight
        mamba.dt_proj.weight, mamba.dt_proj.bias
        mamba.A_log, mamba.D
        mamba.out_proj.weight
    """

    def __init__(
        self,
        d_model: int = 128,
        *,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand_factor: float = 2.0,
        dropout: float = 0.0,
        input_dim: int = 32,  # API compat — unused
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = _MambaCore(
            d_model=d_model,
            d_state=state_dim,
            d_conv=max(1, conv_kernel),
            expand=max(1, round(expand_factor)),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected (batch, seq, channels)")
        return x + self.dropout(self.mamba(self.norm(x)))
