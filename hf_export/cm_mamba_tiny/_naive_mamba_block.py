"""Pure-PyTorch Mamba-style block — platform-independent fallback.

This is the naive reference implementation using a sequential selective scan
driven by HiPPO-LegS-initialised SSM matrices.  It runs on any platform
(including Windows) without CUDA extensions.

Public surface
--------------
MambaBlock(d_model, *, state_dim, conv_kernel, expand_factor, dropout)
    – standard ``(batch, seq, channels)`` → ``(batch, seq, channels)`` block.
hippo_legS_matrix(state_dim)
    – returns the HiPPO-LegS (A, B, C) initialisation tensors.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def hippo_legS_matrix(
    state_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return HiPPO-LegS initialisation matrices (A, B, C).

    Parameters
    ----------
    state_dim:
        Dimension of the latent state.  Must be positive.
    dtype:
        Tensor dtype for the output matrices.
    device:
        Target device.  Defaults to CPU.

    Returns
    -------
    A : (state_dim, state_dim)
    B : (state_dim,)
    C : (state_dim,)
    """
    if state_dim <= 0:
        raise ValueError("state_dim must be positive for HiPPO initialisation")
    device = torch.device("cpu") if device is None else device

    A = torch.zeros((state_dim, state_dim), dtype=dtype, device=device)
    for n in range(state_dim):
        coeff_n = math.sqrt(2 * n + 1)
        for m in range(n + 1):
            coeff_m = math.sqrt(2 * m + 1)
            if n == m:
                A[n, m] = -(2 * n + 1)
            else:
                sign = -1.0 if (n - m) % 2 == 0 else 1.0
                A[n, m] = sign * coeff_n * coeff_m

    indices = torch.arange(state_dim, dtype=dtype, device=device)
    B = torch.sqrt(2.0 * indices + 1.0)
    C = B.clone()
    C[1::2] *= -1.0

    return A, B, C


class MambaBlock(nn.Module):
    """Pure-PyTorch Mamba-style block with a selective scan core.

    This implementation is self-contained and runs on any platform.
    On Linux/macOS consider using the ``mamba_block`` module (which wraps the
    CUDA-optimised ``mamba-ssm`` library) instead.

    Parameters
    ----------
    d_model:
        Number of channels after the input projection.
    state_dim:
        Internal state dimension for the selective scan.
    conv_kernel:
        Depthwise convolution kernel size used as a lightweight local mixer.
    expand_factor:
        Expansion ratio controlling the inner dimensionality of the block.
        Must be > 1.
    dropout:
        Dropout probability applied to the residual branch output.
    input_dim:
        Unused – kept for API compatibility with callers that pass it.
    """

    def __init__(
        self,
        d_model: int = 128,
        *,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand_factor: float = 1.5,
        dropout: float = 0.0,
        input_dim: int = 32,  # API compat – not used internally
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if expand_factor <= 1.0:
            raise ValueError("expand_factor must be greater than 1.0")

        self.d_model = d_model
        self.state_dim = state_dim
        self.inner_dim = math.ceil(d_model * expand_factor)
        kernel_size = max(1, conv_kernel)

        self.norm = nn.LayerNorm(d_model)
        # Input projection: d_model → 3 × inner_dim  (data, gate, delta)
        self.in_proj = nn.Linear(d_model, self.inner_dim * 3, bias=False)
        self.depthwise_conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=self.inner_dim,
            bias=False,
        )
        self.conv_activation = nn.SiLU()

        # HiPPO-LegS initialised SSM parameters (A, B, C)
        A_init, B_init, C_init = hippo_legS_matrix(state_dim)
        scale = 1.0 / math.sqrt(self.inner_dim)
        self.A = nn.Parameter(A_init)
        self.B = nn.Parameter(B_init.unsqueeze(1).repeat(1, self.inner_dim) * scale)
        self.C = nn.Parameter(C_init.unsqueeze(0).repeat(self.inner_dim, 1) * scale)

        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.delta_min = 1e-4
        self.delta_max = 3.0
        self.register_buffer("eye_state", torch.eye(state_dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to a ``(batch, seq, channels)`` tensor."""
        if x.ndim != 3:
            raise ValueError("Expected input of shape (batch, seq, channels)")

        residual = x
        x = self.norm(x)
        projected = self.in_proj(x)
        data, gate, delta_raw = projected.chunk(3, dim=-1)

        # Depthwise conv expects (batch, channels, seq)
        data_conv = data.transpose(1, 2)
        data_conv = self.depthwise_conv(data_conv)
        seq_len = x.size(1)
        if data_conv.size(-1) > seq_len:
            data_conv = data_conv[..., :seq_len]
        data_conv = data_conv.transpose(1, 2)
        data_conv = self.conv_activation(data_conv)

        delta = F.softplus(delta_raw).mean(dim=-1, keepdim=True) + self.delta_min
        delta = delta.clamp(max=self.delta_max)

        selective = self._selective_scan(data_conv, delta)
        gated = torch.sigmoid(gate) * selective
        out = self.out_proj(gated)
        out = self.dropout(out)
        return residual + out

    def _selective_scan(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Selective scan with HiPPO-initialised SSM (A, B, C).

        Uses a zero-order-hold discretisation driven by a data-dependent step
        size *delta*.  Discretisation (``matrix_exp`` + ``linalg.solve``) is
        vectorised over the whole sequence before the recurrent pass.
        """
        batch, seq_len, _ = x.shape
        if delta.shape != (batch, seq_len, 1):
            raise ValueError("delta must have shape (batch, seq_len, 1)")

        # Vectorised discretisation: compute A_disc and B_disc for all t at once.
        # delta: (B, T, 1) → (B, T, 1, 1)
        # self.A: (state, state)
        # scaled_A: (B, T, state, state)
        scaled_A = delta.unsqueeze(-1) * self.A
        A_disc = torch.matrix_exp(scaled_A)  # (B, T, state, state)

        integral = torch.linalg.solve(self.A, A_disc - self.eye_state)
        B_disc = torch.matmul(integral, self.B)  # (B, T, state, inner)

        # Recurrent pass (only matrix multiplications remain in the loop).
        state = x.new_zeros(batch, self.state_dim, 1)
        outputs = x.new_empty(batch, seq_len, self.inner_dim)

        for t in range(seq_len):
            u_t = x[:, t, :].unsqueeze(-1)  # (B, inner, 1)
            state = torch.bmm(A_disc[:, t], state) + torch.bmm(B_disc[:, t], u_t)
            outputs[:, t, :] = torch.matmul(self.C, state).squeeze(-1)

        return outputs
