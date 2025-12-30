import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

def _canonical_kernel_size(kernel_size: int) -> int:
    if kernel_size < 1:
        raise ValueError("conv_kernel must be >= 1")
    return kernel_size


def hippo_legS_matrix(
    state_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return HiPPO-LegS initialization matrices (A, B, C)."""
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
    """A minimal Mamba-style block with a selective scan core.

    Parameters
    ----------
    d_model: int
        Number of channels in the model (after the input projection).
    state_dim: int, default=16
        Internal state dimension for the selective scan. Smaller values keep the
        parameter count down.
    conv_kernel: int, default=3
        Depthwise convolution kernel size used as a light-weight local mixer.
    expand_factor: float, default=1.5
        Expansion ratio that controls the inner dimensionality of the block.
    dropout: float, default=0.0
        Dropout probability applied to the residual branch output.
    """

    def __init__(
        self,
        d_model: int = 128,
        *,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand_factor: float = 1.5,
        dropout: float = 0.0,
        input_dim: int = 32,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if expand_factor <= 1.0:
            raise ValueError("expand_factor must be greater than 1.0")

        self.hidden_model = 128
        self.input_dim = input_dim
        self.d_model = d_model
        self.state_dim = state_dim
        self.inner_dim = math.ceil(d_model * expand_factor)
        kernel_size = _canonical_kernel_size(conv_kernel)

        self.norm = nn.LayerNorm(d_model)
        # project from d_model to 3 * inner_dim
        # so that projected.chunk(3, dim=-1) yields tensors with last dim == inner_dim
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

        # HiPPO-LegS initialised SSM parameters (A, B, C).
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
        """Apply the block to a `(batch, seq, channels)` tensor."""
        if x.ndim != 3:
            raise ValueError("Expected input of shape (batch, seq, channels)")

        residual = x
        x = self.norm(x)
        projected = self.in_proj(x)
        data, gate, delta_raw = projected.chunk(3, dim=-1)

        # Depth-wise conv expects (batch, channels, seq)
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
        """Selective scan driven by HiPPO-initialised SSM (A, B, C).
        Uses a zero-order hold discretisation per time step based on a
        data-dependent positive step size ``delta``.
        """
        batch, seq_len, _ = x.shape
        if delta.shape[0] != batch or delta.shape[1] != seq_len or delta.shape[2] != 1:
            raise ValueError("delta must have shape (batch, seq_len, 1)")

        state = x.new_zeros(batch, self.state_dim, 1)
        outputs = x.new_empty(batch, seq_len, self.inner_dim)

        A_expand = self.A.unsqueeze(0).expand(batch, -1, -1)
        B_expand = self.B.unsqueeze(0).expand(batch, -1, -1)
        C_expand = self.C.unsqueeze(0).expand(batch, -1, -1)
        eye = self.eye_state.unsqueeze(0).expand(batch, -1, -1).to(device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            u_t = x[:, t, :]
            delta_t = delta[:, t, :]

            scaled_A = delta_t.view(batch, 1, 1) * A_expand
            A_expm = torch.matrix_exp(scaled_A)
            integral = torch.linalg.solve(A_expand, A_expm - eye)
            B_disc = torch.bmm(integral, B_expand)

            state = torch.bmm(A_expm, state)
            state = state + torch.bmm(B_disc, u_t.unsqueeze(-1))

            outputs[:, t, :] = torch.bmm(C_expand, state).squeeze(-1)

        return outputs
    
if __name__ == '__main__':
    # Example of use (model dim = 128)
    b,f,t = 4,1,384
    x = torch.randn(b,f,t)
    projection = nn.Linear(t, 128)(x)
    mamba_block = MambaBlock()
    out = mamba_block(projection)
    print(out.shape)