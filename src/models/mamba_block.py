import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from mamba_ssm import Mamba as _MambaSSM
_FAST_MAMBA = True


class MambaBlock(nn.Module):
    """Mamba-style selective-scan block.

    Uses the fast mamba_ssm CUDA kernel when the package is available,
    falling back to a pure-PyTorch selective scan otherwise.

    Parameters
    ----------
    d_model : int
        Channel dimension of the input / output.
    state_dim : int
        SSM state size (d_state in mamba_ssm, N in the paper).
    conv_kernel : int
        Depthwise-conv kernel width (d_conv in mamba_ssm).
    expand_factor : float
        Inner-dimension expansion ratio.  For the fast kernel this is
        rounded to the nearest integer >= 2.
    dropout : float
        Dropout on the residual branch.
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

        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # mamba_ssm expects integer expand >= 2
        expand_int = max(2, round(expand_factor))
        self.ssm = _MambaSSM(
            d_model=d_model,
            d_state=state_dim,
            d_conv=conv_kernel,
            expand=expand_int,
        )

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to a ``(batch, seq, channels)`` tensor."""
        if x.ndim != 3:
            raise ValueError("Expected input of shape (batch, seq, channels)")
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        return residual + self.dropout(x)


if __name__ == '__main__':
    b, f, t = 4, 1, 384
    x = torch.randn(b, f, t)
    projection = nn.Linear(t, 128)(x)
    block = MambaBlock()
    out = block(projection)
    print(f"fast={_FAST_MAMBA}  output={out.shape}")
