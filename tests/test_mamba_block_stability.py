import sys
from pathlib import Path
import unittest

import torch

# Ensure the src directory is on the path when running the tests directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.mamba_block import MambaBlock  # noqa: E402


class TestMambaBlockStability(unittest.TestCase):
    def test_forward_backward_finite(self) -> None:
        torch.manual_seed(0)
        block = MambaBlock(d_model=64, state_dim=16, conv_kernel=3, expand_factor=1.5, dropout=0.0)
        block.train()

        x = torch.randn(2, 128, 64, requires_grad=True)
        y = block(x)
        self.assertTrue(torch.isfinite(y).all().item())

        loss = (y ** 2).mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())


if __name__ == "__main__":
    unittest.main()
