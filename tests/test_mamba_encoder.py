import sys
from pathlib import Path
import unittest

import torch

# Ensure the src directory is on the path when running the tests directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssm_time_series.models.mamba_encoder import MambaEncoder  # noqa: E402


class TestMambaEncoder(unittest.TestCase):
    def test_forward_shape(self) -> None:
        encoder = MambaEncoder(embedding_dim=96)
        batch, seq_len = 3, 50
        dummy = torch.randn(batch, seq_len, 384)
        out = encoder(dummy)
        self.assertEqual(out.shape, (batch, 96))

    def test_parameter_count(self) -> None:
        encoder = MambaEncoder(model_dim=96, depth=1, state_dim=8, embedding_dim=64)
        params = encoder.count_parameters()
        self.assertTrue(params < 1_000_000)
        self.assertGreater(params, 0)


if __name__ == "__main__":
    unittest.main()
