"""Shared pytest fixtures and path-bootstrap for the SSM-time-series test suite.

Every test file imports from ``conftest`` automatically via pytest's discovery.
The module guarantees that ``src/`` is on ``sys.path`` before any project
import is attempted, eliminating the need for per-file boilerplate.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path bootstrap – makes ``import models``, ``import dataloaders`` etc. work.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(autouse=True)
def fixed_seed():
    """Reset RNG seeds before every test for reproducibility."""
    set_global_seed(42)
    yield


# ---------------------------------------------------------------------------
# Device fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")   # Tests run on CPU; GPU is optional


# ---------------------------------------------------------------------------
# Tiny synthetic time-series factory
# ---------------------------------------------------------------------------

@pytest.fixture()
def make_batch():
    """Return a factory: make_batch(B, F, T) → (B, F, T) float tensor."""
    def _factory(batch: int = 4, features: int = 1, time: int = 96) -> torch.Tensor:
        return torch.randn(batch, features, time)
    return _factory


@pytest.fixture()
def make_seq():
    """Return a (B, T, F) float tensor – 'sequence-first' convention."""
    def _factory(batch: int = 4, time: int = 96, features: int = 1) -> torch.Tensor:
        return torch.randn(batch, time, features)
    return _factory


@pytest.fixture()
def make_numpy_series():
    """Return a factory: make_numpy_series(n, T) → list of 1-D numpy arrays."""
    def _factory(n: int = 8, length: int = 100) -> List[np.ndarray]:
        rng = np.random.default_rng(42)
        return [rng.standard_normal(length).astype(np.float32) for _ in range(n)]
    return _factory
