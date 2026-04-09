"""Lightweight Mamba-style encoder for 384-dim time series inputs.

This module implements a compact encoder inspired by the Mamba state-space
architecture (https://arxiv.org/abs/2312.00752). It is designed to keep the
parameter count low while still capturing temporal structure in fixed-length
(time-major) sequences. The core building block is a simplified selective scan
that mixes information along the sequence using inexpensive operations.

"""

from __future__ import annotations
from typing import Literal, Optional
import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

try:
    from utils import time_series_2_recurrence_plot, recurrence_plot_gpu
    from mamba_block import MambaBlock
except:
    from .utils import time_series_2_recurrence_plot, recurrence_plot_gpu
    from .mamba_block import MambaBlock

Pooling = Literal["mean", "last", "cls"]
RpMvStrategy = Literal["per_channel", "mean", "pca", "joint"]
ReprType = Literal["rp", "gasf", "mtf", "stft"]


class Tokenizer:
    """
    Convert a (B, T, F) sequence into a sequence of tokens (B, N, F) by
    windowing along the time dimension.

    Usage:
        tk = Tokenizer(token_size=32, stride=None, method="values", pad=True)
        tokens = tk(x)  # where x is (B, T, F) or (B, F, T) — note: this implementation
                        # will swap axes at the start like the original function.

    Parameters
    - token_size: length of each token window (number of timesteps per token)
    - stride: step between token starts. If None, defaults to token_size (non-overlap).
    - method: how to aggregate values inside a token: "mean", "max", "first", or
              "values". "values" returns the raw window values with shape
              (batch, n_tokens, token_size, features).
    - pad: whether to pad the end with zeros so the final token is full length.
    """

    def __init__(
        self,
        *,
        token_size: int = 32,
        stride: Optional[int] = None,
        method: Literal["mean", "max", "first", "values"] = "values",
        pad: bool = True,
    ) -> None:
        if token_size <= 0:
            raise ValueError("token_size must be positive")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive")
        if method not in ("mean", "max", "first", "values"):
            raise ValueError(f"Unknown tokenization method: {method}")

        self.token_size = token_size
        self.stride = stride if stride is not None else token_size
        self.method = method
        self.pad = pad

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("tokenize_sequence expects input of shape (batch, time, features)")

        B, T, F = x.shape
        token_size = self.token_size
        stride = self.stride

        # compute required padding so that unfold will include the final (possibly partial) block
        if T <= token_size:
            n_steps = 1
            required_total = token_size
        else:
            n_steps = math.ceil((T - token_size) / stride) + 1
            required_total = token_size + stride * (n_steps - 1)
        pad_len = max(0, required_total - T)

        if pad_len > 0:
            if self.pad:
                pad_tensor = x.new_zeros((B, pad_len, F))
                x_padded = torch.cat([x, pad_tensor], dim=1)
            else:
                raise ValueError(
                    "Sequence length is shorter than token_size and pad=False; cannot form full tokens"
                )
        else:
            x_padded = x

        # Create the patches for the number of tokens: (B, n_tokens, F, token_size)
        patches = x_padded.unfold(dimension=1, size=token_size, step=stride).contiguous()
        # Transpose to (B, n_tokens, token_size, F)
        patches = patches.transpose(2, 3)

        if self.method == "values":
            tokens = patches
        elif self.method == "mean":
            tokens = patches.mean(dim=2)
        elif self.method == "max":
            tokens, _ = patches.max(dim=2)
        elif self.method == "first":
            tokens = patches[:, :, 0, :]
        else:
            # should be unreachable due to check in __init__
            raise ValueError(f"Unknown tokenization method: {self.method}")

        return tokens

def tokenize_sequence(
    x: torch.Tensor,
    *,
    token_size: int = 32,
    stride: Optional[int] = None,
    method: Literal["mean", "max", "first", "values"] = "values",
    pad: bool = True,
) -> torch.Tensor:
    return Tokenizer(token_size=token_size, stride=stride, method=method, pad=pad)(x)

class _InputConv(nn.Module):
    """
    Accept input of shape (B, windows, window_size, window_size) and produce
    (B, windows, out_dim) by applying a Conv2d over each window and collapsing
    the spatial dimensions to a single vector per window.

    token_len here is the spatial window_size (H = W = token_len).
    """
    def __init__(self, token_len: int, out_dim: int):
        super().__init__()
        if token_len <= 0:
            raise ValueError("token_len must be positive")
        self.token_len = token_len
        self.out_dim = out_dim
        # single-channel input windows -> produce out_dim channels, kernel covers whole window
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_dim, kernel_size=(token_len, token_len), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # allow array-like inputs
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        residual = x
        # expect (B, windows, H, W)
        if x.ndim != 4:
            raise ValueError("Expected input of shape (B, windows, window_size, window_size)")

        b, windows, h, w = x.shape
        if h != self.token_len or w != self.token_len:
            raise ValueError(f"Window size mismatch: got {h}x{w}, expected {self.token_len}x{self.token_len}")

        # treat each window as a single-channel image: (B*windows, 1, H, W)
        x = x.view(b * windows, 1, h, w)
        x = self.conv(x)  # -> (B*windows, out_dim, 1, 1)
        x = x.view(b, windows, -1)  # -> (B, windows, out_dim)
        return x

class MambaVisualEncoder(nn.Module):
    """Compact Mamba-style encoder for 384-dim time series inputs."""

    def __init__(
        self,
        *,
        input_dim: int = 32,
        model_dim: int = 768,
        depth: int = 6,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand_factor: float = 1.5,
        embedding_dim: int = 128,
        pooling: Pooling = "mean",
        dropout: float = 0.05,
        rp_mode: str = "correct",
        rp_mv_strategy: RpMvStrategy = "joint",
        repr_type: ReprType = "rp",
        use_gpu_rp: bool = False,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.pooling: Pooling = pooling
        self.rp_mode = rp_mode
        self.rp_mv_strategy = rp_mv_strategy
        self.repr_type = repr_type
        self.use_gpu_rp = use_gpu_rp

        self.input_proj = _InputConv(token_len=self.input_dim, out_dim=model_dim)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    model_dim,
                    state_dim=state_dim,
                    conv_kernel=conv_kernel,
                    expand_factor=expand_factor,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, embedding_dim, bias=False)

    # ------------------------------------------------------------------
    # Multivariate strategy helpers
    # ------------------------------------------------------------------

    def _apply_mv_strategy(self, arr: np.ndarray) -> np.ndarray:
        """Reduce (N, F, L) array to (N, 1, L) according to rp_mv_strategy."""
        N, F, L = arr.shape
        if F <= 1 or self.rp_mv_strategy == "per_channel":
            return arr  # handled downstream (per-channel RP then avg)

        if self.rp_mv_strategy == "mean":
            return arr.mean(axis=1, keepdims=True)  # (N, 1, L)

        if self.rp_mv_strategy == "pca":
            try:
                from sklearn.decomposition import PCA
            except ImportError:
                # fallback to mean if sklearn unavailable
                return arr.mean(axis=1, keepdims=True)
            arr_t = arr.transpose(0, 2, 1).reshape(N * L, F)  # (N*L, F)
            pca = PCA(n_components=1)
            proj = pca.fit_transform(arr_t).reshape(N, L)  # (N, L)
            return proj[:, np.newaxis, :]  # (N, 1, L)

        if self.rp_mv_strategy == "joint":
            # Global L2 Distance RP: multivariate recurrence in F-dim state space
            # Each timestep is a vector in R^F (F = n_channels, e.g., 321)
            # RP[i,j] = ||x_i - x_j||_2 / max_distance (continuous, normalized to [0,1])
            arr_t = arr.transpose(0, 2, 1)  # (N, L, F)
            imgs = np.zeros((N, L, L), dtype=np.float32)
            for n in range(N):
                x = arr_t[n]  # (L, F) — L timesteps, each a F-dimensional vector
                diffs = x[:, None, :] - x[None, :, :]  # (L, L, F) — pairwise differences
                dists = np.sqrt((diffs ** 2).sum(-1))   # (L, L) — L2 distances in R^F
                # Normalize to [0, 1] by dividing by max distance
                dists_max = dists.max()
                if dists_max > 0:
                    dists = dists / dists_max
                imgs[n] = dists
            return imgs  # (N, L, L) — continuous RP with values in [0,1]

        raise ValueError(f"Unknown rp_mv_strategy: {self.rp_mv_strategy}")

    # ------------------------------------------------------------------
    # Visual representation helpers
    # ------------------------------------------------------------------

    def _compute_repr(self, arr: np.ndarray) -> np.ndarray:
        """
        Compute the 2-D visual representation from (N, F, L) or (N, L, L) array.
        Returns (N, L, L) float32.
        """
        # Joint MV strategy already returns (N, L, L)
        if arr.ndim == 3 and arr.shape[1] == arr.shape[2] and self.rp_mv_strategy == "joint":
            return arr

        # At this point arr is (N, F, L) with F already reduced (or per_channel)
        if self.repr_type == "rp":
            # Ensure no infinite values before RP
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            x = time_series_2_recurrence_plot(arr)
            # multichannel case returns (N, F, L, L) → average
            if x.ndim == 4:
                x = x.mean(axis=1)
            return x.astype(np.float32)

        # For non-RP types, collapse to 1 channel if needed
        if arr.ndim == 3 and arr.shape[1] > 1:
            arr = arr.mean(axis=1, keepdims=True)  # (N, 1, L)
        arr_2d = arr[:, 0, :]  # (N, L)
        L = arr_2d.shape[1]

        # Stabilize input
        arr_2d = np.nan_to_num(arr_2d, nan=0.0, posinf=100.0, neginf=-100.0)

        if self.repr_type == "gasf":
            from pyts.image import GramianAngularField
            # GASF requires scaling to [-1, 1] for acos
            # We use a robust scaler or simple min-max
            mins = arr_2d.min(axis=1, keepdims=True)
            maxs = arr_2d.max(axis=1, keepdims=True)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            arr_scaled = 2 * (arr_2d - mins) / ranges - 1
            arr_scaled = np.clip(arr_scaled, -1.0 + 1e-6, 1.0 - 1e-6)

            gaf = GramianAngularField(method="summation")
            imgs = gaf.fit_transform(arr_scaled)
            return imgs.astype(np.float32)

        if self.repr_type == "mtf":
            from pyts.image import MarkovTransitionField
            # MTF works best on [0, 1]
            mins = arr_2d.min(axis=1, keepdims=True)
            maxs = arr_2d.max(axis=1, keepdims=True)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            arr_scaled = (arr_2d - mins) / ranges
            
            mtf = MarkovTransitionField(n_bins=4)
            imgs = mtf.fit_transform(arr_scaled)
            return imgs.astype(np.float32)

        if self.repr_type == "stft":
            x_t = torch.from_numpy(arr_2d).float()
            n_fft = max(4, min(L, 16))
            hop = max(1, n_fft // 2)
            stft = torch.stft(x_t, n_fft=n_fft, hop_length=hop, return_complex=True)
            mag = stft.abs()  # (N, freqs, time)
            mag_4d = mag.unsqueeze(1)  # (N, 1, freqs, time)
            mag_r = F.interpolate(mag_4d, size=(L, L), mode="bilinear", align_corners=False)
            # Ensure no NaNs in STFT output
            res = mag_r.squeeze(1).numpy().astype(np.float32)
            return np.nan_to_num(res)

        raise ValueError(f"Unknown repr_type: {self.repr_type}")

    # ------------------------------------------------------------------
    # Main image-building entry point
    # ------------------------------------------------------------------

    def _time_series_2_image(self, ts):
        """
        ts: torch.Tensor (N, F, window_len) or numpy array of same shape.
        Returns numpy (N, window_len, window_len) float32.
        """
        if isinstance(ts, torch.Tensor):
            arr = ts.detach().cpu().numpy()
        else:
            arr = np.asarray(ts, dtype=np.float32)

        # Step 1 – apply multivariate strategy (may return (N,L,L) for joint)
        arr = self._apply_mv_strategy(arr)

        # Step 2 – compute visual representation
        x = self._compute_repr(arr)

        # Step 3 – apply rp_mode perturbations (shuffled / random)
        if self.rp_mode == "shuffled":
            orig_shape = x.shape
            L = orig_shape[-1]
            x_flat = x.reshape(-1, L * L)
            idx = np.argsort(np.random.rand(x_flat.shape[0], L * L), axis=1)
            x_flat = x_flat[np.arange(x_flat.shape[0])[:, None], idx]
            x = x_flat.reshape(orig_shape)
        elif self.rp_mode == "random":
            x = np.random.normal(0, 1, size=x.shape).astype(np.float32)

        return x

    def _time_series_2_image_gpu(self, ts: torch.Tensor) -> torch.Tensor:
        """GPU-native recurrence plot for ``rp_mode='correct'`` and
        ``rp_mv_strategy`` in ``{'per_channel', 'mean'}``.

        Falls back to the CPU path for other modes / repr_types so that
        ablations (GASF, MTF, STFT, shuffled RP, …) are unaffected.

        Args:
            ts: ``(N, F, L)`` tensor already on the target device.

        Returns:
            ``(N, L, L)`` float32 tensor on the same device.
        """
        if (self.rp_mode != "correct"
                or self.repr_type != "rp"
                or self.rp_mv_strategy in ("pca", "joint")):
            # pca/joint need sklearn/numpy; non-RP repr_types also need CPU path
            cpu_result = self._time_series_2_image(ts)
            if isinstance(cpu_result, np.ndarray):
                return torch.from_numpy(cpu_result).float().to(ts.device)
            return cpu_result

        N, F, L = ts.shape
        if self.rp_mv_strategy == "mean" and F > 1:
            x = ts.mean(dim=1)                          # (N, L)
            return recurrence_plot_gpu(x)               # (N, L, L)
        else:
            # per_channel (default): RP per channel, then average
            x = ts.reshape(N * F, L)                    # (N*F, L)
            rp = recurrence_plot_gpu(x)                 # (N*F, L, L)
            rp = rp.reshape(N, F, L, L)
            return rp.mean(dim=1) if F > 1 else rp[:, 0]   # (N, L, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_sequence(x)
        pooled = self._pool_sequence(features, x)
        return self.output_proj(pooled)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sequence of hidden states before pooling."""
        # MambaEncoder's tokenizer does swapaxes(1,2) internally to convert (B,F,T)→(B,T,F).
        # We must do the same so both encoders handle the same (B,channels,time) input format.
        x = x.swapaxes(1, 2)  # (B, F, T) → (B, T, F)
        tokens = self.tokenizer(x) # (B, windows, window_len, F)
        B, windows, window_len, F = tokens.shape

        tokens_for_rp = tokens.permute(0, 1, 3, 2).reshape(B * windows, F, window_len)

        if self.use_gpu_rp:
            img_from_patches = self._time_series_2_image_gpu(tokens_for_rp)
        else:
            img_from_patches = self._time_series_2_image(tokens_for_rp)

        if isinstance(img_from_patches, np.ndarray):
            img_from_patches = torch.from_numpy(img_from_patches).float().to(x.device)

        # If multichannel (F > 1), we need to reduce to 1 channel for _InputConv
        if img_from_patches.ndim == 4:
            img_from_patches = img_from_patches.mean(dim=1)
        
        # Reshape to (B, windows, L, L) for _InputConv
        img_from_patches = img_from_patches.view(B, windows, window_len, window_len)

        x = self.input_proj(img_from_patches) 

        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
    
    def tokenizer(self, x):
        tokens = tokenize_sequence(x, token_size=self.input_dim)
        return tokens

    def _pool_sequence(self, hidden: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        if self.pooling == "last":
            return hidden[:, -1, :]
        if self.pooling == "cls":
            # Expect the first timestep to act as a CLS token; fall back to mean if seq len == 1
            if hidden.size(1) == 1:
                return hidden[:, 0, :]
            return hidden[:, 0, :]
        raise ValueError(f"Unknown pooling mode: {self.pooling}")

def create_default_encoder(**overrides: object) -> MambaVisualEncoder:
    """Factory that mirrors the defaults while allowing overrides."""
    return MambaVisualEncoder(**overrides)


# =============================================================================
# Ablation H — Visual Encoder Architecture for Recurrence Plots
# =============================================================================
# Common interface: forward(rp: Tensor[B, P, l, l]) -> Tensor[B, P, d]
#
# Four variants with distinct inductive biases:
#   1. CNNVisualEncoder      — local spatial filters
#   2. FlattenMambaEncoder   — 1-D scan over P windows (Option A)
#   3. RPSS2DEncoder(n=2)   — semantic 2-D dual scan (RP-SS2D)
#   4. RPSS2DEncoder(n=4)   — full VMamba-style 4-scan (SS2D)
# =============================================================================


class BaseVisualEncoder(nn.Module):
    """Abstract base — common (B, P, l, l) → (B, P, d) interface."""

    def forward(self, rp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rp: Recurrence plot tensor of shape (B, P, l, l).
        Returns:
            Token embeddings of shape (B, P, d).
        """
        raise NotImplementedError


# ── Variant 1: CNN Baseline ───────────────────────────────────────────────────

class CNNVisualEncoder(BaseVisualEncoder):
    """Two Conv2D layers + AdaptiveAvgPool + Linear.

    Extracts local spatial features via a receptive field that, with two 3×3
    layers, already covers the full matrix for l ∈ {16, 32}.

    Architecture:
        Conv2d(1→32, k=3, pad=1) → ReLU
        Conv2d(32→64, k=3, pad=1) → ReLU
        AdaptiveAvgPool2d(4, 4)
        Flatten → Linear(64*4*4, d_model)
    """

    def __init__(self, patch_len: int, d_model: int = 128) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(64 * 4 * 4, d_model)

    def forward(self, rp: torch.Tensor) -> torch.Tensor:
        B, P, l, _ = rp.shape
        # (B*P, 1, l, l)
        x = rp.reshape(B * P, 1, l, l)
        x = self.convs(x)                  # (B*P, 64, 4, 4)
        x = x.flatten(1)                   # (B*P, 64*4*4)
        x = self.proj(x)                   # (B*P, d)
        return x.view(B, P, self.d_model)  # (B, P, d)


# ── Variant 2: Flatten + Mamba 1D ─────────────────────────────────────────────

class FlattenMambaEncoder(BaseVisualEncoder):
    """Project each flat RP to d_model, then run Mamba over the P-window sequence.

    Option A (approved): sequence length = P (one token per RP window).
    Each token is a d-dimensional summary of the full flattened RP patch.

    Optionally supports *diagonal ordering* (order pixels by i+j index) for a
    second ablation run to check spatial-order sensitivity.

    Architecture:
        Flatten(l, l) → Linear(l², d_model)
        MambaBlock × n_layers  [seq_len = P]
        LayerNorm
    """

    def __init__(
        self,
        patch_len: int,
        d_model: int = 128,
        n_layers: int = 2,
        diagonal_order: bool = False,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.diagonal_order = diagonal_order

        if diagonal_order:
            # Pre-compute anti-diagonal permutation indices (row-major → diagonal)
            l = patch_len
            idx = sorted(range(l * l), key=lambda k: (k // l) + (k % l))
            self.register_buffer("diag_idx", torch.tensor(idx, dtype=torch.long))

        self.proj = nn.Linear(patch_len * patch_len, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, state_dim=16, conv_kernel=3, expand_factor=2.0)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, rp: torch.Tensor) -> torch.Tensor:
        B, P, l, _ = rp.shape
        # Flatten each RP patch: (B, P, l²)
        x = rp.reshape(B, P, l * l)
        if self.diagonal_order:
            x = x[:, :, self.diag_idx]  # reorder pixels by diagonal
        x = self.proj(x)               # (B, P, d)
        for block in self.blocks:
            x = block(x)               # (B, P, d)
        return self.norm(x)            # (B, P, d)


# ── Variants 3 & 4: RP-SS2D (2-scan) / SS2D full (4-scan) ───────────────────

class RPSS2DEncoder(BaseVisualEncoder):
    """2-D-aware semantic scan encoder for Recurrence Plots.

    The RP is patchified into (l/p)² = 64 sub-tokens (p = max(2, l//8)) so that
    the number of tokens is *constant* across l ∈ {16, 32, 64}. Each sub-token is
    projected to d_model.

    Two (RP-SS2D, n_scans=2) or four (SS2D full, n_scans=4) Mamba streams are run:
      Scan A — row-major (captures "profile of a reference time point")
      Scan B — column-major / transposed (captures "recurrence of a fixed delay")
      [Scan C — row-major backward ]  (only for n_scans=4)
      [Scan D — column-major backward]  (only for n_scans=4)

    Cross-merge: element-wise sum of all scan outputs.
    GlobalAvgPool over spatial dimension → (B, P, d).

    Justification for 2 scans: RP symmetry means R_{i,j}=R_{j,i}, so reverse
    scans are semantically redundant.
    """

    def __init__(
        self,
        patch_len: int,
        d_model: int = 128,
        n_scans: int = 2,
        n_layers: int = 2,
    ) -> None:
        if n_scans not in (2, 4):
            raise ValueError("n_scans must be 2 or 4")
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_scans = n_scans

        # Internal patch size: p = max(2, l//8) → 64 tokens always
        p = max(2, patch_len // 8)
        n_tokens_side = patch_len // p  # e.g. 16//2=8 → 8×8=64 tokens
        if patch_len % p != 0:
            raise ValueError(
                f"patch_len={patch_len} must be divisible by internal patch size p={p}"
            )
        self.p = p
        self.n_tokens_side = n_tokens_side
        self.n_tokens = n_tokens_side * n_tokens_side  # =64

        # Sub-patch projection
        self.patch_proj = nn.Linear(p * p, d_model)

        # One independent Mamba stack per scan direction
        def _make_mamba():
            return nn.ModuleList([
                MambaBlock(d_model, state_dim=16, conv_kernel=3, expand_factor=2.0)
                for _ in range(n_layers)
            ])

        self.mamba_A = _make_mamba()  # row-major forward
        self.mamba_B = _make_mamba()  # col-major forward
        if n_scans == 4:
            self.mamba_C = _make_mamba()  # row-major backward
            self.mamba_D = _make_mamba()  # col-major backward

        self.norm = nn.LayerNorm(d_model)

    # ---- patchify helper ----
    def _patchify(self, rp: torch.Tensor) -> torch.Tensor:
        """
        rp: (N, l, l)  →  tokens: (N, T, p²)   where T = (l/p)²
        Uses unfold for efficiency.
        """
        N, l, _ = rp.shape
        p = self.p
        # (N, l/p, l/p, p, p) via unfold twice
        x = rp.unfold(1, p, p).unfold(2, p, p)   # (N, n_t, n_t, p, p)
        n_t = self.n_tokens_side
        x = x.contiguous().view(N, n_t * n_t, p * p)  # (N, T, p²)
        return x

    def _run_mamba(self, x: torch.Tensor, mamba_list: nn.ModuleList) -> torch.Tensor:
        for block in mamba_list:
            x = block(x)
        return x

    def forward(self, rp: torch.Tensor) -> torch.Tensor:
        B, P, l, _ = rp.shape
        N = B * P
        T = self.n_tokens      # 64
        n_t = self.n_tokens_side

        # ── Patchify: (B*P, T, p²) → project → (B*P, T, d) ──────────────────
        x_flat = rp.reshape(N, l, l)
        tokens = self._patchify(x_flat)            # (N, T, p²)
        tokens = self.patch_proj(tokens)           # (N, T, d)

        # ── Scan A: row-major forward ─────────────────────────────────────────
        # tokens already in row-major order from patchify
        out_A = self._run_mamba(tokens, self.mamba_A)  # (N, T, d)

        # ── Scan B: column-major forward ──────────────────────────────────────
        # Logical reshape: (N, n_t, n_t, d) → transpose H,W → flatten
        tokens_2d = tokens.view(N, n_t, n_t, self.d_model)
        tokens_col = tokens_2d.transpose(1, 2).reshape(N, T, self.d_model)
        out_B_raw = self._run_mamba(tokens_col, self.mamba_B)  # (N, T, d)
        # Un-transpose to original spatial order
        out_B = out_B_raw.view(N, n_t, n_t, self.d_model).transpose(1, 2).reshape(N, T, self.d_model)

        merge = out_A + out_B  # (N, T, d)

        if self.n_scans == 4:
            # ── Scan C: row-major backward ────────────────────────────────────
            tokens_rev = tokens.flip(1)
            out_C_raw = self._run_mamba(tokens_rev, self.mamba_C)
            out_C = out_C_raw.flip(1)  # un-reverse

            # ── Scan D: column-major backward ─────────────────────────────────
            tokens_col_rev = tokens_col.flip(1)
            out_D_raw = self._run_mamba(tokens_col_rev, self.mamba_D)
            out_D_unreverse = out_D_raw.flip(1)
            out_D = out_D_unreverse.view(N, n_t, n_t, self.d_model).transpose(1, 2).reshape(N, T, self.d_model)

            merge = merge + out_C + out_D  # (N, T, d)

        # ── Global average pool over spatial tokens ───────────────────────────
        out = merge.mean(dim=1)         # (N, d)
        out = self.norm(out)            # (N, d)
        return out.view(B, P, self.d_model)  # (B, P, d)


# ── Variant 5: Upper-Triangle Anti-Diagonal Mamba ─────────────────────────────

class _UpperTriDiagCore(BaseVisualEncoder):
    """RP-geometry-native inner encoder exploiting exact symmetry of Recurrence Plots.

    Motivation
    ----------
    An RP is symmetric by construction: RP[i,j] = RP[j,i], and the main diagonal
    is identically zero (RP[i,i] = 0 always).  This means:
      - The upper triangle (excluding diagonal) contains 100% of the information.
      - The lower triangle is fully redundant.
      - The diagonal carries zero information.
    For l=64: full matrix = 4096 values → upper triangle = 2016 values (~50% saving).

    Geometric meaning of anti-diagonals
    ------------------------------------
    All cells (i, j) with j - i = k share the same **lag k**:
        RP[i, i+k] = distance(x_i, x_{i+k})

    Anti-diagonal k is the **lag-k recurrence profile** — how similar the series is
    to itself shifted by k timesteps.  Reading anti-diagonals from lag=1 to lag=l-1
    gives a structured sequence where each token encodes recurrence at one specific
    lag, ordered from short-range to long-range.

    This ordering is semantically natural for SSMs:  the Mamba SSM receives a
    sequence of lag tokens [lag1, lag2, ..., lag_{l-1}] and can model how recurrence
    *evolves as the lag grows* — directly capturing periodicity, trend stationarity,
    and regime changes in a geometrically meaningful way.

    Architecture
    ------------
    1. Extract upper triangle by anti-diagonal order: (l-1) tokens, each of variable
       length (padded to l-1 with zeros so all tokens are equal-length vectors).
    2. Each token (anti-diagonal = one lag): project from (l-1,) to d_model via Linear.
    3. MambaBlock × n_layers processes the sequence of (l-1) lag tokens.
    4. LayerNorm → output: (B, P, d_model).

    Complexity
    ----------
    Input tokens: (l-1) instead of l² — e.g. 63 instead of 4096 for l=64.
    Each token: at most l-1 values (padded to l-1).
    Total input: (l-1)² ≈ l²/2 values — 50% reduction vs full RP.
    """

    def __init__(
        self,
        patch_len: int,
        d_model: int = 128,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.l = patch_len

        # Each anti-diagonal k (lag k) has min(k, l-k) entries, padded to (l-1)
        self.token_dim = patch_len - 1   # max anti-diag length = l-1
        self.n_tokens = patch_len - 1    # lags 1 … l-1

        # Pre-compute gather indices for upper-triangle anti-diagonals:
        # For lag k (1-indexed): cells (i, i+k) for i in [0, l-1-k]
        # Stored as flat index into the l×l matrix: i*l + (i+k)
        indices = []   # list of 1-D index tensors, one per lag
        lengths = []   # actual number of entries at each lag
        for k in range(1, patch_len):
            rows = torch.arange(patch_len - k)         # i = 0..l-1-k
            cols = rows + k                             # j = i+k
            flat = rows * patch_len + cols              # flat index in l×l
            lengths.append(len(flat))
            indices.append(flat)

        self.register_buffer("_lengths", torch.tensor(lengths, dtype=torch.long))

        # Pad all anti-diagonals to length (l-1) for batched processing
        max_len = patch_len - 1
        padded = torch.zeros(patch_len - 1, max_len, dtype=torch.long)
        mask = torch.zeros(patch_len - 1, max_len, dtype=torch.bool)
        for k_idx, idx in enumerate(indices):
            n = len(idx)
            padded[k_idx, :n] = idx
            mask[k_idx, :n] = True
        self.register_buffer("_gather_idx", padded)   # (l-1, l-1)
        self.register_buffer("_mask", mask)            # (l-1, l-1) True = valid

        # Project each padded anti-diagonal token to d_model
        self.token_proj = nn.Linear(max_len, d_model)

        # SSM processes the sequence of lag tokens
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, state_dim=16, conv_kernel=3, expand_factor=2.0)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, rp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rp: (B, P, l, l)
        Returns:
            (B, P, d_model)
        """
        B, P, l, _ = rp.shape
        N = B * P
        l1 = self.patch_len - 1  # number of lag tokens = l-1

        # Flatten spatial dims: (N, l*l)
        x_flat = rp.reshape(N, l * l)

        # Gather upper-triangle anti-diagonals: (N, l-1, l-1)
        # _gather_idx: (l-1, l-1) flat indices into l×l
        idx = self._gather_idx.unsqueeze(0).expand(N, -1, -1)  # (N, l-1, l-1)
        tokens = torch.gather(
            x_flat.unsqueeze(1).expand(-1, l1, -1),  # (N, l-1, l*l)
            dim=2,
            index=idx,
        )  # (N, l-1, l-1)

        # Zero out padding positions (invalid entries beyond each anti-diagonal length)
        mask = self._mask.unsqueeze(0)  # (1, l-1, l-1)
        tokens = tokens * mask.float()  # (N, l-1, l-1)

        # Project each lag token to d_model: (N, l-1, d_model)
        tokens = self.token_proj(tokens)

        # Mamba SSM over the lag sequence
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)              # (N, l-1, d_model)
        out = tokens.mean(dim=1)               # (N, d_model) — pool over lags
        return out.view(B, P, self.d_model)    # (B, P, d_model)


# ── End-to-end wrapper: raw time series → UpperTriDiagRPEncoder ───────────────

class UpperTriDiagRPEncoder(nn.Module):
    """End-to-end RP encoder: (B, F, T) raw time series → (B, embedding_dim) embedding.

    Pipeline
    --------
    1. Swap axes: (B, F, T) → (B, T, F)
    2. Tokenize: sliding windows of size patch_len → (B, P, patch_len, F)
    3. Mean over channels (F): (B, P, patch_len)
    4. Compute RP per patch: (B, P, patch_len, patch_len)
    5. UpperTriDiagRPEncoder (inner): (B, P, d_model)
    6. Mean-pool over patches: (B, d_model)
    7. Linear projection: (B, embedding_dim)

    Generic RP visual encoder used across CLIP, GRAM, VL-JEPA, and BYOL.
    Exploits the upper-triangle anti-diagonal structure of Recurrence Plots.
    """

    def __init__(
        self,
        patch_len: int = 32,
        d_model: int = 128,
        n_layers: int = 2,
        embedding_dim: int = 128,
        rp_mv_strategy: str = "mean",
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.embedding_dim = embedding_dim
        self.rp_mv_strategy = rp_mv_strategy

        self.encoder = _UpperTriDiagCore(
            patch_len=patch_len,
            d_model=d_model,
            n_layers=n_layers,
        )
        self.output_proj = nn.Linear(d_model, embedding_dim, bias=False)

    def _compute_rp(self, x: torch.Tensor) -> torch.Tensor:
        """Compute recurrence plot on GPU.

        Args:
            x: (N, L) normalized time series
        Returns:
            (N, L, L) RP with values in [0, 1]
        """
        # Pairwise L2 distances: (N, L, L)
        diff = x.unsqueeze(2) - x.unsqueeze(1)   # (N, L, L)
        rp = diff.abs()                            # univariate: |x_i - x_j|
        # Normalize per sample to [0, 1]
        mx = rp.flatten(1).max(dim=1).values.clamp(min=1e-8)
        rp = rp / mx.view(-1, 1, 1)
        return rp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F, T) raw time series (channels-first)
        Returns:
            (B, embedding_dim)
        """
        B, F, T = x.shape

        # 1. Channels → time-major
        x = x.transpose(1, 2)                                   # (B, T, F)

        # 2. Mean over feature channels → univariate
        x = x.mean(dim=2)                                       # (B, T)

        # 3. Tokenize: unfold into patches of size patch_len
        L = self.patch_len
        if T < L:
            # pad if sequence shorter than one patch
            pad = x.new_zeros(B, L - T)
            x = torch.cat([x, pad], dim=1)
            T = L

        # Non-overlapping patches
        n_patches = T // L
        x = x[:, :n_patches * L].reshape(B, n_patches, L)      # (B, P, L)

        # 4. RP per patch: (B, P, L, L)
        N = B * n_patches
        x_flat = x.reshape(N, L)                                # (N, L)
        rp = self._compute_rp(x_flat)                           # (N, L, L)
        rp = rp.view(B, n_patches, L, L)                        # (B, P, L, L)

        # 5. _UpperTriDiagCore: (B, P, d_model)
        out = self.encoder(rp)                                   # (B, P, d_model)

        # 6. Pool over patches
        out = out.mean(dim=1)                                    # (B, d_model)

        # 7. Project to embedding_dim
        return self.output_proj(out)                             # (B, embedding_dim)


# ── Factory ───────────────────────────────────────────────────────────────────

class VisualEncoderFactory:
    """Instantiate a visual encoder by type name and patch length.

    All returned encoders implement the BaseVisualEncoder interface:
        forward(rp: Tensor[B, P, l, l]) -> Tensor[B, P, d]

    Supported encoder_type values:
        "cnn"              → CNNVisualEncoder
        "flatten_mamba"    → FlattenMambaEncoder (diagonal_order=False)
        "flatten_mamba_diag" → FlattenMambaEncoder (diagonal_order=True)
        "rp_ss2d_2"       → RPSS2DEncoder(n_scans=2)
        "ss2d_4"          → RPSS2DEncoder(n_scans=4)
        "upper_tri_diag"  → _UpperTriDiagCore (proposed — RP-geometry-native)
    """

    @staticmethod
    def build(
        encoder_type: str,
        patch_len: int,
        d_model: int = 128,
        n_layers: int = 2,
    ) -> BaseVisualEncoder:
        if encoder_type == "cnn":
            return CNNVisualEncoder(patch_len, d_model)
        elif encoder_type == "flatten_mamba":
            return FlattenMambaEncoder(patch_len, d_model, n_layers=n_layers)
        elif encoder_type == "flatten_mamba_diag":
            return FlattenMambaEncoder(patch_len, d_model, n_layers=n_layers, diagonal_order=True)
        elif encoder_type == "rp_ss2d_2":
            return RPSS2DEncoder(patch_len, d_model, n_scans=2, n_layers=n_layers)
        elif encoder_type == "ss2d_4":
            return RPSS2DEncoder(patch_len, d_model, n_scans=4, n_layers=n_layers)
        elif encoder_type == "upper_tri_diag":
            return _UpperTriDiagCore(patch_len, d_model, n_layers=n_layers)
        else:
            raise ValueError(
                f"Unknown encoder_type: '{encoder_type}'. "
                "Choose from: cnn, flatten_mamba, flatten_mamba_diag, rp_ss2d_2, ss2d_4, upper_tri_diag"
            )


# Backward-compat aliases (old names kept so existing imports don't break)
UpperTriDiagSimCLREncoder = UpperTriDiagRPEncoder
UpperTriDiagMambaEncoder = _UpperTriDiagCore


if __name__ == "__main__":
    tokens_dim = 32
    encoder = MambaVisualEncoder(
        depth=6,
        input_dim=tokens_dim,
        pooling="mean",
        model_dim=768,
        embedding_dim=128,
        expand_factor=1.5
    )
    
    # print(f"Trainable parameters: {encoder.count_parameters():,}")
    dummy = torch.randn(4, 307, 96)
    out = encoder(dummy)
    print("Output embedding shape:", out.shape)
    tokens = tokenize_sequence(dummy, token_size=tokens_dim)
    print("Output tokens shape:", tokens.shape)

