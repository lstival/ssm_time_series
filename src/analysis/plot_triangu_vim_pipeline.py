#!/usr/bin/env python3
"""Visualize TriangU/ViM-style RP tokenization from one ICML time series.

Produces:
1) A composite figure with:
    - One raw series segment with 3 highlighted patches (len=64)
    - The 3 patches
    - RP for each patch (showing they are different)
    - TriangU token matrix (valid values + padded slots in red)
    - Decreasing token-length profile by lag
    - Flattened valid lag-ordered sequence used by ViM
    - Fixed-size token embeddings after projection
2) A markdown file explaining how this is fed into the ViM-style visual path.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_rp(x: np.ndarray) -> np.ndarray:
    """Univariate RP like UpperTriDiagRPEncoder._compute_rp: abs diff + max-normalize."""
    rp = np.abs(x[:, None] - x[None, :])
    mx = float(np.max(rp))
    if mx > 1e-8:
        rp = rp / mx
    return rp


def triang_u_tokens(rp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Replicate _UpperTriDiagCore gather/pad logic.

    Returns:
      tokens: (L-1, L-1) padded lag-token matrix
      mask:   (L-1, L-1) valid positions
      lengths per lag token
    """
    l = rp.shape[0]
    max_len = l - 1
    tokens = np.zeros((l - 1, max_len), dtype=np.float32)
    mask = np.zeros((l - 1, max_len), dtype=bool)
    lengths: List[int] = []

    for lag in range(1, l):
        rows = np.arange(l - lag)
        cols = rows + lag
        vals = rp[rows, cols]
        n = len(vals)
        tokens[lag - 1, :n] = vals
        mask[lag - 1, :n] = True
        lengths.append(n)

    return tokens, mask, lengths


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "results" / "triangu_vim"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = root / "ICML_datasets" / "ETT-small" / "ETTh1.csv"
    df = pd.read_csv(csv_path)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise RuntimeError("No numeric column found in ETTh1.csv")

    col = "OT" if "OT" in df.columns else numeric_cols[0]
    series = df[col].to_numpy(dtype=np.float32)

    patch_len = 64
    n_patches = 3
    start = 0
    end = start + patch_len * n_patches
    if end > len(series):
        raise RuntimeError("Series too short for 3 patches of 64")

    seg = series[start:end]
    patches = [seg[i * patch_len:(i + 1) * patch_len] for i in range(n_patches)]

    rps = [compute_rp(p) for p in patches]
    tokens, mask, lengths = triang_u_tokens(rps[0])
    flat_valid = tokens[mask]  # length = 2016 for L=64

    # Demonstrate fixed-size embeddings from padded lag tokens: (63,63) -> (63,d_model)
    d_model = 48
    rng = np.random.default_rng(7)
    proj_w = rng.normal(scale=0.08, size=(patch_len - 1, d_model)).astype(np.float32)
    proj_b = np.zeros((d_model,), dtype=np.float32)
    embeds = tokens @ proj_w + proj_b

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(seg))
    ax0.plot(x, seg, lw=1.2, color="black")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i in range(n_patches):
        s = i * patch_len
        e = (i + 1) * patch_len
        ax0.axvspan(s, e - 1, color=colors[i], alpha=0.20)
    ax0.set_title("Raw time series with 3 patches (size=64)")
    ax0.set_xlabel("time index")
    ax0.set_ylabel(col)
    ax0.grid(alpha=0.25)

    ax1 = fig.add_subplot(gs[0, 1])
    for i, p in enumerate(patches):
        ax1.plot(np.arange(patch_len), p, lw=1.3, color=colors[i], label=f"patch {i+1}")
    ax1.set_title("Patch values")
    ax1.set_xlabel("within-patch index")
    ax1.set_ylabel(col)
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(np.arange(1, patch_len), lengths, marker="o", ms=2.5, lw=1.0, color="#d62728")
    ax2.set_title("Decreasing lag-token length: L-k")
    ax2.set_xlabel("lag k")
    ax2.set_ylabel("length")
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(rps[0], cmap="viridis", aspect="equal")
    ax3.set_title("RP of patch 1")
    ax3.set_xlabel("j")
    ax3.set_ylabel("i")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(rps[1], cmap="viridis", aspect="equal")
    ax4.set_title("RP of patch 2")
    ax4.set_xlabel("j")
    ax4.set_ylabel("i")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(gs[1, 2])
    im5 = ax5.imshow(rps[2], cmap="viridis", aspect="equal")
    ax5.set_title("RP of patch 3")
    ax5.set_xlabel("j")
    ax5.set_ylabel("i")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(gs[2, 0])
    # Visualization-only alignment: move valid values to the upper-right corner
    # so the triangular support matches RP upper-triangle intuition.
    arr = np.ma.array(np.fliplr(tokens), mask=np.fliplr(~mask))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="red")
    im6 = ax6.imshow(arr, cmap=cmap, aspect="auto")
    ax6.set_title("Lag-token matrix (upper-right support, red = zero padding)")
    ax6.set_xlabel("token element index (max=63)")
    ax6.set_ylabel("lag token (1..63)")
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(np.arange(len(flat_valid)), flat_valid, lw=1.0, color="#6a3d9a")
    ax7.set_title(f"Flattened valid sequence to ViM (length={len(flat_valid)})")
    ax7.set_xlabel("flat index (lag-order)")
    ax7.set_ylabel("RP value")
    ax7.grid(alpha=0.25)

    ax8 = fig.add_subplot(gs[2, 2])
    emb_show = embeds[:, :24]  # display a subset of dims for readability
    im8 = ax8.imshow(emb_show, cmap="magma", aspect="auto")
    ax8.set_title("Fixed-size token embeddings after linear projection")
    ax8.set_xlabel("embedding dim (shown first 24)")
    ax8.set_ylabel("lag token index (1..63)")
    fig.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    fig.suptitle("TriangU / ViM-style visual preprocessing from one ICML time series", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = out_dir / "triangu_vim_pipeline.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    md_path = out_dir / "TRIANGU_VIM_EXPLANATION.md"
    md_text = f"""# TriangU ViM-style Pipeline (Example)

Source series: `ETTh1.csv`, column `{col}`.

![TriangU ViM pipeline](triangu_vim_pipeline.png)

## How it works

1. Select one raw time series and split into 3 non-overlapping patches of size 64.
2. For each patch, compute recurrence plot values with `RP[i,j] = |x_i - x_j|`, then normalize by the patch RP maximum.
3. Each patch has different values, so each patch produces a different RP.
4. Keep lag-ordered upper-triangle values: lag token `k` uses cells `(i, i+k)` for `i = 0..L-k-1`, giving 63 lag tokens for `L=64`.
5. Pad each lag token to fixed length 63; padded (unused) positions are zeros and are shown in red in the figure.
6. Token lengths decrease with lag: `63, 62, ..., 1`.
7. A linear projection maps each padded lag token from length 63 to fixed embedding size `d_model`.
8. The SSM then processes a sequence of fixed-size embeddings (one per lag token).

## How this is fed to the ViM-style visual path

- Model input matrix before projection has shape `(63, 63)` per patch (lag tokens x padded token length).
- Token lengths are naturally decreasing with lag, but padding makes all tokens equal size for batching.
- Then a linear token projection maps each lag token to `d_model`.
- Mamba blocks process the lag-token sequence.
- The model pools across lag tokens, and then across patches.

## SSM heads in this project

- `upper_tri` (TriangU path): no multi-head split. It uses one lag-token sequence per patch, processed by a stack of Mamba blocks in series.
- `rp_ss2d` path: multiple scan branches are used (`A/B` or `A/B/C/D`). Their outputs are combined by element-wise sum, then pooled.

This mirrors the code path used by `UpperTriDiagRPEncoder` / `_UpperTriDiagCore`.
"""
    md_path.write_text(md_text, encoding="utf-8")

    print(f"Saved figure: {fig_path}")
    print(f"Saved explanation: {md_path}")


if __name__ == "__main__":
    main()
