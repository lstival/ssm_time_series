"""\
CM-Mamba Presentation Builder

Creates a Beamer presentation for the paper:
  "CM-Mamba: Mamba Based Multimodal Contrastive Learning for Time Series Forecasting"

It generates:
- Python-produced feature visualizations (step-by-step through toy Mamba blocks/encoders)
- Numeric examples (projection, recurrence plot, contrastive similarity)
- Numeric + visual HiPPO (LegS) initialization demo

Usage:
    python build_cm_mamba_presentation.py

Output:
    ./presentation_output/cm_mamba/cm_mamba_presentation.pdf
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import re


OUTPUT_DIR = Path("presentation_output") / "cm_mamba"
LATEX_FILE = "cm_mamba_presentation.tex"
PDF_FILE = "cm_mamba_presentation.pdf"


def _format_vector(v, precision: int = 3) -> str:
    return ",\\;".join(f"{x:.{precision}f}" for x in v)


def _format_matrix(M, precision: int = 3) -> str:
    rows = []
    for r in M:
        rows.append(" & ".join(f"{x:.{precision}f}" for x in r))
    # LaTeX matrix rows are separated by `\\` + newline.
    return "\\\\\n".join(rows)


def hippo_legs_A_B_C(N: int):
    """HiPPO-LegS init used in the paper (lower-triangular A, B, C)."""
    import numpy as np

    A = np.zeros((N, N), dtype=float)
    for n in range(N):
        for k in range(N):
            if n == k:
                A[n, k] = -(2 * n + 1)
            elif k < n:
                A[n, k] = ((-1) ** (n - k + 1)) * math.sqrt((2 * n + 1) * (2 * k + 1))
            else:
                A[n, k] = 0.0

    B = np.array([math.sqrt(2 * n + 1) for n in range(N)], dtype=float)
    C = np.array([(((-1) ** n) * math.sqrt(2 * n + 1)) for n in range(N)], dtype=float)
    return A, B, C


def _expm(M):
    """Matrix exponential with a SciPy fast path and a small fallback.

    NumPy does not ship expm; SciPy usually does. The fallback uses an eig
    decomposition which is sufficient for the small demo matrices we create.
    """
    try:
        from scipy.linalg import expm  # type: ignore

        return expm(M)
    except Exception:
        import numpy as np

        w, V = np.linalg.eig(M)
        Vinv = np.linalg.inv(V)
        return V @ np.diag(np.exp(w)) @ Vinv


def create_directory_structure():
    print(f"Creating directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    print("✓ Directory created")


def generate_visualizations_and_numbers():
    """Generate figures + compute numeric examples used in the LaTeX."""

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["font.size"] = 11

    COLORS = {
        "input": "#2E86AB",
        "state": "#A23B72",
        "output": "#06A77D",
        "patch": "#F18F01",
        "highlight": "#C73E1D",
    }

    # ------------------------------------------------------------------
    # Running example time series
    # ------------------------------------------------------------------
    np.random.seed(42)
    T = 96
    t = np.linspace(0, 10, T)
    trend = 0.05 * t
    seasonality = 0.3 * np.sin(2 * np.pi * t / 2) + 0.2 * np.sin(2 * np.pi * t / 0.5)
    noise = 0.08 * np.random.randn(T)
    x = 0.5 + trend + seasonality + noise
    x[0] = 0.5
    x[1] = 0.8

    # ------------------------------------------------------------------
    # Patching illustration
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(T), x, color="gray", linewidth=1.5, alpha=0.7)
    patch_len = 16
    stride = 16
    for i in range(4):
        s = i * stride
        e = s + patch_len
        ax.axvspan(s, e, alpha=0.25, color=[COLORS["patch"], COLORS["input"], COLORS["state"], COLORS["output"]][i])
        ax.text((s + e) / 2, float(np.max(x)) + 0.1, f"Patch {i+1}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Tokenization via Non-overlapping Patches (Example)", fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_xlim(-1, T)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "patching.png", bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Recurrence plot example (small patch)
    # ------------------------------------------------------------------
    patch_small = np.array([0.5, 0.8, 0.2, 0.6], dtype=float)
    rp = np.abs(patch_small[:, None] - patch_small[None, :])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(len(patch_small)), patch_small, "o-", color=COLORS["patch"], linewidth=2.5)
    axes[0].set_title("1D Patch (l=4)", fontweight="bold")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3)

    im = axes[1].imshow(rp, cmap="Blues", origin="lower")
    axes[1].set_title(r"Recurrence Plot $R_{ij}=|x_i-x_j|$", fontweight="bold")
    axes[1].set_xlabel("j")
    axes[1].set_ylabel("i")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "rp_example.png", bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Temporal projector numeric example
    # ------------------------------------------------------------------
    np.random.seed(0)
    d_small = 4
    l_small = 8
    patch8 = x[:l_small].copy()
    Wt = 0.25 * np.random.randn(d_small, l_small)
    bt = 0.05 * np.random.randn(d_small)
    emb_t = Wt @ patch8 + bt

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.imshow(Wt, cmap="coolwarm", aspect="auto")
    ax.set_title("Temporal Projector Weights (toy)  W_t in R^{4x8}", fontweight="bold")
    ax.set_xlabel("Patch index")
    ax.set_ylabel("Embedding dim")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "temporal_projector_W.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.bar(range(d_small), emb_t, color=COLORS["input"], edgecolor="black")
    ax.set_title("Temporal Token Embedding (toy output)", fontweight="bold")
    ax.set_xlabel("Embedding dim")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "temporal_projector_out.png", bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Visual projector (conv) numeric example (RP -> conv -> embedding)
    # ------------------------------------------------------------------
    rp8 = np.abs(patch8[:, None] - patch8[None, :])
    kernel = (1.0 / 3.0) * np.array(
      [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]], dtype=float
    )

    H, W = rp8.shape
    kh, kw = kernel.shape
    out_h, out_w = H - kh + 1, W - kw + 1
    feat = np.zeros((out_h, out_w), dtype=float)
    for i in range(out_h):
      for j in range(out_w):
        feat[i, j] = float(np.sum(rp8[i : i + kh, j : j + kw] * kernel))

    Wv = 0.15 * np.random.randn(d_small, feat.size)
    bv = 0.05 * np.random.randn(d_small)
    emb_v = Wv @ feat.flatten() + bv

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.imshow(kernel, cmap="coolwarm", aspect="equal")
    ax.set_title("Visual Projector Kernel (toy 3×3)", fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "visual_projector_kernel.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(feat, cmap="magma", origin="lower")
    ax.set_title("Conv feature map on RP (toy)", fontweight="bold")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "visual_projector_featmap.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.bar(range(d_small), emb_v, color=COLORS["highlight"], edgecolor="black")
    ax.set_title("Visual Token Embedding (toy output)", fontweight="bold")
    ax.set_xlabel("Embedding dim")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "visual_projector_out.png", bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Toy Mamba block feature walkthrough (heatmaps)
    # ------------------------------------------------------------------
    np.random.seed(7)
    Ttoy = 32
    d_model = 16
    d_inner = 24
    x_in = np.stack([
        np.sin(np.linspace(0, 3 * np.pi, Ttoy) + p) for p in np.linspace(0, 1.0, d_model)
    ], axis=1)
    x_in += 0.05 * np.random.randn(Ttoy, d_model)

    Wz = 0.5 * np.random.randn(d_model, d_inner)
    Wu = 0.5 * np.random.randn(d_model, d_inner)
    Wout = 0.4 * np.random.randn(d_inner, d_model)

    z = x_in @ Wz
    u = x_in @ Wu

    # causal conv (simplified): same kernel for all channels
    k = np.array([0.15, 0.6, 0.25], dtype=float)
    u_conv = np.zeros_like(u)
    for t_idx in range(Ttoy):
        for kk in range(3):
            src = t_idx - kk
            if src >= 0:
                u_conv[t_idx] += k[kk] * u[src]

    # HiPPO (shared small state) selective scan toy
    Nstate = 6
    A, B, C = hippo_legs_A_B_C(Nstate)
    delta = 0.2
    Abar = _expm(delta * A)
    eye = np.eye(Nstate)
    integral = np.linalg.solve(delta * A, Abar - eye)

    WB = 0.2 * np.random.randn(d_inner, Nstate)
    WC = 0.2 * np.random.randn(d_inner, Nstate)

    h = np.zeros((Nstate,), dtype=float)
    y_ssm = np.zeros((Ttoy, d_inner), dtype=float)

    for t_idx in range(Ttoy):
        B_t = u_conv[t_idx] @ WB
        C_t = u_conv[t_idx] @ WC
        u_drive = float(np.mean(u_conv[t_idx]))
        Bbar_t = integral @ B_t
        h = Abar @ h + Bbar_t * u_drive
        y_scalar = float(C_t @ h)
        y_ssm[t_idx] = y_scalar

    gated = (1.0 / (1.0 + np.exp(-z))) * y_ssm
    x_out = x_in + (gated @ Wout)

    def heatmap(mat, title, fname, cmap="viridis"):
        fig, ax = plt.subplots(figsize=(10, 3.6))
        im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Feature dim")
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "figures" / fname, bbox_inches="tight")
        plt.close()

    heatmap(x_in, "(Toy) Input features into a Mamba block: x in R^{T x d_model}", "mamba_step1_input.png")
    heatmap(u_conv, "(Toy) After input projection + causal conv: u_conv in R^{T x d_inner}", "mamba_step2_conv.png")
    heatmap(y_ssm, "(Toy) After selective scan SSM (shown as inner features)", "mamba_step3_ssm.png", cmap="magma")
    heatmap(gated, "(Toy) Gating: sigmoid(z) * ssm_out", "mamba_step4_gated.png", cmap="plasma")
    heatmap(x_out, "(Toy) Output + residual: x_out = x + W_out(gated)", "mamba_step5_out.png")

    # ------------------------------------------------------------------
    # Contrastive similarity matrix demo
    # ------------------------------------------------------------------
    np.random.seed(123)
    N = 8
    zt = np.random.randn(N, 16)
    zv = np.random.randn(N, 16)
    zv = 0.6 * zt + 0.4 * zv
    zt = zt / np.linalg.norm(zt, axis=1, keepdims=True)
    zv = zv / np.linalg.norm(zv, axis=1, keepdims=True)
    tau = 0.07
    S = (zt @ zv.T) / tau

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sns.heatmap(S, ax=ax, cmap="coolwarm", center=0.0, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title(r"Similarity matrix $S_{ij} = (z^t_i)^\top z^v_j / \tau$ (toy)", fontweight="bold")
    ax.set_xlabel("visual j")
    ax.set_ylabel("temporal i")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "contrastive_similarity.png", bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # HiPPO (LegS) numeric + visual demo
    # ------------------------------------------------------------------
    A4, B4, C4 = hippo_legs_A_B_C(4)
    A16, _, _ = hippo_legs_A_B_C(16)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(A16, ax=ax, cmap="PuOr", center=0.0, cbar_kws={"shrink": 0.8})
    ax.set_title("HiPPO-LegS initialization: A (N=16) heatmap", fontweight="bold")
    ax.set_xlabel("k")
    ax.set_ylabel("n")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "hippo_A_heatmap.png", bbox_inches="tight")
    plt.close()

    Nresp = 6
    A6, B6, C6 = hippo_legs_A_B_C(Nresp)
    delta = 0.15
    Abar = _expm(delta * A6)
    integral = np.linalg.solve(delta * A6, Abar - np.eye(Nresp))

    steps = 60
    u_imp = np.zeros((steps,), dtype=float)
    u_imp[0] = 1.0
    h = np.zeros((Nresp,), dtype=float)
    y = np.zeros((steps,), dtype=float)
    for t_idx in range(steps):
        Bbar = integral @ B6
        h = Abar @ h + Bbar * u_imp[t_idx]
        y[t_idx] = float(C6 @ h)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.stem(range(steps), y, basefmt=" ")
    ax.set_title("HiPPO-LegS: impulse response of (A,B,C) discretized", fontweight="bold")
    ax.set_xlabel("t")
    ax.set_ylabel("y_t")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "hippo_impulse.png", bbox_inches="tight")
    plt.close()

    numbers = {
        "patch_small": patch_small,
        "rp": rp,
        "Wt": Wt,
        "bt": bt,
        "patch8": patch8,
        "emb_t": emb_t,
        "rp8": rp8,
        "kernel": kernel,
        "bv": bv,
        "emb_v": emb_v,
        "A4": A4,
        "B4": B4,
        "C4": C4,
        "tau": tau,
        "S": S,
    }

    return numbers


def write_latex_file(numbers):
    import numpy as np
    import re

    patch_small = numbers["patch_small"]
    rp = numbers["rp"]
    bt = numbers["bt"]
    bv = numbers["bv"]
    patch8 = numbers["patch8"]
    emb_t = numbers["emb_t"]
    emb_v = numbers["emb_v"]
    A4 = numbers["A4"]
    B4 = numbers["B4"]
    C4 = numbers["C4"]
    tau = float(numbers["tau"])

    S = numbers["S"]
    S4 = np.array(S[:4, :4], dtype=float)

    latex_content = rf"""\documentclass[aspectratio=169]{{beamer}}

\usetheme{{Madrid}}
\usecolortheme{{beaver}}
\usefonttheme{{professionalfonts}}

\usepackage{{amsmath, amssymb, amsfonts}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{tikz}}
\usetikzlibrary{{positioning, arrows.meta}}

  itle[CM-Mamba]{{CM-Mamba: Multimodal Contrastive Mamba for Time Series Forecasting}}
\subtitle{{Time + Recurrence Plots \;\; + \;\; Contrastive Alignment}}
\author{{Leandro Stival}}
\institute[WUR]{{Wageningen University \& Research}}
\date{{\today}}

\begin{{document}}

\begin{{frame}}
  	itlepage
\end{{frame}}

\begin{{frame}}{{Overview}}
  ableofcontents
\end{{frame}}

\section{{Motivation \& Idea}}

\begin{{frame}}{{Problem: Why CM-Mamba?}}
\begin{{itemize}}
  \item Mamba/SSMs are efficient for long horizons, but can miss fine-grained local patterns (lossy fixed-state memory).
  \item Recurrence plots (RP) make local dynamics explicit by turning a 1D patch into a 2D structure.
  \item CM-Mamba aligns temporal and visual views with contrastive learning (no attention blocks added).
\end{{itemize}}
\vspace{{0.1cm}}
\begin{{center}}
  \includegraphics[width=0.82\textwidth,height=0.45\textheight,keepaspectratio]{{figures/patching.png}}
\end{{center}}
\end{{frame}}

\begin{{frame}}{{CM-Mamba: Pipeline Overview}}
\vspace{{-0.1cm}}
\centering
\begin{{tikzpicture}}[node distance=0.75cm, every node/.style={{font=\scriptsize}}, scale=0.60, transform shape]
  \node[draw, rounded corners, fill=blue!10] (x) {{Time series $\mathcal{{T}}$}};
  \node[draw, rounded corners, right=of x, fill=orange!15] (patch) {{Patching}};
  \node[draw, rounded corners, right=of patch, fill=green!10] (tproj) {{$P_t$ (linear)}};
  \node[draw, rounded corners, below=0.6cm of tproj, fill=red!10] (rp) {{RP}};
  \node[draw, rounded corners, right=of rp, fill=green!10] (vproj) {{$P_v$ (conv)}};
  \node[draw, rounded corners, right=of tproj, fill=purple!10] (Et) {{$E_t$ (Mamba stack)}};
  \node[draw, rounded corners, right=of vproj, fill=purple!10] (Ev) {{$E_v$ (Mamba stack)}};
  \node[draw, rounded corners, right=0.8cm of Et, fill=yellow!15] (cl) {{Contrastive loss}};

  \draw[-{{Stealth}}] (x) -- (patch);
  \draw[-{{Stealth}}] (patch) -- (tproj);
  \draw[-{{Stealth}}] (patch) -- (rp);
  \draw[-{{Stealth}}] (tproj) -- (Et);
  \draw[-{{Stealth}}] (rp) -- (vproj);
  \draw[-{{Stealth}}] (vproj) -- (Ev);
  \draw[-{{Stealth}}] (Et) -- (cl);
  \draw[-{{Stealth}}] (Ev) -- (cl);
\end{{tikzpicture}}
\end{{frame}}

\begin{{frame}}{{Token Shapes \& Similarity}}
\small
\begin{{itemize}}
  \item Temporal tokens $x^t \in \mathbb{{R}}^{{P\times l}}$ and visual tokens $x^v \in \mathbb{{R}}^{{P\times l\times l}}$.
  \item Encoders output normalized embeddings: $z^t, z^v$; similarity $S_{{ij}}=(z^t_i)^\top z^v_j/\tau$.
\end{{itemize}}
\end{{frame}}

\section{{Recurrence Plots: Visual View}}

\begin{{frame}}{{Recurrence Plot: Numerical Example}}
\begin{{columns}}
  \begin{{column}}{{0.52\textwidth}}
    Patch (toy, $l=4$):
    \[
      x = [{_format_vector(patch_small, 1)}]
    \]
    \[
      R_{{ij}} = |x_i - x_j| =
      \begin{{pmatrix}}
      {_format_matrix(rp, 1)}
      \end{{pmatrix}}
    \]
    \vspace{{0.2cm}}
    This 2D structure makes short-term dynamics explicit.
  \end{{column}}
  \begin{{column}}{{0.48\textwidth}}
    \centering
    \includegraphics[width=\textwidth]{{figures/rp_example.png}}
  \end{{column}}
\end{{columns}}
\end{{frame}}

\section{{Projectors: From Raw Tokens to Embeddings}}

\begin{{frame}}{{Temporal Projector $P_t$: Numeric + Visual}}
Given a patch $x \in \mathbb{{R}}^{{l}}$, a toy linear projector outputs $e^t=W_t x + b_t$.

\vspace{{0.2cm}}
Toy patch ($l=8$):
\[
 x = [{_format_vector(patch8, 3)}]
\]
\[
 e^t = W_t x + b_t \in \mathbb{{R}}^4,
 \quad b_t=[{_format_vector(bt, 3)}]
\]
\[
 e^t = [{_format_vector(emb_t, 3)}]
\]

\begin{{columns}}
  \begin{{column}}{{0.52\textwidth}}
    \centering
    \includegraphics[width=\textwidth]{{figures/temporal_projector_W.png}}
  \end{{column}}
  \begin{{column}}{{0.48\textwidth}}
    \centering
    \includegraphics[width=\textwidth]{{figures/temporal_projector_out.png}}
  \end{{column}}
\end{{columns}}
\end{{frame}}

\begin{{frame}}{{Visual Projector $P_v$: RP \rightarrow Conv \rightarrow Embedding (intuition)}}
\small
We build the visual token by convolving the recurrence plot:
\[
x \in \mathbb{{R}}^l \Rightarrow RP \in \mathbb{{R}}^{{l\times l}} \Rightarrow \text{{Conv}}(RP) \Rightarrow e^v \in \mathbb{{R}}^d
\]

\begin{{columns}}
  \begin{{column}}{{0.34\textwidth}}
    \centering
    \includegraphics[width=\textwidth,height=0.32\textheight,keepaspectratio]{{figures/visual_projector_kernel.png}}
  \end{{column}}
  \begin{{column}}{{0.33\textwidth}}
    \centering
    \includegraphics[width=\textwidth,height=0.32\textheight,keepaspectratio]{{figures/visual_projector_featmap.png}}
  \end{{column}}
  \begin{{column}}{{0.33\textwidth}}
    \centering
    \includegraphics[width=\textwidth,height=0.32\textheight,keepaspectratio]{{figures/visual_projector_out.png}}
  \end{{column}}
\end{{columns}}
\end{{frame}}

\begin{{frame}}{{Visual Projector $P_v$: Toy numeric example}}
\small
Toy output embedding ($d=4$):
\[
e^v = W_v\,\mathrm{{vec}}(\mathrm{{Conv}}(RP)) + b_v,
\quad b_v=[{_format_vector(bv, 3)}]
\]
\[
e^v = [{_format_vector(emb_v, 3)}]
\]
\end{{frame}}

\section{{Inside a Mamba Block (Feature Walkthrough)}}

\begin{{frame}}{{Mamba Block: What features are produced?}}
We visualize a \emph{{toy}} Mamba block (small dims, fixed seed) to show how features evolve.

\begin{{itemize}}
  \item Input features: $x \in \mathbb{{R}}^{{T\times d_{{model}}}}$
  \item After projection + causal conv: $u_{{conv}} \in \mathbb{{R}}^{{T\times d_{{inner}}}}$
  \item Selective scan produces an SSM-mixed signal (then gated).
  \item Output is projected back + residual.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}{{Toy Mamba Block: Input features}}
\centering
\includegraphics[width=0.95\textwidth,height=0.48\textheight,keepaspectratio]{{figures/mamba_step1_input.png}}
\end{{frame}}

\begin{{frame}}{{Toy Mamba Block: After causal conv}}
\centering
\includegraphics[width=0.95\textwidth,height=0.48\textheight,keepaspectratio]{{figures/mamba_step2_conv.png}}
\end{{frame}}

\begin{{frame}}{{Toy Mamba Block: After selective scan (SSM mixing)}}
\centering
\includegraphics[width=0.95\textwidth,height=0.48\textheight,keepaspectratio]{{figures/mamba_step3_ssm.png}}
\end{{frame}}

\begin{{frame}}{{Toy Mamba Block: Gating + output + residual}}
\begin{{columns}}
  \begin{{column}}{{0.5\textwidth}}
    \centering
    \includegraphics[width=\textwidth]{{figures/mamba_step4_gated.png}}
  \end{{column}}
  \begin{{column}}{{0.5\textwidth}}
    \centering
    \includegraphics[width=\textwidth]{{figures/mamba_step5_out.png}}
  \end{{column}}
\end{{columns}}
\end{{frame}}

\section{{Contrastive Alignment (Temporal vs Visual)}}

\begin{{frame}}{{Contrastive Similarity: Visual + Numeric Example}}
We normalize embeddings and compute:
\[
S_{{ij}} = \frac{{(z^t_i)^\top z^v_j}}{{\tau}}, \quad \tau={tau:.2f}
\]

\begin{{columns}}
  \begin{{column}}{{0.52\textwidth}}
    \small Toy $4\times 4$ slice of $S$:
    {{\tiny
    \[
    S \approx
    \begin{{pmatrix}}
    {_format_matrix(S4, 2)}
    \end{{pmatrix}}
    \]
    }}
  \end{{column}}
  \begin{{column}}{{0.48\textwidth}}
    \centering
    \includegraphics[width=\textwidth,height=0.42\textheight,keepaspectratio]{{figures/contrastive_similarity.png}}
  \end{{column}}
\end{{columns}}
\end{{frame}}

\section{{HiPPO Initialization (Numeric + Visual)}}

\begin{{frame}}{{HiPPO-LegS: Numeric Example (N=4)}}
Using the paper's initialization:
\[
A_{{n,k}} = -(2n+1)\,\delta_{{n,k}} + (-1)^{{n-k+1}}\sqrt{{(2n+1)(2k+1)}}\,\mathbb{{I}}_{{k<n}}
\]
\[
A = \begin{{pmatrix}}
{_format_matrix(A4, 3)}
\end{{pmatrix}}
\]
\[
B=[{_format_vector(B4, 3)}],\quad C=[{_format_vector(C4, 3)}]
\]
\end{{frame}}

\begin{{frame}}{{HiPPO-LegS: Visual intuition (A structure)}}
\centering
\vspace{{-0.2cm}}
\includegraphics[width=0.9\textwidth,height=0.55\textheight,keepaspectratio]{{figures/hippo_A_heatmap.png}}
\\
\small Heatmap of $A$ (N=16)
\end{{frame}}

\begin{{frame}}{{HiPPO-LegS: Visual intuition (memory dynamics)}}
\centering
\vspace{{-0.2cm}}
\includegraphics[width=0.9\textwidth,height=0.55\textheight,keepaspectratio]{{figures/hippo_impulse.png}}
\\
\small Discretized impulse response: memory dynamics
\end{{frame}}

\section{{Wrap-up}}

\begin{{frame}}{{Takeaways}}
\begin{{itemize}}
  \item CM-Mamba preserves Mamba's efficiency while injecting local structure via recurrence plots.
  \item The python-generated feature heatmaps show how a Mamba block transforms features step by step.
  \item HiPPO-LegS provides a principled initialization for long-range memory; its structure is visible in $A$.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}
\centering\Huge Questions?
\end{{frame}}

\end{{document}}
"""

    # Guardrail: if any `\t...` sequences were introduced while editing this
    # file through JSON-escaped patches, they may appear as literal TABs.
    # Fix the known Beamer commands that would otherwise break slides 1-2.
    # First handle the most common "\t"-to-TAB corruption patterns.
    latex_content = (
      latex_content
      .replace("\tite[", "\\title[")
      .replace("\titepage", "\\titlepage")
      .replace("\tableofcontents", "\\tableofcontents")
      .replace("\today", "\\today")
    )

    # Then handle cases where the leading backslash was lost entirely.
    latex_content = re.sub(r"(?m)^[ \t]*itle\[", r"\\title[", latex_content)
    latex_content = re.sub(r"(?m)^[ \t]*itlepage\b", r"\\titlepage", latex_content)
    latex_content = re.sub(r"(?m)^[ \t]*ableofcontents\b", r"\\tableofcontents", latex_content)

    latex_path = OUTPUT_DIR / LATEX_FILE
    print(f"Writing LaTeX: {latex_path}")
    latex_path.write_text(latex_content, encoding="utf-8")
    print("✓ LaTeX file written")


def _summarize_latex_log(log_text: str, max_lines: int = 20) -> list[str]:
    """Return the most relevant warning lines from a LaTeX log."""

    interesting: list[str] = []
    for line in log_text.splitlines():
        if (
            "Overfull \\hbox" in line
            or "Underfull \\hbox" in line
            or "Overfull \\vbox" in line
            or "Underfull \\vbox" in line
        ):
            interesting.append(line.strip())

    if len(interesting) > max_lines:
        head = interesting[: max_lines // 2]
        tail = interesting[-(max_lines // 2) :]
        return head + [f"... ({len(interesting) - len(head) - len(tail)} more) ..."] + tail
    return interesting


def compile_pdf() -> tuple[bool, bool]:
    print("\nCompiling PDF...")
    pdf_path = OUTPUT_DIR / PDF_FILE
    log_path = OUTPUT_DIR / (Path(LATEX_FILE).stem + ".log")
    saw_warning_exit = False

    for i in range(2):
        print(f"  Pass {i+1}/2...")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", LATEX_FILE],
            cwd=OUTPUT_DIR,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # MiKTeX sometimes returns 1 for non-fatal issues (e.g., overfull boxes)
            # while still producing a usable PDF.
            if pdf_path.exists():
                saw_warning_exit = True
                print("! pdflatex returned non-zero, but PDF exists (treating as warning).")
                print(result.stdout[-1200:])
            else:
                print("✗ pdflatex failed and no PDF was produced")
                print(result.stdout[-2000:])
                print(result.stderr[-2000:])
                return False, True

    keep_log = saw_warning_exit
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        warning_lines = _summarize_latex_log(log_text)
        if warning_lines:
            keep_log = True
            print("\nLaTeX box warnings (from .log):")
            for line in warning_lines:
                print(f"  {line}")

    if pdf_path.exists():
        if keep_log:
            print(f"✓ PDF produced (kept log): {pdf_path}")
        else:
            print(f"✓ PDF compiled: {pdf_path}")
        return True, keep_log

    print("✗ PDF not found")
    return False, True


def clean_auxiliary_files(*, keep_log: bool = False):
  print("\\nCleaning auxiliary files...")
  exts = [".aux", ".nav", ".out", ".snm", ".toc", ".vrb"]
  if not keep_log:
    exts.append(".log")
  for ext in exts:
    p = OUTPUT_DIR / (Path(LATEX_FILE).stem + ext)
    if p.exists():
      p.unlink()
  if keep_log:
    print("✓ Cleanup complete (log preserved)")
  else:
    print("✓ Cleanup complete")


def main():
    print("=" * 60)
    print("CM-Mamba Presentation Builder")
    print("=" * 60)

    create_directory_structure()
    numbers = generate_visualizations_and_numbers()
    write_latex_file(numbers)
    ok, keep_log = compile_pdf()
    clean_auxiliary_files(keep_log=keep_log)

    print("\\n" + "=" * 60)
    if ok:
        print("✓ BUILD SUCCESSFUL")
        print(f"Output: {OUTPUT_DIR / PDF_FILE}")
    else:
        print("✗ BUILD FAILED")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
