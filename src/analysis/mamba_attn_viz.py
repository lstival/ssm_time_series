"""Mamba Encoder Attention Visualization Tool.

Inspired by "The Hidden Attention of Mamba Models" (Ali et al., 2024)
https://github.com/AmeenAli/HiddenMambaAttn

Works by:
1. Temporarily replacing the CUDA-only mamba_ssm kernel with a pure-PyTorch
   selective scan (for CPU / visualization purposes only).
2. Using gradient saliency (GradCAM-style) over the SSM token sequence.
3. For the visual UpperTriDiag branch: also using activation magnitudes to
   identify which anti-diagonal lag tokens the Mamba attends to most.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import pandas as pd
from scipy.ndimage import gaussian_filter

# ── path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent   # src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.mamba_encoder import MambaEncoder
from models.mamba_visual_encoder import UpperTriDiagRPEncoder, _UpperTriDiagCore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pure-PyTorch Mamba replacement (CPU-compatible, for visualization only)
#
#  This re-implements the mamba_ssm selective scan in pure Python/PyTorch.
#  Based on the Mamba paper (Gu & Dao, 2023): https://arxiv.org/abs/2312.00752
#  and the reference implementation at github.com/state-spaces/mamba.
#
#  On GPU we use the fast CUDA kernel; on CPU this fallback activates.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _PurePyMamba(nn.Module):
    """Pure-PyTorch Mamba-1 selective scan module.

    Closely mirrors mamba_ssm.Mamba interface so we can hot-swap it.
    Uses the recurrent form (slow but device-agnostic).
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.expand   = expand
        d_inner       = expand * d_model

        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        # depthwise conv
        self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv,
                                   padding=d_conv - 1, groups=d_inner, bias=True)
        # SSM projections
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # dt, B, C
        self.dt_proj  = nn.Linear(1, d_inner, bias=True)
        # A (log-parameterised)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log    = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_inner, -1))  # (d_inner, N)
        self.D        = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_inner)

    # ------------------------------------------------------------------ #
    def _selective_scan(self, u, dt, A, B, C, D):
        """Vectorized selective scan using cumulative prefix products.

        Computes the same result as the recurrent form but much faster on CPU
        by using tensor operations instead of a Python loop.

        u  : (L, d_inner)
        dt : (L, d_inner)
        A  : (d_inner, N)   — negative continuous-time A
        B  : (L, N)
        C  : (L, N)
        D  : (d_inner,)
        Returns y : (L, d_inner)
        """
        L, d   = u.shape
        N      = A.shape[1]
        device = u.device
        dtype  = torch.float32  # use float32 for numerical stability

        u  = u.float()
        dt = dt.float()
        C  = C.float()
        B  = B.float()

        # Discretise
        dt_A = torch.einsum("ld,dn->ldn", dt, A.float())  # (L, d, N)
        dA   = torch.exp(dt_A)                              # (L, d, N)
        dBu  = dt.unsqueeze(2) * B.unsqueeze(1) * u.unsqueeze(2)  # (L, d, N)

        # Parallel prefix scan:
        #   h_t = dA_t * h_{t-1} + dBu_t
        #   y_t = sum_n C_t[n] * h_t[:, n]
        # We compute h_t = sum_{s<=t} (prod_{k=s+1}^{t} dA_k) * dBu_s
        # using the log-domain cumsum trick for numerical stability.

        # log(dA): (L, d, N) — dA > 0 by construction (exp of reals)
        log_dA = dt_A  # already the log (before exp): A*dt, which is <=0

        # Cumulative log-A from position s to t:  cum_log_dA[t] = sum_{k=1}^{t} log_dA[k]
        cum_log_dA = torch.cumsum(log_dA, dim=0)      # (L, d, N)

        # Contribution of input at step s to output at step t (s<=t):
        #   coeff[t,s] = exp(cum_log_dA[t] - cum_log_dA[s])
        # h_t = sum_{s=0}^{t} coeff[t,s] * dBu[s]  (with coeff[t,t]=1)
        # Equivalent: h_t = exp(cum_log_dA[t]) * sum_{s=0}^{t} exp(-cum_log_dA[s]) * dBu[s]

        inv_A = torch.exp(-cum_log_dA)                     # (L, d, N)
        weighted  = inv_A * dBu                            # (L, d, N)
        prefix    = torch.cumsum(weighted, dim=0)          # (L, d, N)
        h = torch.exp(cum_log_dA) * prefix                 # (L, d, N)

        # y_t = sum_n C_t[n] * h_t[:, n]   → (L, d)
        y = (h * C.unsqueeze(1)).sum(dim=2)               # (L, d)
        y = y + D.unsqueeze(0) * u                        # skip
        return y.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, L, d_model). Returns same shape."""
        B, L, d = x.shape
        xz = self.in_proj(x)            # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Depthwise conv over L
        x_c = x_in.transpose(1, 2)             # (B, d_inner, L)
        x_c = self.conv1d(x_c)[..., :L]        # (B, d_inner, L)
        x_c = F.silu(x_c.transpose(1, 2))      # (B, L, d_inner)

        # SSM parameters
        ssm_in = self.x_proj(x_c)              # (B, L, 2N+1)
        dt_raw, B_mat, C_mat = ssm_in.split([1, self.d_state, self.d_state], dim=-1)
        dt_full = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())     # (d_inner, N) — negative

        # Process each batch element
        outs = []
        for b in range(B):
            y = self._selective_scan(
                x_c[b],
                dt_full[b],
                A,
                B_mat[b],
                C_mat[b],
                self.D.float(),
            )
            outs.append(y)
        y = torch.stack(outs, dim=0)    # (B, L, d_inner)
        y = y * F.silu(z)               # gating
        return self.out_proj(y)


def _patch_mamba_to_cpu(model: nn.Module, device: torch.device) -> None:
    """Replace all mamba_ssm.Mamba SSM modules with _PurePyMamba for CPU inference.

    This is a non-destructive hot-swap applied in-place; the original weights
    are transferred where compatible.  Only activates when device is CPU.
    """
    if device.type != "cpu":
        return

    from mamba_ssm import Mamba as _MambaSSM

    def _swap(parent: nn.Module, name: str, child: nn.Module) -> None:
        """Replace a mamba_ssm.Mamba child with a _PurePyMamba and copy weights."""
        d_model  = child.d_model
        d_state  = child.d_state
        d_conv   = child.d_conv
        expand   = child.expand

        new_ssm = _PurePyMamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # Transfer weights that have identical shapes
        with torch.no_grad():
            src_sd = {k: v.clone().contiguous() for k, v in child.state_dict().items()}
            dst_sd = new_ssm.state_dict()
            for k_dst in list(dst_sd.keys()):
                if k_dst in src_sd and src_sd[k_dst].shape == dst_sd[k_dst].shape:
                    dst_sd[k_dst] = src_sd[k_dst].clone()
        new_ssm.load_state_dict(dst_sd)
        setattr(parent, name, new_ssm)

    # Walk all modules
    for parent_name, parent_mod in model.named_modules():
        for child_name, child_mod in list(parent_mod.named_children()):
            if isinstance(child_mod, _MambaSSM):
                _swap(parent_mod, child_name, child_mod)
                print(f"    Patched {parent_name}.{child_name} → _PurePyMamba")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Gradient saliency hook
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _GradSaliency:
    """GradCAM-style gradient × activation saliency for sequence models."""

    def __init__(self, target_module: nn.Module) -> None:
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._fh = target_module.register_forward_hook(self._save_act)
        self._bh = target_module.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        if isinstance(out, torch.Tensor):
            self._acts = out.detach().clone()

    def _save_grad(self, module, grad_in, grad_out):
        if grad_out[0] is not None:
            self._grads = grad_out[0].detach().clone()

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def importance(self) -> torch.Tensor:
        """(batch, seq) importance normalised to [0,1]."""
        if self._acts is None or self._grads is None:
            raise RuntimeError("Forward + backward must be called first.")
        sal = (self._grads * self._acts).abs().sum(dim=-1)   # (B, seq)
        mx  = sal.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        return (sal / mx).cpu()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Temporal attention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_temporal_attention(
    ts_encoder: MambaEncoder,
    x: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gradient saliency over temporal Mamba token sequence.

    Returns (timesteps [T], attn [T]) with values in [0,1].
    """
    ts_encoder.eval()
    x = x.to(device).float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    _, _, T = x.shape
    patch = ts_encoder.input_dim

    # Hook onto the final block (before LayerNorm flattens the signal variances)
    hook = _GradSaliency(ts_encoder.blocks[-1])

    x_req = x.clone().requires_grad_(True)
    out   = ts_encoder(x_req)       # (1, emb_dim)
    score = out.pow(2).sum()        # scalar proxy
    ts_encoder.zero_grad()
    score.backward()

    token_imp = hook.importance()[0].numpy()   # (n_tokens,)
    hook.remove()

    # Map tokens → timesteps
    attn_ts = np.zeros(T, dtype=np.float32)
    for tok_i, imp in enumerate(token_imp):
        s = tok_i * patch;  e = min(s + patch, T)
        if s >= T: break
        attn_ts[s:e] = imp

    mx = attn_ts.max()
    if mx > 0:
        attn_ts /= mx

    return np.arange(T), attn_ts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Visual (RP lag) attention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _ActHook:
    def __init__(self):
        self.output: Optional[torch.Tensor] = None
    def __call__(self, m, i, o):
        self.output = o.detach().clone()


def extract_visual_attention(
    rp_encoder: UpperTriDiagRPEncoder,
    x: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lag attention for UpperTriDiag visual Mamba encoder.

    Combines:
      (a) Gradient saliency over lag tokens via final_norm hook.
      (b) Activation L2-norm per lag token (direct 'interest' signal).

    Returns lags, lag_imp (L-1,), rp_imp_map (L,L).
    """
    rp_encoder.eval()
    x = x.to(device).float()
    if x.ndim == 2:
        x = x.unsqueeze(0)

    core: _UpperTriDiagCore = rp_encoder.encoder
    L = rp_encoder.patch_len

    # ── (a) gradient saliency ─────────────────────────────────────────────
    hook_gs = _GradSaliency(core.blocks[-1])
    x_req   = x.clone().requires_grad_(True)
    out_gs  = rp_encoder(x_req)
    score   = out_gs.pow(2).sum()
    rp_encoder.zero_grad()
    score.backward()

    # hook_gs.importance(): (B*P, L-1)
    lag_imp_grad = hook_gs.importance().mean(dim=0).numpy()   # (L-1,)
    hook_gs.remove()

    # ── (b) activation magnitude ──────────────────────────────────────────
    hook_act = _ActHook()
    handle   = core.blocks[-1].register_forward_hook(hook_act)
    with torch.no_grad():
        rp_encoder(x)
    handle.remove()

    acts = hook_act.output   # (B*P, L-1, d_model)
    lag_imp_act = acts.norm(dim=-1).float().mean(dim=0).cpu().numpy()  # (L-1,)

    # Normalise both
    for arr in [lag_imp_grad, lag_imp_act]:
        mx = arr.max()
        if mx > 0: arr /= mx

    # Combine
    lag_imp = 0.5 * lag_imp_grad + 0.5 * lag_imp_act
    mx = lag_imp.max()
    if mx > 0: lag_imp /= mx

    # ── Build (L, L) map ──────────────────────────────────────────────────
    lags    = np.arange(1, L)
    rp_map  = np.zeros((L, L), dtype=np.float32)
    for k_idx, lag_k in enumerate(lags):
        if k_idx >= len(lag_imp): break
        imp = float(lag_imp[k_idx])
        for i in range(L - lag_k):
            j = i + lag_k
            rp_map[i, j] = imp
            rp_map[j, i] = imp

    return lags, lag_imp, rp_map


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Recurrence plot helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_recurrence_plot(x1d: np.ndarray, patch_len: int = 64) -> np.ndarray:
    L = min(patch_len, len(x1d))
    s = x1d[:L].astype(np.float32)
    r = np.abs(s[:, None] - s[None, :])
    mx = r.max()
    if mx > 0: r /= mx
    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Plot helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_BG        = "#0d1117"
_ATTN_CMAP = plt.cm.jet


def _dark(ax):
    ax.set_facecolor(_BG)
    ax.tick_params(colors="#888888", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")


def _ts_raw(ax, ts, title):
    _dark(ax)
    ax.plot(np.arange(len(ts)), ts, color="#4fc3f7", lw=1.1)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Time step", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Value", color="#aaaaaa", fontsize=8)


def _ts_attn(ax, ts, attn, title):
    _dark(ax)
    T = len(ts); t = np.arange(T)
    norm   = attn / (attn.max() + 1e-8)
    colors = _ATTN_CMAP(norm)
    bl     = ts.min() - 0.05 * (ts.max() - ts.min() + 1e-8)
    for i in range(T - 1):
        ax.fill_between([t[i], t[i+1]], [ts[i], ts[i+1]], [bl],
                        color=colors[i], alpha=0.7, linewidth=0)
    ax.plot(t, ts, color="white", lw=0.9, alpha=0.85)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Time step", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Value", color="#aaaaaa", fontsize=8)


def _tok_bar(ax, patch, T, tok_imp, title):
    _dark(ax)
    n  = len(tok_imp)
    cx = _ATTN_CMAP(tok_imp)
    ax.bar(np.arange(n), tok_imp, color=cx, width=0.7)
    labels = [f"[{t*patch}–{min((t+1)*patch, T)}]" for t in range(n)]
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Token window", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Importance", color="#aaaaaa", fontsize=8)


def _rp_raw(ax, rp, title):
    _dark(ax)
    L = rp.shape[0]
    s = rp.copy().astype(np.float32)
    s[np.tril(np.ones((L, L), bool))] = np.nan
    ax.imshow(1 - s, cmap="Blues_r", origin="upper", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Time j", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Time i", color="#aaaaaa", fontsize=8)


def _rp_attn(ax, rp, attn_map, title):
    _dark(ax)
    L = rp.shape[0]; mask = np.tril(np.ones((L, L), bool))
    rs = (1 - rp).copy().astype(np.float32); rs[mask] = np.nan
    as_ = attn_map.copy().astype(np.float32); as_[mask] = np.nan
    ax.imshow(rs,  cmap="Greys", origin="upper", vmin=0, vmax=1, aspect="auto")
    ax.imshow(as_, cmap="jet",   origin="upper", vmin=0, vmax=1, aspect="auto", alpha=0.65)
    step = max(1, (L - 1) // 6)
    for lag_k in range(step, L, step):
        rows = np.arange(L - lag_k); cols = rows + lag_k
        ax.plot(cols, rows, color="white", lw=0.4, alpha=0.3)
        mid = len(rows) // 2
        ax.text(cols[mid], rows[mid], f"k={lag_k}", color="white",
                fontsize=5.5, ha="center", va="center", alpha=0.75)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Time j", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Time i", color="#aaaaaa", fontsize=8)


def _lag_bar(ax, lags, lag_imp, title):
    _dark(ax)
    cx = _ATTN_CMAP(lag_imp)
    ax.bar(lags, lag_imp, color=cx, width=0.9)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Lag k", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Importance", color="#aaaaaa", fontsize=8)
    top5 = np.argsort(lag_imp)[-5:][::-1]
    for idx in top5:
        ax.axvline(x=lags[idx], color="white", lw=0.8, alpha=0.5, ls="--")
        yp = min(lag_imp[idx] + 0.04, 0.95)
        ax.text(lags[idx], yp, f"k={lags[idx]}", color="white", fontsize=5.5, ha="center")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main figure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def visualize_mamba_attention(
    ts_encoder, rp_encoder, x, ts_np, device,
    channel_idx=6, patch_len=64, save_path=None, title_prefix="BYOL Bimodal (Full)",
):
    if x.ndim == 2: x = x.unsqueeze(0)
    x = x.to(device).float()
    ts_1d = ts_np[:, channel_idx] if ts_np.ndim == 2 else ts_np

    print("  → Temporal attention …"); _, ts_attn = extract_temporal_attention(ts_encoder, x, device)
    print("  → Visual lag attention …"); lags, lag_imp, rp_imp = extract_visual_attention(rp_encoder, x, device)

    rp_raw = compute_recurrence_plot(ts_1d, patch_len=patch_len)
    T = len(ts_1d); pt = ts_encoder.input_dim; n_tok = T // pt
    tok_imp = np.array([ts_attn[t*pt:(t+1)*pt].mean() for t in range(n_tok)])
    if tok_imp.max() > 0: tok_imp /= tok_imp.max()

    plt.rcParams.update({"figure.facecolor": _BG, "text.color": "white", "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(18, 10), facecolor=_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.36,
                            left=0.06, right=0.97, top=0.88, bottom=0.08)

    _ts_raw(fig.add_subplot(gs[0,0]), ts_1d, "Raw Time Series\nETTh1 – OT channel")
    _ts_attn(fig.add_subplot(gs[0,1]), ts_1d, ts_attn,
             "Mamba Temporal Attention\n(Gradient × Activation Saliency)")
    _tok_bar(fig.add_subplot(gs[0,2]), pt, T, tok_imp,
             "Token Importance\n(Temporal Mamba Branch)")
    _rp_raw(fig.add_subplot(gs[1,0]), rp_raw,
            "Recurrence Plot (Upper △)\nETTh1 – OT channel")
    _rp_attn(fig.add_subplot(gs[1,1]), rp_raw, rp_imp,
             "RP + Mamba Lag Attention\n(Visual Branch – Upper △)")
    _lag_bar(fig.add_subplot(gs[1,2]), lags, lag_imp,
             "Lag Importance\n(Visual Mamba Branch)")

    sm = plt.cm.ScalarMappable(cmap="jet"); sm.set_clim(0, 1)
    cax = fig.add_axes([0.645, 0.09, 0.007, 0.36])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("Importance", color="white", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=6)
    cb.outline.set_edgecolor("#444444")

    elems = [Patch(facecolor=_ATTN_CMAP(0.95), label="High attention"),
             Patch(facecolor=_ATTN_CMAP(0.15), label="Low attention")]
    fig.legend(handles=elems, loc="upper right", ncol=2, bbox_to_anchor=(0.97, 0.945),
               framealpha=0.2, edgecolor="#555555", labelcolor="white", fontsize=8)

    fig.suptitle(
        f"{title_prefix} — Mamba Encoder Attention  •  "
        "HiddenMambaAttn-style (Ali et al., 2024)",
        color="white", fontsize=12, fontweight="bold", y=0.975)

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=_BG)
        print(f"  → Saved: {save_path}")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RP panel (HiddenMambaAttn 1×4 format)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_rp_panel(rp_encoder, x, ts_1d, device,
                  lags=None, lag_imp=None, rp_imp_map=None, save_path=None):
    """1×4 panel: Raw RP | Raw Attn | Smoothed | RP×Attn."""
    if lags is None:
        lags, lag_imp, rp_imp_map = extract_visual_attention(rp_encoder, x, device)
    L = rp_encoder.patch_len
    rp_raw    = compute_recurrence_plot(ts_1d, patch_len=L)
    rp_smooth = gaussian_filter(rp_imp_map, sigma=1.5)
    mx = rp_smooth.max()
    if mx > 0: rp_smooth /= mx

    mask = np.tril(np.ones((L, L), bool))
    def _up(a): o = a.copy().astype(np.float32); o[mask] = np.nan; return o

    panels = [_up(1 - rp_raw), _up(rp_imp_map), _up(rp_smooth),
              _up((1 - rp_raw) * rp_imp_map)]
    titles = ["Raw Recurrence Plot", "Raw Mamba Attention",
              "Smoothed Attention\n(Rollout style)", "RP × Mamba Attention\n(Mamba Attr)"]
    cmaps  = ["Blues_r", "jet", "hot", "inferno"]

    plt.rcParams.update({"figure.facecolor": _BG, "text.color": "white"})
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor=_BG)
    for ax, panel, cmap, title in zip(axes, panels, cmaps, titles):
        _dark(ax)
        ax.imshow(panel, cmap=cmap, origin="upper", vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Time j", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("Time i", color="#aaaaaa", fontsize=8)

    fig.suptitle(
        "UpperTriDiag Mamba Visual Encoder — Recurrence Plot Attention\n"
        "ETTh1 · OT channel · Upper-triangle anti-diagonal structure",
        color="white", fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=_BG)
        print(f"  → Saved RP panel: {save_path}")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_byol_bimodal_encoders(
    checkpoint_dir, device,
    input_dim=64, model_dim=256, depth=8, state_dim=16,
    conv_kernel=4, expand_factor=1.5, dropout=0.05,
    pooling="mean", embedding_dim=128,
):
    ckpt = Path(checkpoint_dir)
    ts_encoder = MambaEncoder(
        input_dim=input_dim, model_dim=model_dim, depth=depth,
        state_dim=state_dim, conv_kernel=conv_kernel, expand_factor=expand_factor,
        embedding_dim=embedding_dim, pooling=pooling, dropout=dropout,
    )
    sd = torch.load(ckpt / "time_series_best.pt", map_location="cpu", weights_only=False)
    ts_encoder.load_state_dict(sd["model_state_dict"])

    if device.type == "cpu":
        print("  CPU mode: patching Mamba SSM blocks to pure-PyTorch …")
        _patch_mamba_to_cpu(ts_encoder, device)

    ts_encoder = ts_encoder.to(device).eval()
    print(f"  ✓ Temporal encoder  [{sum(p.numel() for p in ts_encoder.parameters()):,} params]")

    rp_encoder = UpperTriDiagRPEncoder(
        patch_len=input_dim, d_model=model_dim, n_layers=depth,
        embedding_dim=embedding_dim, rp_mv_strategy="mean",
    )
    sd = torch.load(ckpt / "visual_encoder_best.pt", map_location="cpu", weights_only=False)
    rp_encoder.load_state_dict(sd["model_state_dict"])

    if device.type == "cpu":
        _patch_mamba_to_cpu(rp_encoder, device)

    rp_encoder = rp_encoder.to(device).eval()
    print(f"  ✓ Visual encoder    [{sum(p.numel() for p in rp_encoder.parameters()):,} params]")

    return ts_encoder, rp_encoder


def load_etth1(path, context_len=336, start=0):
    df   = pd.read_csv(path)
    cols = [c for c in df.columns if c != "date"]
    data = df[cols].values.astype(np.float32)
    seg  = data[start:start + context_len]
    mu   = seg.mean(0, keepdims=True)
    std  = seg.std(0, keepdims=True) + 1e-8
    x    = torch.from_numpy(((seg - mu) / std).T[np.newaxis]).float()  # (1,F,T)
    return seg, x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir",
        default="checkpoints/byol_bimodal_full/ts_byol_bimodal_full_lotsa_20260414_171343")
    p.add_argument("--etth1", default="ICML_datasets/ETT-small/ETTh1.csv")
    p.add_argument("--context-len", type=int, default=336)
    p.add_argument("--start-idx",   type=int, default=0)
    p.add_argument("--channel",     type=int, default=6)
    p.add_argument("--output",      default="results/mamba_attn_viz.png")
    p.add_argument("--device",      default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    root  = Path(__file__).resolve().parent.parent.parent
    ckpt  = Path(args.checkpoint_dir) if Path(args.checkpoint_dir).is_absolute() else root / args.checkpoint_dir
    eth   = Path(args.etth1)          if Path(args.etth1).is_absolute()          else root / args.etth1
    out   = Path(args.output)         if Path(args.output).is_absolute()         else root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nCheckpoint : {ckpt}\nETTh1      : {eth}\nOutput     : {out}")

    print("\nLoading encoders …")
    ts_enc, rp_enc = load_byol_bimodal_encoders(ckpt, device)

    print(f"\nLoading ETTh1 (T={args.context_len}, start={args.start_idx}) …")
    ts_np, x = load_etth1(eth, args.context_len, args.start_idx)
    print(f"  x: {x.shape}")

    print("\nComputing Mamba attention …")
    fig = visualize_mamba_attention(
        ts_enc, rp_enc, x, ts_np, device=device,
        channel_idx=args.channel, patch_len=64,
        save_path=str(out), title_prefix="BYOL Bimodal (Full)",
    )
    plt.close(fig)

    ts_1d = ts_np[:, args.channel] if ts_np.ndim == 2 else ts_np
    rp_panel_path = out.parent / (out.stem + "_rp_panel.png")
    print("\nBuilding RP panel …")
    lags, lag_imp, rp_imp_map = extract_visual_attention(rp_enc, x, device)
    fig2 = make_rp_panel(rp_enc, x, ts_1d, device,
                          lags=lags, lag_imp=lag_imp, rp_imp_map=rp_imp_map,
                          save_path=str(rp_panel_path))
    plt.close(fig2)

    print(f"\n✓  Done!\n   Main   : {out}\n   RP panel: {rp_panel_path}")


if __name__ == "__main__":
    main()
