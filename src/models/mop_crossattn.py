"""MoP cross-attention variants.

Variant D (original): cross-attn readout over temporal tokens; MoP routing
    from mean-pooled tokens; visual encoder unused.

Variant A (mop_crossattn_A): visual encoder tokens appended to temporal tokens
    as additional keys/values in cross-attn. Reintegrates the visual modality
    that variant D dropped.

Variant B (mop_crossattn_B): horizon-conditioned MoP routing. The horizon
    index is embedded and concatenated to z_pool before MoP routing, so each
    horizon gets a different prompt mixture.

Variant C (mop_crossattn_C): shared projection space + alignment loss. Both
    temporal and visual embeddings are projected to a shared space before
    fusion, with an auxiliary cosine alignment loss that keeps the two
    modalities consistent and makes fusion easier for the MoP.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.mop_forecast import RevIN, ModuleOfPrompts
except ImportError:
    from mop_forecast import RevIN, ModuleOfPrompts


# ── Shared building block ──────────────────────────────────────────────────────

class CrossAttnHead(nn.Module):
    """Single cross-attention head: horizon query attends over a token sequence."""

    def __init__(self, emb_dim: int, horizon: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.horizon = horizon
        self.query = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        self.proj = nn.Linear(emb_dim, horizon)

    def forward(self, tokens: torch.Tensor, prompt_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        tokens:      (N, T, emb_dim)
        prompt_bias: (N, emb_dim) optional additive bias
        Returns:     (N, horizon)
        """
        N = tokens.shape[0]
        q = self.query.expand(N, -1, -1)
        if prompt_bias is not None:
            q = q + prompt_bias.unsqueeze(1)
        attn_out, _ = self.attn(q, tokens, tokens)
        attn_out = self.norm(attn_out.squeeze(1))
        return self.proj(attn_out)


# ── Original (Variant D) ───────────────────────────────────────────────────────

class MoPCrossAttnModel(nn.Module):
    """Original cross-attn MoP (variant D). Visual encoder unused in forward."""

    def __init__(
        self,
        encoder: nn.Module,
        visual_encoder: nn.Module,
        emb_dim: int,
        num_prompts: int = 16,
        horizons: List[int] = [96, 192, 336, 720],
        target_features: int = 1,
        freeze_encoders: bool = True,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
        norm_mode: str = "revin",
        mop_hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.visual_encoder = visual_encoder
        self.horizons = sorted(set(int(h) for h in horizons))
        self.target_features = target_features
        self.norm_mode = norm_mode
        self.emb_dim = emb_dim

        if freeze_encoders:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.visual_encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            self.visual_encoder.eval()

        if norm_mode == "revin":
            self.revin = RevIN(target_features)
        else:
            self.revin = None

        model_dim = getattr(encoder, "model_dim", emb_dim)
        self.token_proj = nn.Linear(model_dim, emb_dim, bias=False)

        self.mop = ModuleOfPrompts(
            embedding_dim=emb_dim,
            num_prompts=num_prompts,
            hidden_dim=mop_hidden_dim,
            temperature=1.0,
        )
        self.mop_to_query = nn.Linear(mop_hidden_dim, emb_dim)

        self.heads = nn.ModuleDict({
            str(h): CrossAttnHead(emb_dim, h * target_features, n_heads=n_heads, dropout=attn_dropout)
            for h in self.horizons
        })

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.visual_encoder.eval()
        return self

    def _get_tokens(self, x_norm: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.encoder.forward_sequence(x_norm)
        return self.token_proj(hidden)

    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        B, C, L = x.shape
        x_ci = x.reshape(B * C, 1, L)

        if self.norm_mode == "revin":
            x_norm = self.revin(x_ci.permute(0, 2, 1), "norm").permute(0, 2, 1)
        else:
            x_norm = x_ci

        tokens = self._get_tokens(x_norm)
        z_pool = tokens.mean(dim=1)
        mop_out = self.mop(z_pool)
        prompt_bias = self.mop_to_query(mop_out)

        h_str = str(horizon)
        if h_str not in self.heads:
            raise ValueError(f"No head for horizon {horizon}")
        out = self.heads[h_str](tokens, prompt_bias)
        out_final = out.view(B * C, horizon, self.target_features)

        if self.norm_mode == "revin":
            out_final = self.revin(out_final, "denorm")
        return out_final

    def greedy_predict(self, x: torch.Tensor, target_horizon: int, context_length: int) -> torch.Tensor:
        if target_horizon in self.horizons:
            return self.forward(x, target_horizon)

        remaining = target_horizon
        predictions = []

        if self.norm_mode == "revin":
            x_norm = self.revin(x.permute(0, 2, 1), "norm").permute(0, 2, 1)
            saved_mean = self.revin.mean.clone()
            saved_stdev = self.revin.stdev.clone()
        else:
            x_norm = x.clone()
            saved_mean = saved_stdev = None

        while remaining > 0:
            valid = [h for h in self.horizons if h <= remaining]
            head_to_use = max(valid) if valid else min(self.horizons)
            tokens = self._get_tokens(x_norm)
            z_pool = tokens.mean(dim=1)
            mop_out = self.mop(z_pool)
            prompt_bias = self.mop_to_query(mop_out)
            out = self.heads[str(head_to_use)](tokens, prompt_bias)
            step_pred = out.view(x_norm.shape[0], head_to_use, self.target_features)
            step_pred = step_pred[:, :remaining, :]
            predictions.append(step_pred)
            remaining -= head_to_use
            if remaining > 0:
                x_norm = torch.cat([x_norm, step_pred.transpose(1, 2)], dim=2)
                x_norm = x_norm[:, :, -context_length:]

        raw_pred = torch.cat(predictions, dim=1)
        if self.norm_mode == "revin" and saved_mean is not None:
            self.revin.mean = saved_mean
            self.revin.stdev = saved_stdev
            raw_pred = self.revin(raw_pred, "denorm")
        return raw_pred


# ── Variant A: visual tokens as additional keys/values ────────────────────────

class MoPCrossAttnModelA(nn.Module):
    """Variant A: visual encoder tokens appended to temporal tokens.

    The visual encoder produces a single embedding per series. We treat it
    as one extra token prepended to the temporal token sequence, so the
    cross-attention heads can directly attend to visual pattern information
    (recurrence-plot structure) alongside the temporal hidden states.
    """

    def __init__(
        self,
        encoder: nn.Module,
        visual_encoder: nn.Module,
        emb_dim: int,
        num_prompts: int = 16,
        horizons: List[int] = [96, 192, 336, 720],
        target_features: int = 1,
        freeze_encoders: bool = True,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
        norm_mode: str = "revin",
        mop_hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.visual_encoder = visual_encoder
        self.horizons = sorted(set(int(h) for h in horizons))
        self.target_features = target_features
        self.norm_mode = norm_mode
        self.emb_dim = emb_dim

        if freeze_encoders:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.visual_encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            self.visual_encoder.eval()

        if norm_mode == "revin":
            self.revin = RevIN(target_features)
        else:
            self.revin = None

        model_dim = getattr(encoder, "model_dim", emb_dim)
        self.token_proj = nn.Linear(model_dim, emb_dim, bias=False)

        # Project visual embedding (emb_dim) to emb_dim — identity if same dim,
        # but explicit so we can handle visual_enc with different output dim.
        vis_dim = getattr(visual_encoder, "embedding_dim", emb_dim)
        self.visual_proj = nn.Linear(vis_dim, emb_dim, bias=False)

        self.mop = ModuleOfPrompts(
            embedding_dim=emb_dim,
            num_prompts=num_prompts,
            hidden_dim=mop_hidden_dim,
            temperature=1.0,
        )
        self.mop_to_query = nn.Linear(mop_hidden_dim, emb_dim)

        self.heads = nn.ModuleDict({
            str(h): CrossAttnHead(emb_dim, h * target_features, n_heads=n_heads, dropout=attn_dropout)
            for h in self.horizons
        })

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.visual_encoder.eval()
        return self

    def _get_tokens(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Temporal tokens + 1 visual token prepended. Returns (N, T+1, emb_dim)."""
        with torch.no_grad():
            hidden = self.encoder.forward_sequence(x_norm)   # (N, T, model_dim)
            z_vis  = self.visual_encoder(x_norm)              # (N, vis_dim)
        t_tokens = self.token_proj(hidden)                    # (N, T, emb_dim)
        v_token  = self.visual_proj(z_vis).unsqueeze(1)       # (N, 1, emb_dim)
        return torch.cat([v_token, t_tokens], dim=1)          # (N, T+1, emb_dim)

    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        B, C, L = x.shape
        x_ci = x.reshape(B * C, 1, L)

        if self.norm_mode == "revin":
            x_norm = self.revin(x_ci.permute(0, 2, 1), "norm").permute(0, 2, 1)
        else:
            x_norm = x_ci

        tokens = self._get_tokens(x_norm)           # (B*C, T+1, emb_dim)
        z_pool = tokens.mean(dim=1)
        mop_out = self.mop(z_pool)
        prompt_bias = self.mop_to_query(mop_out)

        h_str = str(horizon)
        if h_str not in self.heads:
            raise ValueError(f"No head for horizon {horizon}")
        out = self.heads[h_str](tokens, prompt_bias)
        out_final = out.view(B * C, horizon, self.target_features)

        if self.norm_mode == "revin":
            out_final = self.revin(out_final, "denorm")
        return out_final

    def greedy_predict(self, x: torch.Tensor, target_horizon: int, context_length: int) -> torch.Tensor:
        if target_horizon in self.horizons:
            return self.forward(x, target_horizon)

        remaining = target_horizon
        predictions = []

        if self.norm_mode == "revin":
            x_norm = self.revin(x.permute(0, 2, 1), "norm").permute(0, 2, 1)
            saved_mean  = self.revin.mean.clone()
            saved_stdev = self.revin.stdev.clone()
        else:
            x_norm = x.clone()
            saved_mean = saved_stdev = None

        while remaining > 0:
            valid = [h for h in self.horizons if h <= remaining]
            head_to_use = max(valid) if valid else min(self.horizons)
            tokens = self._get_tokens(x_norm)
            z_pool = tokens.mean(dim=1)
            mop_out = self.mop(z_pool)
            prompt_bias = self.mop_to_query(mop_out)
            out = self.heads[str(head_to_use)](tokens, prompt_bias)
            step_pred = out.view(x_norm.shape[0], head_to_use, self.target_features)
            step_pred = step_pred[:, :remaining, :]
            predictions.append(step_pred)
            remaining -= head_to_use
            if remaining > 0:
                x_norm = torch.cat([x_norm, step_pred.transpose(1, 2)], dim=2)
                x_norm = x_norm[:, :, -context_length:]

        raw_pred = torch.cat(predictions, dim=1)
        if self.norm_mode == "revin" and saved_mean is not None:
            self.revin.mean  = saved_mean
            self.revin.stdev = saved_stdev
            raw_pred = self.revin(raw_pred, "denorm")
        return raw_pred


# ── Variant B: horizon-conditioned MoP routing ────────────────────────────────

class MoPCrossAttnModelB(nn.Module):
    """Variant B: horizon index embedded and concatenated to z_pool before MoP routing.

    Each horizon gets a different prompt mixture — H=96 selects prompts
    biased toward short-range patterns; H=720 toward long-range. Previously
    all horizons shared the same routing, only differing in the final linear.
    """

    def __init__(
        self,
        encoder: nn.Module,
        visual_encoder: nn.Module,
        emb_dim: int,
        num_prompts: int = 16,
        horizons: List[int] = [96, 192, 336, 720],
        target_features: int = 1,
        freeze_encoders: bool = True,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
        norm_mode: str = "revin",
        mop_hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.visual_encoder = visual_encoder
        self.horizons = sorted(set(int(h) for h in horizons))
        self.target_features = target_features
        self.norm_mode = norm_mode
        self.emb_dim = emb_dim

        if freeze_encoders:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.visual_encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            self.visual_encoder.eval()

        if norm_mode == "revin":
            self.revin = RevIN(target_features)
        else:
            self.revin = None

        model_dim = getattr(encoder, "model_dim", emb_dim)
        self.token_proj = nn.Linear(model_dim, emb_dim, bias=False)

        # Learnable horizon embedding — one vector per horizon
        self.horizon_emb = nn.Embedding(len(self.horizons), emb_dim)
        self.horizon_to_idx = {h: i for i, h in enumerate(self.horizons)}

        # MoP routing input: z_pool (emb_dim) + horizon_emb (emb_dim)
        self.mop = ModuleOfPrompts(
            embedding_dim=emb_dim * 2,
            num_prompts=num_prompts,
            hidden_dim=mop_hidden_dim,
            temperature=1.0,
        )
        self.mop_to_query = nn.Linear(mop_hidden_dim, emb_dim)

        self.heads = nn.ModuleDict({
            str(h): CrossAttnHead(emb_dim, h * target_features, n_heads=n_heads, dropout=attn_dropout)
            for h in self.horizons
        })

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.visual_encoder.eval()
        return self

    def _get_tokens(self, x_norm: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.encoder.forward_sequence(x_norm)
        return self.token_proj(hidden)

    def _route(self, tokens: torch.Tensor, horizon: int) -> torch.Tensor:
        """Returns prompt_bias (N, emb_dim) conditioned on pooled tokens + horizon."""
        N = tokens.shape[0]
        z_pool = tokens.mean(dim=1)                                    # (N, emb_dim)
        h_idx  = torch.tensor(self.horizon_to_idx[horizon], device=tokens.device)
        h_emb  = self.horizon_emb(h_idx).unsqueeze(0).expand(N, -1)   # (N, emb_dim)
        z_cond = torch.cat([z_pool, h_emb], dim=-1)                   # (N, emb_dim*2)
        mop_out = self.mop(z_cond)                                     # (N, mop_hidden_dim)
        return self.mop_to_query(mop_out)                              # (N, emb_dim)

    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        B, C, L = x.shape
        x_ci = x.reshape(B * C, 1, L)

        if self.norm_mode == "revin":
            x_norm = self.revin(x_ci.permute(0, 2, 1), "norm").permute(0, 2, 1)
        else:
            x_norm = x_ci

        tokens      = self._get_tokens(x_norm)
        prompt_bias = self._route(tokens, horizon)

        h_str = str(horizon)
        if h_str not in self.heads:
            raise ValueError(f"No head for horizon {horizon}")
        out = self.heads[h_str](tokens, prompt_bias)
        out_final = out.view(B * C, horizon, self.target_features)

        if self.norm_mode == "revin":
            out_final = self.revin(out_final, "denorm")
        return out_final

    def greedy_predict(self, x: torch.Tensor, target_horizon: int, context_length: int) -> torch.Tensor:
        if target_horizon in self.horizons:
            return self.forward(x, target_horizon)

        remaining = target_horizon
        predictions = []

        if self.norm_mode == "revin":
            x_norm = self.revin(x.permute(0, 2, 1), "norm").permute(0, 2, 1)
            saved_mean  = self.revin.mean.clone()
            saved_stdev = self.revin.stdev.clone()
        else:
            x_norm = x.clone()
            saved_mean = saved_stdev = None

        while remaining > 0:
            valid = [h for h in self.horizons if h <= remaining]
            head_to_use = max(valid) if valid else min(self.horizons)
            tokens      = self._get_tokens(x_norm)
            prompt_bias = self._route(tokens, head_to_use)
            out         = self.heads[str(head_to_use)](tokens, prompt_bias)
            step_pred   = out.view(x_norm.shape[0], head_to_use, self.target_features)
            step_pred   = step_pred[:, :remaining, :]
            predictions.append(step_pred)
            remaining -= head_to_use
            if remaining > 0:
                x_norm = torch.cat([x_norm, step_pred.transpose(1, 2)], dim=2)
                x_norm = x_norm[:, :, -context_length:]

        raw_pred = torch.cat(predictions, dim=1)
        if self.norm_mode == "revin" and saved_mean is not None:
            self.revin.mean  = saved_mean
            self.revin.stdev = saved_stdev
            raw_pred = self.revin(raw_pred, "denorm")
        return raw_pred


# ── Variant C: shared projection space + alignment loss ───────────────────────

class MoPCrossAttnModelC(nn.Module):
    """Variant C: temporal + visual projected to shared space, fused by addition.

    Both modality embeddings (mean-pooled temporal, visual) are mapped to a
    shared d_shared space before being summed and fed to MoP routing. An
    auxiliary cosine alignment loss (returned alongside the forecast loss)
    encourages the two projections to agree for the same series, making the
    MoP's routing task easier and the fused representation more stable.

    Training loop must use:
        pred, align_loss = model.forward_with_loss(x, horizon)
        loss = forecast_loss(pred, y) + alpha * align_loss
    """

    def __init__(
        self,
        encoder: nn.Module,
        visual_encoder: nn.Module,
        emb_dim: int,
        num_prompts: int = 16,
        horizons: List[int] = [96, 192, 336, 720],
        target_features: int = 1,
        freeze_encoders: bool = True,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
        norm_mode: str = "revin",
        mop_hidden_dim: int = 256,
        align_alpha: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.visual_encoder = visual_encoder
        self.horizons = sorted(set(int(h) for h in horizons))
        self.target_features = target_features
        self.norm_mode = norm_mode
        self.emb_dim = emb_dim
        self.align_alpha = align_alpha

        if freeze_encoders:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.visual_encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            self.visual_encoder.eval()

        if norm_mode == "revin":
            self.revin = RevIN(target_features)
        else:
            self.revin = None

        model_dim = getattr(encoder, "model_dim", emb_dim)
        self.token_proj = nn.Linear(model_dim, emb_dim, bias=False)

        vis_dim = getattr(visual_encoder, "embedding_dim", emb_dim)

        # Shared projection: both modalities → emb_dim
        self.ts_shared  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.vis_shared = nn.Linear(vis_dim,  emb_dim, bias=False)

        # MoP routing on fused (sum) representation
        self.mop = ModuleOfPrompts(
            embedding_dim=emb_dim,
            num_prompts=num_prompts,
            hidden_dim=mop_hidden_dim,
            temperature=1.0,
        )
        self.mop_to_query = nn.Linear(mop_hidden_dim, emb_dim)

        self.heads = nn.ModuleDict({
            str(h): CrossAttnHead(emb_dim, h * target_features, n_heads=n_heads, dropout=attn_dropout)
            for h in self.horizons
        })

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.visual_encoder.eval()
        return self

    def _get_tokens_and_vis(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (tokens, z_ts_shared, z_vis_shared)."""
        with torch.no_grad():
            hidden = self.encoder.forward_sequence(x_norm)   # (N, T, model_dim)
            z_vis  = self.visual_encoder(x_norm)              # (N, vis_dim)
        tokens       = self.token_proj(hidden)                # (N, T, emb_dim)
        z_ts_shared  = self.ts_shared(tokens.mean(dim=1))    # (N, emb_dim)
        z_vis_shared = self.vis_shared(z_vis)                 # (N, emb_dim)
        return tokens, z_ts_shared, z_vis_shared

    def _alignment_loss(self, z_ts: torch.Tensor, z_vis: torch.Tensor) -> torch.Tensor:
        """Cosine distance loss — 0 when perfectly aligned, up to 2 when opposite."""
        z_ts_n  = F.normalize(z_ts,  dim=-1)
        z_vis_n = F.normalize(z_vis, dim=-1)
        return (1.0 - (z_ts_n * z_vis_n).sum(dim=-1)).mean()

    def forward_with_loss(self, x: torch.Tensor, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (prediction, alignment_loss). Use during training."""
        B, C, L = x.shape
        x_ci = x.reshape(B * C, 1, L)

        if self.norm_mode == "revin":
            x_norm = self.revin(x_ci.permute(0, 2, 1), "norm").permute(0, 2, 1)
        else:
            x_norm = x_ci

        tokens, z_ts, z_vis = self._get_tokens_and_vis(x_norm)
        align_loss = self._alignment_loss(z_ts, z_vis)

        z_fused     = z_ts + z_vis                    # (N, emb_dim) — shared space sum
        mop_out     = self.mop(z_fused)
        prompt_bias = self.mop_to_query(mop_out)

        h_str = str(horizon)
        if h_str not in self.heads:
            raise ValueError(f"No head for horizon {horizon}")
        out = self.heads[h_str](tokens, prompt_bias)
        out_final = out.view(B * C, horizon, self.target_features)

        if self.norm_mode == "revin":
            out_final = self.revin(out_final, "denorm")
        return out_final, align_loss

    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        pred, _ = self.forward_with_loss(x, horizon)
        return pred

    def greedy_predict(self, x: torch.Tensor, target_horizon: int, context_length: int) -> torch.Tensor:
        if target_horizon in self.horizons:
            return self.forward(x, target_horizon)

        remaining = target_horizon
        predictions = []

        if self.norm_mode == "revin":
            x_norm = self.revin(x.permute(0, 2, 1), "norm").permute(0, 2, 1)
            saved_mean  = self.revin.mean.clone()
            saved_stdev = self.revin.stdev.clone()
        else:
            x_norm = x.clone()
            saved_mean = saved_stdev = None

        while remaining > 0:
            valid = [h for h in self.horizons if h <= remaining]
            head_to_use = max(valid) if valid else min(self.horizons)
            tokens, z_ts, z_vis = self._get_tokens_and_vis(x_norm)
            z_fused     = z_ts + z_vis
            mop_out     = self.mop(z_fused)
            prompt_bias = self.mop_to_query(mop_out)
            out         = self.heads[str(head_to_use)](tokens, prompt_bias)
            step_pred   = out.view(x_norm.shape[0], head_to_use, self.target_features)
            step_pred   = step_pred[:, :remaining, :]
            predictions.append(step_pred)
            remaining -= head_to_use
            if remaining > 0:
                x_norm = torch.cat([x_norm, step_pred.transpose(1, 2)], dim=2)
                x_norm = x_norm[:, :, -context_length:]

        raw_pred = torch.cat(predictions, dim=1)
        if self.norm_mode == "revin" and saved_mean is not None:
            self.revin.mean  = saved_mean
            self.revin.stdev = saved_stdev
            raw_pred = self.revin(raw_pred, "denorm")
        return raw_pred
