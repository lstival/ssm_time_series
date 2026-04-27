"""MoP (Mixture of Prompts) forecasting models for Zero-Shot scheduling."""

from __future__ import annotations
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Reversible Instance Normalization.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / self.affine_weight
        x = x * self.stdev
        x = x + self.mean
        return x


class FlexibleHead(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        head_type: str = 'linear', 
        use_ln: bool = False,
        dropout: float = 0.0,
        residual: bool = False
    ):
        super().__init__()
        self.head_type = head_type
        self.residual = residual
        
        layers = []
        if use_ln:
            layers.append(nn.LayerNorm(input_dim))
            
        if head_type == 'linear':
            layers.append(nn.Linear(input_dim, output_dim))
        elif head_type == 'mlp':
            hidden_dim = max(input_dim, output_dim // 2)
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            ])
        else:
            raise ValueError(f"Unknown head type: {head_type}")
            
        self.base = nn.Sequential(*layers)
        
        # Residual mapping if needed
        if residual:
            if input_dim != output_dim:
                self.res_proj = nn.Linear(input_dim, output_dim)
            else:
                self.res_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.residual:
            out = out + self.res_proj(x)
        return out


class ModuleOfPrompts(nn.Module):
    """
    Mixture of Prompts (MoP) module applied post-encoder.
    Learns K prompts and routes input embeddings to a blended prompt.
    """
    def __init__(
        self, 
        embedding_dim: int, 
        num_prompts: int = 8, 
        hidden_dim: int = 512,
        temperature: float = 1.0,
        scale_cond: bool = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_prompts = num_prompts
        self.temperature = temperature
        self.scale_cond = scale_cond
        
        # Learnable prompt components
        self.prompt_keys = nn.Parameter(torch.randn(num_prompts, embedding_dim) * 0.02)
        self.prompt_values = nn.Parameter(torch.randn(num_prompts, embedding_dim) * 0.02)
        
        # Projection after combining embedding and prompt
        proj_input_dim = embedding_dim * 2
        if scale_cond:
            proj_input_dim += 2 # mean and std
            
        self.proj = nn.Sequential(
            nn.Linear(proj_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, z: torch.Tensor, stats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z: (B, D) encoder embeddings
        stats: (B, 2) optional mean/std
        Returns: (B, hidden_dim) prompted embeddings
        """
        # Routing: compute attention from z to prompt keys
        scale = self.embedding_dim ** 0.5
        scores = torch.matmul(z, self.prompt_keys.t()) / (scale * self.temperature)
        routing_weights = F.softmax(scores, dim=-1)             # (B, num_prompts)
        
        # Blended prompt for each sample
        blended_prompt = torch.matmul(routing_weights, self.prompt_values)  # (B, D)
        
        # Concatenate and project
        to_cat = [z, blended_prompt]
        if self.scale_cond and stats is not None:
            to_cat.append(stats)
            
        z_combined = torch.cat(to_cat, dim=-1)
        return self.proj(z_combined)


class MoPForecastModel(nn.Module):
    """
    Combines frozen dual encoders with a trainable MoP and multiple specific horizon heads.
    Supports flexible normalization (RevIN, MinMax) and head architectures.
    """
    def __init__(
        self,
        encoder: nn.Module,
        visual_encoder: nn.Module,
        input_dim: int,
        hidden_dim: int = 512,
        num_prompts: int = 8,
        horizons: List[int] = [96, 192, 336, 720],
        target_features: int = 1,
        freeze_encoders: bool = True,
        # Flexible options
        norm_mode: str = 'revin', # 'revin', 'minmax', 'identity', 'global'
        head_type: str = 'linear', # 'linear', 'mlp'
        use_ln_head: bool = False,
        residual_head: bool = False,
        temperature: float = 1.0,
        scale_cond: bool = False,
        learnable_scale: bool = False,
        dropout: float = 0.0,
        fusion_mode: str = 'concat',  # 'concat' or 'film'
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.visual_encoder = visual_encoder
        self.horizons = sorted(set(int(h) for h in horizons))
        self.target_features = int(target_features)
        
        self.norm_mode = norm_mode
        self.learnable_scale = learnable_scale
        self.fusion_mode = fusion_mode

        # FiLM: z_v gates z_e per-channel; input_dim is enc_dim*2 for concat, enc_dim for film
        enc_dim = input_dim // 2 if fusion_mode == 'concat' else input_dim
        if fusion_mode == 'film':
            self.film_gate = nn.Linear(enc_dim, enc_dim, bias=True)
            mop_input_dim = enc_dim
        else:
            mop_input_dim = input_dim

        if freeze_encoders:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.visual_encoder.eval()

        # Normalization
        if norm_mode == 'revin':
            self.revin = RevIN(target_features)
        else:
            self.revin = None

        # MoP Module
        self.mop = ModuleOfPrompts(
            embedding_dim=mop_input_dim,
            num_prompts=num_prompts,
            hidden_dim=hidden_dim,
            temperature=temperature,
            scale_cond=scale_cond
        )
        
        # Fixed-horizon prediction heads
        self.heads = nn.ModuleDict({
            str(h): FlexibleHead(
                hidden_dim, h * target_features, 
                head_type=head_type, use_ln=use_ln_head, 
                dropout=dropout, residual=residual_head
            )
            for h in self.horizons
        })

        if learnable_scale:
            self.horizon_scales = nn.ParameterDict({
                str(h): nn.Parameter(torch.ones(1)) for h in self.horizons
            })

    def train(self, mode: bool = True):
        super().train(mode)
        # Always keep encoders frozen
        self.encoder.eval()
        self.visual_encoder.eval()
        return self

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts embeddings from frozen encoders and fuses them.

        fusion_mode='concat': z = [z_e | z_v]  (B*C, 2D)
        fusion_mode='film':   z = z_e * sigmoid(W @ z_v)  (B*C, D)
            z_v is computed once over the full multivariate input (B, C, L) so it
            carries cross-channel context; z_e remains channel-independent.
        """
        B, C, L = x.shape
        x_reshaped = x.reshape(B * C, 1, L)

        with torch.no_grad():
            ze = self.encoder(x_reshaped)          # (B*C, D)
            if self.fusion_mode == 'film':
                # Visual encoder sees full multivariate context via Global-L2 RP
                # Input shape (B, C, L) → visual encoder reduces to one embedding per sample
                zv_global = self.visual_encoder(x)  # (B, D) — one embedding per batch item
                # Broadcast to all channels: each channel receives the same cross-variate context
                zv = zv_global.repeat_interleave(C, dim=0)  # (B*C, D)
            else:
                zv = self.visual_encoder(x_reshaped)  # (B*C, D) — per-channel CI

        if self.fusion_mode == 'film':
            gate = torch.sigmoid(self.film_gate(zv))  # (B*C, D)
            z = ze * gate
        else:
            z = torch.cat([ze, zv], dim=-1)            # (B*C, 2D)
        return z

    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Forward pass for a specific target horizon.
        x: (B, C, L)
        Returns: (B, horizon, target_features)
        """
        B, C, L = x.shape
        stats = None
        
        # Normalization
        if self.norm_mode == 'revin':
            # RevIN expects (B, L, C)
            x_norm = self.revin(x.permute(0, 2, 1), 'norm').permute(0, 2, 1)
            stats = torch.cat([self.revin.mean.squeeze(1), self.revin.stdev.squeeze(1)], dim=-1)
        elif self.norm_mode == 'minmax':
            x_min = x.min(dim=-1, keepdim=True)[0]
            x_max = x.max(dim=-1, keepdim=True)[0]
            x_norm = (x - x_min) / (x_max - x_min + 1e-8)
            stats = torch.cat([x_min.squeeze(-1), x_max.squeeze(-1)], dim=-1)
        else:
            x_norm = x
            
        z = self.forward_embeddings(x_norm)
        # z: (B*C, D)
        
        # If stats available, repeat for channels
        if stats is not None:
            # stats is (B, 2)
            stats = stats.repeat_interleave(C, dim=0) # (B*C, 2)
            
        z_prompted = self.mop(z, stats) # (B*C, hidden_dim)
        
        h_str = str(horizon)
        if h_str not in self.heads:
            raise ValueError(f"No head for horizon {horizon}")
            
        out = self.heads[h_str](z_prompted) # (B*C, horizon * features)
        
        if self.learnable_scale:
            out = out * self.horizon_scales[h_str]
            
        out = out.view(B, C, horizon, self.target_features).permute(0, 2, 1, 3).squeeze(-1)
        # Assuming target_features=1 for most benchmarks, output is (B, horizon, C)
        # Re-permute to (B, horizon, C) is standard in some evaluations, 
        # but let's stick to the view that returns (B*C, horizon, features) and then reshape
        
        out_final = out.reshape(B * C, horizon, self.target_features)
        
        # Denormalization
        if self.norm_mode == 'revin':
            # out_final is (B*C, horizon, 1). revin expects (B, horizon, 1)
            # We need to expand revin stats to match B*C if they weren't already
            out_final = self.revin(out_final, 'denorm')
        elif self.norm_mode == 'minmax':
            x_min_ch = x_min.reshape(B * C, 1, 1)
            x_max_ch = x_max.reshape(B * C, 1, 1)
            out_final = out_final * (x_max_ch - x_min_ch + 1e-8) + x_min_ch
            
        return out_final # (B*C, horizon, target_features)

    def greedy_predict(self, x: torch.Tensor, target_horizon: int, context_length: int) -> torch.Tensor:
        """
        Zero-Shot Greedy Scheduling.
        x: (B*C, 1, context_length) input.
        """
        # Note: greedy predict here assumes x is already (B*C, 1, L) or similar
        # and it DOES NOT apply internal normalization inside the loop if it's already normalized outside?
        # Actually MoP model should probably handle its own normalization if greedy_predict calls forward.
        
        if target_horizon in self.horizons:
            # We need to be careful with x shape here. forward expects (B, C, L)
            # If x is (N, 1, L), it treats it as B=N, C=1.
            return self.forward(x, target_horizon)

        remaining = target_horizon
        current_x = x.clone() # (B*C, 1, L)
        predictions = []
        
        while remaining > 0:
            valid_heads = [h for h in self.horizons if h <= remaining]
            head_to_use = max(valid_heads) if valid_heads else min(self.horizons)
                
            pred = self.forward(current_x, head_to_use) # (B*C, head_to_use, 1)
            step_pred = pred[:, :remaining, :]
            predictions.append(step_pred)
            
            remaining -= head_to_use
            if remaining > 0:
                step_pred_t = step_pred.transpose(1, 2) # [B*C, 1, L_pred]
                current_x = torch.cat([current_x, step_pred_t], dim=2)
                current_x = current_x[:, :, -context_length:]
                
        return torch.cat(predictions, dim=1)

