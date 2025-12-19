from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class TrainingProbe:
    """Reusable instrumentation for gradients, similarities, and augmentations."""

    def __init__(
        self,
        *,
        experiment: Optional[object],
        noise_std: float,
        min_grad_norm: float = 0.0,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.experiment = experiment
        self.noise_std = float(noise_std)
        self.min_grad_norm = max(0.0, float(min_grad_norm))
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

        self.step = 0
        self.similarity: Optional[float] = None
        self.allclose: Optional[bool] = None
        self.aug_stats: Optional[Tuple[float, float, float]] = None
        self.last_x_q: Optional[torch.Tensor] = None
        self.last_total_grad_norm: Optional[float] = None
        self.last_grad_scale: Optional[float] = None

    def record_similarity(self, q_proj: torch.Tensor, k_proj: torch.Tensor) -> None:
        with torch.no_grad():
            cosine = F.cosine_similarity(q_proj, k_proj, dim=1).mean().item()
            close = bool(torch.allclose(q_proj, k_proj, atol=1e-3, rtol=1e-3))
        self.similarity = cosine
        self.allclose = close

    def record_views(
        self,
        *,
        reference: Optional[torch.Tensor],
        noisy_view: torch.Tensor,
        augmented: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            noisy_delta = (augmented - noisy_view).abs()
            noisy_mean = float(noisy_delta.mean().item())
            noisy_max = float(noisy_delta.max().item())
            ref_delta_mean = float("nan")
            if reference is not None and reference.shape == augmented.shape:
                ref_delta = (augmented - reference).abs()
                ref_delta_mean = float(ref_delta.mean().item())
        self.aug_stats = (noisy_mean, noisy_max, ref_delta_mean)

    def enforce_grad_norms(self, optimizer: torch.optim.Optimizer) -> None:
        grads: list[torch.Tensor] = []
        total_sq = 0.0
        for group in optimizer.param_groups:
            for p in group["params"]:
                g = getattr(p, "grad", None)
                if g is None:
                    continue
                grads.append(g)
                total_sq += float(g.detach().pow(2).sum().item())

        total_norm = math.sqrt(total_sq) if total_sq > 0.0 else 0.0
        scale = 1.0

        # If gradients are tiny, scale them up to a floor.
        if self.min_grad_norm > 0.0 and 0.0 < total_norm < self.min_grad_norm:
            scale = self.min_grad_norm / (total_norm + 1e-12)
            for g in grads:
                g.mul_(scale)
            total_norm = self.min_grad_norm

        # If gradients explode, clip to a ceiling.
        if self.max_grad_norm is not None and self.max_grad_norm > 0.0 and len(grads) > 0:
            torch.nn.utils.clip_grad_norm_(grads, self.max_grad_norm)

        self.last_total_grad_norm = total_norm
        self.last_grad_scale = scale

    def log_step(self, tracked: Sequence[Tuple[str, nn.Parameter]]) -> None:
        grads: dict[str, float] = {}
        weights: dict[str, float] = {}

        for name, param in tracked:
            with torch.no_grad():
                weights[name] = float(param.detach().norm().item())
            grad = param.grad
            grads[name] = float(grad.detach().norm().item()) if grad is not None else float("nan")

        parts = [f"[monitor] step {self.step}"]
        for name in grads.keys():
            parts.append(f"{name}: w={weights[name]:.6f} g={grads[name]:.6f}")

        if self.last_total_grad_norm is not None:
            parts.append(f"grad_norm_total: {self.last_total_grad_norm:.6f}")
            if self.last_grad_scale is not None and self.last_grad_scale != 1.0:
                parts.append(f"grad_scale: {self.last_grad_scale:.3f}")

        if self.similarity is not None:
            parts.append(f"cosine_mean: {self.similarity:.6f}")
            parts.append(f"allclose: {self.allclose}")

        if self.aug_stats is not None:
            noisy_mean, noisy_max, ref_mean = self.aug_stats
            parts.append(f"aug_mean_delta(noisy): {noisy_mean:.6f}")
            parts.append(f"aug_max_delta(noisy): {noisy_max:.6f}")
            if not math.isnan(ref_mean):
                parts.append(f"aug_mean_delta(original): {ref_mean:.6f}")

        print(" | ".join(parts))

        if self.experiment is not None:
            for name, value in weights.items():
                self.experiment.log_metric(f"weight_norm/{name.replace('.', '_')}", value, step=self.step)
            for name, value in grads.items():
                self.experiment.log_metric(f"grad_norm/{name.replace('.', '_')}", value, step=self.step)

            if self.last_total_grad_norm is not None:
                self.experiment.log_metric("grad_norm/total", self.last_total_grad_norm, step=self.step)
            if self.last_grad_scale is not None:
                self.experiment.log_metric("grad_scale", self.last_grad_scale, step=self.step)
            if self.similarity is not None:
                self.experiment.log_metric("similarity/cosine_mean", self.similarity, step=self.step)
                self.experiment.log_metric("similarity/allclose", 1.0 if self.allclose else 0.0, step=self.step)
            if self.aug_stats is not None:
                noisy_mean, noisy_max, ref_mean = self.aug_stats
                self.experiment.log_metric("augment/noisy_mean_delta", noisy_mean, step=self.step)
                self.experiment.log_metric("augment/noisy_max_delta", noisy_max, step=self.step)
                if not math.isnan(ref_mean):
                    self.experiment.log_metric("augment/original_mean_delta", ref_mean, step=self.step)

        self.step += 1
        self.similarity = None
        self.allclose = None
        self.aug_stats = None
        self.last_x_q = None
        self.last_total_grad_norm = None
        self.last_grad_scale = None


class ProbeContext:
    """Patch util/training functions for logging, and optionally override CLIP loss."""

    def __init__(
        self,
        *,
        u_module,
        optimizer: torch.optim.Optimizer,
        probe: TrainingProbe,
        tracked_params: Sequence[Tuple[str, nn.Parameter]],
        loss_impl: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None,
    ) -> None:
        self.u = u_module
        self.optimizer = optimizer
        self.probe = probe
        self.tracked_params = tracked_params
        self.loss_impl = loss_impl
        self._orig: dict[str, object] = {}

    def __enter__(self):
        self._orig["make_positive_view"] = self.u.make_positive_view
        self._orig["clip_contrastive_loss"] = self.u.clip_contrastive_loss
        self._orig["reshape_multivariate_series"] = self.u.reshape_multivariate_series
        self._orig["optimizer_step"] = self.optimizer.step

        def make_positive_view(x: torch.Tensor, *args, **kwargs):
            augmented = self._orig["make_positive_view"](x, *args, **kwargs)  # type: ignore[misc]
            if torch.is_grad_enabled():
                self.probe.record_views(
                    reference=self.probe.last_x_q,
                    noisy_view=x.detach(),
                    augmented=augmented.detach(),
                )
            return augmented

        def clip_contrastive_loss(xq: torch.Tensor, xk: torch.Tensor, *, temperature: float = 0.07) -> torch.Tensor:
            if torch.is_grad_enabled():
                self.probe.record_similarity(xq.detach(), xk.detach())
            if self.loss_impl is not None:
                return self.loss_impl(xq, xk, float(temperature))
            return self._orig["clip_contrastive_loss"](xq, xk, temperature=temperature)  # type: ignore[misc]

        def reshape_multivariate_series(seq: torch.Tensor) -> torch.Tensor:
            result = self._orig["reshape_multivariate_series"](seq)  # type: ignore[misc]
            if torch.is_grad_enabled():
                self.probe.last_x_q = result.detach()
            return result

        def optimizer_step(*args, **kwargs):
            if torch.is_grad_enabled():
                self.probe.enforce_grad_norms(self.optimizer)
            out = self._orig["optimizer_step"](*args, **kwargs)  # type: ignore[misc]
            if torch.is_grad_enabled():
                self.probe.log_step(self.tracked_params)
            return out

        self.u.make_positive_view = make_positive_view
        self.u.clip_contrastive_loss = clip_contrastive_loss
        self.u.reshape_multivariate_series = reshape_multivariate_series
        self.optimizer.step = optimizer_step
        return self

    def __exit__(self, exc_type, exc, tb):
        self.u.make_positive_view = self._orig["make_positive_view"]
        self.u.clip_contrastive_loss = self._orig["clip_contrastive_loss"]
        self.u.reshape_multivariate_series = self._orig["reshape_multivariate_series"]
        self.optimizer.step = self._orig["optimizer_step"]
        return False