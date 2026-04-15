"""Optuna hyperparameter search for the CLIP encoder.

Searches over training hyperparameters (lr, weight_decay, temperature,
noise_std, warmup_epochs) and model hyperparameters (model_dim,
embedding_dim, depth) using a fixed number of short trials on LOTSA data.

Each trial trains for ``--trial_epochs`` epochs and reports the best
validation loss. The best trial's config is written to
``--output_config`` for use in a full training run.

Usage
-----
    python src/experiments/optuna_hpo.py \
        --base_config src/configs/lotsa_best.yaml \
        --n_trials 30 \
        --trial_epochs 20 \
        --output_config src/configs/lotsa_hpo_best.yaml \
        --study_name clip_hpo
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Optional

import optuna
import torch
import yaml

# ── make src importable ───────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

import training_utils as tu
import util as u
from path_utils import resolve_path


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _build_loaders(cfg: dict, config_path: Path, seed: int):
    """Build train/val loaders from a config dict."""
    from cosine_training import _build_time_series_loaders
    data_cfg = cfg["data"]
    return _build_time_series_loaders(config_path, data_cfg, seed=seed)


def _run_trial(
    cfg: dict,
    config_path: Path,
    trial_epochs: int,
    device: torch.device,
    checkpoint_dir: Path,
    trial_number: int,
) -> float:
    """Train for trial_epochs and return best val loss."""

    seed = int(cfg.get("seed", 42))
    tu.set_seed(seed)

    train_loader, val_loader, _ = _build_loaders(cfg, config_path, seed)

    encoder = tu.build_encoder_from_config(cfg["model"])
    visual_encoder = tu.build_visual_encoder_from_config(cfg["model"])

    model_cfg = cfg["model"]
    projection_dim = int(model_cfg.get("model_dim", 128))
    projection_head = u.build_projection_head(encoder, output_dim=projection_dim)
    visual_projection_head = u.build_projection_head(visual_encoder, output_dim=projection_dim)

    train_cfg = cfg["training"]
    lr = float(train_cfg["learning_rate"])
    wd = float(train_cfg["weight_decay"])
    noise_std = float(train_cfg["noise_std"])
    alignment_strategy = str(train_cfg.get("alignment_strategy", "clip_symm"))

    params = (
        list(encoder.parameters())
        + list(visual_encoder.parameters())
        + list(projection_head.parameters())
        + list(visual_projection_head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    trial_ckpt = checkpoint_dir / f"trial_{trial_number}"
    trial_ckpt.mkdir(parents=True, exist_ok=True)

    best_loss: list[float] = []

    class _CaptureExperiment:
        """Minimal Comet shim — captures best val metric."""
        def log_metric(self, name, value, step=None): pass
        def log_parameter(self, name, value): pass
        def log_parameters(self, d): pass
        def end(self): pass

    # Monkey-patch run_clip_training to capture best val loss
    original = u.run_clip_training

    def _patched(**kwargs):
        # We override epochs to trial_epochs
        kwargs["epochs"] = trial_epochs
        kwargs["experiment"] = _CaptureExperiment()
        original(**kwargs)

    u.run_clip_training(
        encoder=encoder,
        visual_encoder=visual_encoder,
        projection_head=projection_head,
        visual_projection_head=visual_projection_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=trial_ckpt,
        epochs=trial_epochs,
        noise_std=noise_std,
        optimizer=optimizer,
        alignment_strategy=alignment_strategy,
        experiment=_CaptureExperiment(),
    )

    # Read best loss from saved checkpoint
    best_ckpt = trial_ckpt / "time_series_best.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        val_loss = float(state.get("loss", float("inf")))
    else:
        val_loss = float("inf")

    return val_loss


# ── objective ─────────────────────────────────────────────────────────────────

def make_objective(base_cfg: dict, config_path: Path, trial_epochs: int,
                   device: torch.device, checkpoint_dir: Path):

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)

        # ── training hyperparameters ──────────────────────────────────────────
        cfg["training"]["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-4, 5e-3, log=True)
        cfg["training"]["weight_decay"] = trial.suggest_float(
            "weight_decay", 1e-5, 1e-2, log=True)
        cfg["training"]["temperature"] = trial.suggest_float(
            "temperature", 0.05, 0.5, log=True)
        cfg["training"]["noise_std"] = trial.suggest_float(
            "noise_std", 0.001, 0.1, log=True)
        cfg["training"]["warmup_epochs"] = trial.suggest_int(
            "warmup_epochs", 2, 10)

        # ── model hyperparameters ─────────────────────────────────────────────
        model_dim = trial.suggest_categorical("model_dim", [64, 128, 256])
        cfg["model"]["model_dim"] = model_dim
        cfg["model"]["embedding_dim"] = trial.suggest_categorical(
            "embedding_dim", [32, 64, 128])
        cfg["model"]["depth"] = trial.suggest_int("depth", 2, 6)

        # ── data hyperparameters ──────────────────────────────────────────────
        cfg["data"]["cronos_kwargs"]["context_length"] = trial.suggest_categorical(
            "context_length", [96, 192, 336, 512])

        print(f"\n[Trial {trial.number}] params: {trial.params}")
        t0 = time.time()
        val_loss = _run_trial(cfg, config_path, trial_epochs, device,
                              checkpoint_dir, trial.number)
        elapsed = time.time() - t0
        print(f"[Trial {trial.number}] val_loss={val_loss:.6f}  ({elapsed/60:.1f} min)")
        return val_loss

    return objective


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Optuna HPO for CLIP encoder")
    p.add_argument("--base_config", required=True,
                   help="Base YAML config (lotsa_best.yaml)")
    p.add_argument("--n_trials", type=int, default=30,
                   help="Number of Optuna trials (default: 30)")
    p.add_argument("--trial_epochs", type=int, default=20,
                   help="Epochs per trial (default: 20)")
    p.add_argument("--output_config", default="src/configs/lotsa_hpo_best.yaml",
                   help="Where to write the best config")
    p.add_argument("--study_name", default="clip_hpo",
                   help="Optuna study name (used for SQLite storage)")
    p.add_argument("--storage", default=None,
                   help="Optuna storage URL. Defaults to SQLite in results/")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.base_config).resolve()
    base_cfg = _load_yaml(config_path)

    device = tu.prepare_device(args.device)
    print(f"Device: {device}")

    results_dir = Path("results/optuna")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("checkpoints/optuna")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    storage = args.storage or f"sqlite:///{results_dir}/{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    objective = make_objective(base_cfg, config_path, args.trial_epochs,
                               device, checkpoint_dir)

    print(f"Starting Optuna search: {args.n_trials} trials × {args.trial_epochs} epochs each")
    print(f"Storage: {storage}")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ── report ────────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best trial: #{best.number}  val_loss={best.value:.6f}")
    print(f"Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # ── write best config ─────────────────────────────────────────────────────
    best_cfg = copy.deepcopy(base_cfg)
    best_cfg["experiment_name"] = "ts_hpo_best_lotsa"
    best_cfg["training"]["learning_rate"] = best.params["learning_rate"]
    best_cfg["training"]["weight_decay"]  = best.params["weight_decay"]
    best_cfg["training"]["temperature"]   = best.params["temperature"]
    best_cfg["training"]["noise_std"]     = best.params["noise_std"]
    best_cfg["training"]["warmup_epochs"] = best.params["warmup_epochs"]
    best_cfg["training"]["epochs"]        = 100
    best_cfg["model"]["model_dim"]        = best.params["model_dim"]
    best_cfg["model"]["embedding_dim"]    = best.params["embedding_dim"]
    best_cfg["model"]["depth"]            = best.params["depth"]
    best_cfg["data"]["cronos_kwargs"]["context_length"] = best.params["context_length"]
    best_cfg["logging"]["checkpoint_dir"] = "../../checkpoints/hpo_best"

    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        yaml.dump(best_cfg, fh, default_flow_style=False, sort_keys=False)

    print(f"\nBest config written to: {output_path}")
    print("Run full training with:")
    print(f"  python src/cosine_training.py --config {output_path}")


if __name__ == "__main__":
    main()
