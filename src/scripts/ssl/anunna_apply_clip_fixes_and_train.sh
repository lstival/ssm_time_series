#!/bin/bash
#SBATCH --comment=clip_full_fixed_train
#SBATCH --time=2880
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/clip_full/train_fixed_%j.out
#SBATCH --error=logs/clip_full/train_fixed_%j.err
#SBATCH --job-name=clip_full_fixed
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/clip_full

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

REPO_ROOT=$(pwd)
SRC="${REPO_ROOT}/src"
DIAG_DIR="${REPO_ROOT}/results/diagnose_clip_nan"

echo "════════════════════════════════════════════════════════════════"
echo "CLIP Full — Analyse diagnostics + Apply fixes + Retrain"
echo "Job: $SLURM_JOB_ID  |  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Analyse diagnostic results and apply the appropriate fix to util.py
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "▶ Step 1: Analysing diagnostic results..."

python3 - <<'PYEOF'
import sys, csv, math
from pathlib import Path

diag = Path("results/diagnose_clip_nan")
summary = {}

for run in ["run1_baseline", "run2_amp", "run3_gradclip"]:
    batch_csv = diag / run / "batch_diagnostics.csv"
    epoch_csv = diag / run / "epoch_summary.csv"
    if not batch_csv.exists():
        print(f"  [{run}] — no results found, skipping")
        continue

    with open(batch_csv) as f:
        rows = list(csv.DictReader(f))

    nan_rows = [r for r in rows if r["loss_nan"] == "True"]
    total = len(rows)
    nan_count = len(nan_rows)

    # Grad norm stats per epoch
    from collections import defaultdict
    grad_by_epoch = defaultdict(list)
    logit_by_epoch = defaultdict(list)
    for r in rows:
        ep = int(r["epoch"])
        gn = float(r["grad_norm_enc"]) if r["grad_norm_enc"] else 0
        lx = float(r["logit_max"]) if r["logit_max"] else 0
        grad_by_epoch[ep].append(gn)
        logit_by_epoch[ep].append(lx)

    first_nan = nan_rows[0] if nan_rows else None
    print(f"\n{'='*60}")
    print(f"  Run: {run}")
    print(f"  NaN batches: {nan_count}/{total}")
    if first_nan:
        print(f"  First NaN → epoch={first_nan['epoch']} batch={first_nan['batch']}")
        print(f"    input_nan={first_nan['input_nan']}  input_inf={first_nan['input_inf']}")
        print(f"    q_nan={first_nan['q_nan']}  k_nan={first_nan['k_nan']}")
        print(f"    logit_max={first_nan['logit_max']}  loss={first_nan['loss']}")
        print(f"    grad_norm_enc={first_nan['grad_norm_enc']}")
        print(f"    grad_norm_vis={first_nan['grad_norm_vis']}")
        print(f"    scaler_scale={first_nan['scaler_scale']}")

    print(f"  Grad norm (enc) per epoch:")
    for ep in sorted(grad_by_epoch):
        gns = grad_by_epoch[ep]
        print(f"    epoch {ep:2d}: mean={sum(gns)/len(gns):.3f}  max={max(gns):.3f}")

    print(f"  Logit max per epoch:")
    for ep in sorted(logit_by_epoch):
        lxs = logit_by_epoch[ep]
        print(f"    epoch {ep:2d}: mean={sum(lxs)/len(lxs):.3f}  max={max(lxs):.3f}")

    summary[run] = {"nan_count": nan_count, "total": total}

# Conclusion
print("\n" + "="*60)
print("CONCLUSION:")
r1 = summary.get("run1_baseline", {}).get("nan_count", -1)
r2 = summary.get("run2_amp",      {}).get("nan_count", -1)
r3 = summary.get("run3_gradclip", {}).get("nan_count", -1)

if r3 == 0 and r1 > 0:
    print("  ✅ Gradient clipping FIXES the NaN → root cause: gradient explosion")
elif r2 == 0 and r1 > 0:
    print("  ✅ AMP FIXES the NaN → root cause: fp32 overflow in large model")
elif r1 == 0:
    print("  ℹ️  No NaN in baseline — may need more epochs or different seed")
else:
    print("  ⚠️  NaN persists across all runs — deeper issue (data, loss, or architecture)")

PYEOF

echo ""
echo "▶ Step 2: Applying fixes to src/util.py and src/cosine_training.py..."

python3 - <<'PYEOF'
import re
from pathlib import Path

util_path = Path("src/util.py")
cosine_path = Path("src/cosine_training.py")
src = util_path.read_text()

# ── Fix 1: Add max_grad_norm and use_amp parameters to run_clip_training ──
old_sig = '''def run_clip_training(
    *,
    encoder: nn.Module,
    visual_encoder: nn.Module,
    projection_head: nn.Module,
    visual_projection_head: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    checkpoint_dir: Path,
    epochs: int = 2,
    noise_std: float = 0.01,
    optimizer: Optional[Optimizer] = None,
    initial_epoch: int = 0,
    best_loss: Optional[float] = None,
    experiment: Optional[Any] = None,  # comet_ml.Experiment
    alignment_strategy: str = "clip_symm",
) -> None:
    """Training loop that optimizes a CLIP-style contrastive objective."""'''

new_sig = '''def run_clip_training(
    *,
    encoder: nn.Module,
    visual_encoder: nn.Module,
    projection_head: nn.Module,
    visual_projection_head: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    checkpoint_dir: Path,
    epochs: int = 2,
    noise_std: float = 0.01,
    optimizer: Optional[Optimizer] = None,
    initial_epoch: int = 0,
    best_loss: Optional[float] = None,
    experiment: Optional[Any] = None,  # comet_ml.Experiment
    alignment_strategy: str = "clip_symm",
    max_grad_norm: Optional[float] = 1.0,
    use_amp: bool = False,
) -> None:
    """Training loop that optimizes a CLIP-style contrastive objective."""'''

assert old_sig in src, "ERROR: signature not found in util.py — check manually"
src = src.replace(old_sig, new_sig, 1)
print("  ✅ Added max_grad_norm and use_amp to run_clip_training signature")

# ── Fix 2: Add GradScaler init after alignment_strategy selection ──
old_strategy = '''    if alignment_strategy == "cosine_mse":
        _loss_fn = cosine_mse_loss
    else:
        _loss_fn = clip_contrastive_loss
    print(f"Alignment strategy: {alignment_strategy}")'''

new_strategy = '''    if alignment_strategy == "cosine_mse":
        _loss_fn = cosine_mse_loss
    else:
        _loss_fn = clip_contrastive_loss
    print(f"Alignment strategy: {alignment_strategy}")

    amp_enabled = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    print(f"AMP: {amp_enabled}  |  max_grad_norm: {max_grad_norm}")'''

assert old_strategy in src, "ERROR: strategy block not found in util.py"
src = src.replace(old_strategy, new_strategy, 1)
print("  ✅ Added GradScaler init")

# ── Fix 3: Replace backward/optimizer.step in the fast-path LOTSA block ──
old_backward = '''                optimizer.zero_grad(set_to_none=True)

                loss.backward()
                optimizer.step()

                batch_loss = float(loss.item())
                epoch_loss += batch_loss
                batches += 1
                pbar.set_postfix(batch_loss=f"{batch_loss:.4f}", avg_loss=f"{(epoch_loss / batches):.4f}")'''

new_backward = '''                batch_loss = float(loss.item())
                if not (batch_loss == batch_loss):  # NaN check — skip corrupted batch
                    continue

                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    scaler.scale(loss).backward()
                    if max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(params, max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(params, max_grad_norm)
                    optimizer.step()

                epoch_loss += batch_loss
                batches += 1
                pbar.set_postfix(batch_loss=f"{batch_loss:.4f}", avg_loss=f"{(epoch_loss / batches):.4f}")'''

assert old_backward in src, "ERROR: backward block not found in util.py"
src = src.replace(old_backward, new_backward, 1)
print("  ✅ Added NaN skip + AMP + grad clipping to training step")

util_path.write_text(src)
print(f"  ✅ util.py saved")

# ── Fix cosine_training.py: pass max_grad_norm and use_amp to run_clip_training ──
csrc = cosine_path.read_text()

old_call = '''    u.run_clip_training(
        encoder=encoder,
        visual_encoder=visual_encoder,
        projection_head=projection_head,
        visual_projection_head=visual_projection_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        noise_std=noise_std,
        optimizer=optimizer,
        initial_epoch=initial_epoch,
        best_loss=best_loss,
        experiment=experiment,
        alignment_strategy=alignment_strategy,
    )'''

new_call = '''    max_grad_norm_cfg = training_cfg.get("max_grad_norm", 1.0)
    max_grad_norm = float(max_grad_norm_cfg) if max_grad_norm_cfg is not None else None
    use_amp = bool(training_cfg.get("use_amp", False))

    u.run_clip_training(
        encoder=encoder,
        visual_encoder=visual_encoder,
        projection_head=projection_head,
        visual_projection_head=visual_projection_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        noise_std=noise_std,
        optimizer=optimizer,
        initial_epoch=initial_epoch,
        best_loss=best_loss,
        experiment=experiment,
        alignment_strategy=alignment_strategy,
        max_grad_norm=max_grad_norm,
        use_amp=use_amp,
    )'''

assert old_call in csrc, "ERROR: run_clip_training call not found in cosine_training.py"
csrc = csrc.replace(old_call, new_call, 1)
cosine_path.write_text(csrc)
print(f"  ✅ cosine_training.py updated — now passes max_grad_norm and use_amp")
print("\n  All fixes applied successfully.")

PYEOF

echo ""
echo "▶ Step 3: Quick sanity check — import util.py"
python3 -c "import sys; sys.path.insert(0, 'src'); import util; print('  ✅ util.py imports OK')"

if [ $? -ne 0 ]; then
    echo "  ❌ util.py import failed — aborting training"
    exit 1
fi

echo ""
echo "▶ Step 4: Launching CLIP full training with fixes"
echo "   Config: ${SRC}/configs/lotsa_clip_full.yaml"
echo "   Fixes:  max_grad_norm=1.0  use_amp=true  NaN-skip=true"
echo ""

time python3 "${SRC}/cosine_training.py" \
    --config "${SRC}/configs/lotsa_clip_full.yaml"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"
