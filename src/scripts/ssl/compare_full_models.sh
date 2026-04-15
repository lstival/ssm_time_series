#!/bin/bash
# Compare full-size SSL models: BYOL-temporal, CLIP-best, SimCLR-full, GRAM-full
# Run after all probes are complete.
# Usage: bash compare_full_models.sh

source /home/WUR/stiva001/WUR/timeseries/bin/activate

python3 - <<'EOF'
import pandas as pd

results = {
    "BYOL-temporal":  "/home/WUR/stiva001/WUR/ssm_time_series/results/probe_byol_temporal/probe_lotsa_results.csv",
    "CLIP-best":      "/home/WUR/stiva001/WUR/ssm_time_series/results/probe_lotsa_ablation_best/probe_lotsa_results.csv",
    "SimCLR-full":    "/home/WUR/stiva001/WUR/ssm_time_series/results/simclr_full/probe_lotsa_results.csv",
    "GRAM-full":      "/home/WUR/stiva001/WUR/ssm_time_series/results/gram_full/probe_lotsa_results.csv",
}

datasets = ["ETTm1","ETTm2","ETTh1","ETTh2","weather","traffic","electricity","exchange_rate"]
horizons = [96, 192, 336, 720]

dfs = {}
missing = []
for name, path in results.items():
    try:
        df = pd.read_csv(path)
        df = df[df["dataset"].isin(datasets) & df["horizon"].isin(horizons)]
        dfs[name] = df
    except FileNotFoundError:
        missing.append(name)
        print(f"MISSING (probe not yet done): {name}")

if missing:
    print(f"\n{len(missing)} method(s) still pending. Re-run after probes complete.")

if not dfs:
    print("No results available yet.")
    exit()

# Per-dataset avg MSE (avg over horizons)
rows = []
for name, df in dfs.items():
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        if not sub.empty:
            rows.append({"method": name, "dataset": ds, "avg_mse": sub["mse"].mean()})
pt = pd.DataFrame(rows).pivot(index="dataset", columns="method", values="avg_mse")
pt = pt.reindex(datasets)
pt.loc["AVG"] = pt.mean()

print("\n=== FULL models — Avg MSE per dataset (avg over H=96/192/336/720) ===")
print(pt.round(4).to_string())

# Per-horizon avg MSE
rows2 = []
for name, df in dfs.items():
    for h in horizons:
        sub = df[df["horizon"] == h]
        if not sub.empty:
            rows2.append({"method": name, "horizon": h, "avg_mse": sub["mse"].mean()})
pt2 = pd.DataFrame(rows2).pivot(index="horizon", columns="method", values="avg_mse")
print("\n=== FULL models — Avg MSE per horizon ===")
print(pt2.round(4).to_string())

print("\n=== Overall Avg MSE ===")
for name, df in dfs.items():
    sub = df[df["dataset"].isin(datasets) & df["horizon"].isin(horizons)]
    print(f"  {name}: {sub['mse'].mean():.4f}")

# Save combined CSV
combined = pd.concat([df.assign(method=name) for name, df in dfs.items()], ignore_index=True)
out = "/home/WUR/stiva001/WUR/ssm_time_series/results/full_models_comparison.csv"
combined.to_csv(out, index=False)
print(f"\nCombined results saved to: {out}")
EOF
