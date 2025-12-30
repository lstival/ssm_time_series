"""Run all benchmark scripts and emit a LaTeX table of Params (M) and GFLOPs."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

try:
    from .benchmark_utils import load_yaml
    from .measure_mamba_encoder import benchmark_models as benchmark_mamba
    from .measure_vision_models import benchmark_models as benchmark_vision
    from .measure_time_series_models import benchmark_models as benchmark_time_series
except ImportError:
    tools_dir = Path(__file__).resolve().parents[1]
    sys.path.append(str(tools_dir))
    from tools.benchmark_utils import load_yaml
    from tools.measure_mamba_encoder import benchmark_models as benchmark_mamba
    from tools.measure_vision_models import benchmark_models as benchmark_vision
    from tools.measure_time_series_models import benchmark_models as benchmark_time_series

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "benchmark_params_gflops.tex"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "benchmark_models.yaml"
_CONFIG = load_yaml(CONFIG_PATH)
_RUNNER_CFG = _CONFIG.get("runner", {})

_OUTPUT_SETTING = _RUNNER_CFG.get("output_path")
if _OUTPUT_SETTING is None:
    OUTPUT_PATH = DEFAULT_OUTPUT
else:
    _candidate_path = Path(str(_OUTPUT_SETTING))
    OUTPUT_PATH = _candidate_path if _candidate_path.is_absolute() else ROOT / _candidate_path

VERBOSE_DEFAULT = bool(_RUNNER_CFG.get("verbose", False))


def _rows_from_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for res in results:
        params = res.get("params")
        flops = res.get("flops")
        params_m = params / 1e6 if isinstance(params, (int, float)) else None
        flops_g = flops / 1e9 if isinstance(flops, (int, float)) else None
        rows.append({"label": str(res.get("label", "model")), "params_m": params_m, "flops_g": flops_g})
    return rows


def build_latex_table(rows: Sequence[Dict[str, object]]) -> str:
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\begin{tabular}{lcc}",
        "\\hline",
        "\\textbf{Model} & \\textbf{Params (M)} & \\textbf{GFLOPs} \\\\",
        "\\hline",
    ]

    for row in rows:
        label = row["label"]
        params_m = row.get("params_m")
        flops_g = row.get("flops_g")
        params_str = f"{params_m:.3f}" if isinstance(params_m, (int, float)) else "N/A"
        flops_str = f"{flops_g:.2f}" if isinstance(flops_g, (int, float)) else "N/A"
        lines.append(f"{label} & {params_str} & {flops_str} \\\\")

    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\caption{Comparison of model complexity with Params in millions (M) and FLOPs in GFLOPs.}",
            "\\label{tb:params_gflops}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def run_all(verbose: bool) -> List[Dict[str, object]]:
    combined: List[Dict[str, object]] = []
    combined.extend(_rows_from_results(benchmark_mamba(verbose=verbose)))
    combined.extend(_rows_from_results(benchmark_vision(verbose=verbose)))
    combined.extend(_rows_from_results(benchmark_time_series(verbose=verbose)))
    return combined


def main() -> None:
    rows = run_all(verbose=VERBOSE_DEFAULT)
    latex = build_latex_table(rows)

    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")

    print("Generated LaTeX table using configuration settings:\n")
    print(latex)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
