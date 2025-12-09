"""Compute clustering metrics on t-SNE coordinate exports and emit LaTeX tables."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import silhouette_samples

EPS = 1e-9

SRC_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = SRC_DIR / "configs"
CONFIG_ENV_VAR = "TSNE_CLUSTER_METRICS_CONFIG"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "tsne_clustering_metrics.yaml"


@dataclass(frozen=True)
class MetricsJobConfig:
    """Runtime configuration loaded from YAML."""

    tsne_files: List[Path]
    model_names: Optional[List[str]]
    model_config: Optional[Path]
    model_key: str
    model_name_field: str
    output_path: Optional[Path]
    table_caption: str
    table_label: Optional[str]


def determine_config_path() -> Path:
    env_value = os.environ.get(CONFIG_ENV_VAR)
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Configuration file specified via {CONFIG_ENV_VAR} not found: {candidate}"
        )
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH.resolve()
    raise FileNotFoundError(
        "No configuration file found. Set the environment variable "
        f"{CONFIG_ENV_VAR} or create {DEFAULT_CONFIG_PATH}."
    )


def _ensure_list(value: object) -> List[object]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def resolve_optional_path(base: Path, candidate: Optional[object]) -> Optional[Path]:
    if candidate is None:
        return None
    path = Path(str(candidate)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_required_path(base: Path, candidate: Optional[object], *, description: str) -> Path:
    path = resolve_optional_path(base, candidate)
    if path is None:
        raise ValueError(f"Missing required path for {description}")
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _get_section(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    section = payload.get(key)
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise TypeError(f"Section '{key}' must be a mapping in the configuration file")
    return section


def load_job_config(config_path: Path) -> MetricsJobConfig:
    payload = read_yaml(config_path)
    base_dir = config_path.parent

    tsne_section = _get_section(payload, "tsne") or payload
    raw_files = tsne_section.get("files") or payload.get("tsne_files")
    tsne_file_entries = _ensure_list(raw_files)
    if not tsne_file_entries:
        raise ValueError("Configuration must provide at least one t-SNE CSV under tsne.files")
    tsne_files = [
        resolve_required_path(base_dir, entry, description=f"t-SNE CSV #{idx + 1}")
        for idx, entry in enumerate(tsne_file_entries)
    ]

    models_section = _get_section(payload, "models")
    model_names_raw = models_section.get("names") or payload.get("model_names")
    model_names = [str(name) for name in _ensure_list(model_names_raw)] or None
    model_config_value = models_section.get("config") or payload.get("model_config")
    model_config = resolve_optional_path(base_dir, model_config_value)
    model_key = str(models_section.get("key") or payload.get("model_key") or "evaluation.models")
    model_name_field = str(
        models_section.get("name_field") or payload.get("model_name_field") or "name"
    )

    output_section = _get_section(payload, "output")
    output_path_value = output_section.get("path") or payload.get("output_path")
    output_path = resolve_optional_path(base_dir, output_path_value)
    table_caption = str(
        output_section.get("caption")
        or payload.get("table_caption")
        or "Clustering metrics for t-SNE projections."
    )
    table_label_raw = output_section.get("label") or payload.get("table_label")
    table_label = str(table_label_raw).strip() if table_label_raw else None

    return MetricsJobConfig(
        tsne_files=tsne_files,
        model_names=model_names,
        model_config=model_config,
        model_key=model_key,
        model_name_field=model_name_field,
        output_path=output_path,
        table_caption=table_caption,
        table_label=table_label,
    )


@dataclass(frozen=True)
class MetricDefinition:
    """Container describing how to compute and format a clustering metric."""

    name: str
    display_name: str
    higher_is_better: bool
    precision: int
    compute_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]]

    def format_value(self, value: float) -> str:
        if value is None or math.isnan(value):
            return "--"
        fmt = f"{{:.{self.precision}f}}"
        return fmt.format(value)


def compute_silhouette_by_label(coords: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2 or coords.shape[0] < 2:
        return {label: math.nan for label in unique_labels}
    try:
        sample_scores = silhouette_samples(coords, labels, metric="euclidean")
    except ValueError:
        return {label: math.nan for label in unique_labels}

    results: Dict[str, float] = {}
    for label in unique_labels:
        mask = labels == label
        count = int(np.count_nonzero(mask))
        if count < 2:
            results[label] = math.nan
        else:
            results[label] = float(np.mean(sample_scores[mask]))
    return results


def compute_cohesion_separation_ratio(coords: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    unique_labels = np.unique(labels)
    centroids: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}

    for label in unique_labels:
        mask = labels == label
        masks[label] = mask
        points = coords[mask]
        if points.size == 0:
            continue
        centroids[label] = points.mean(axis=0)

    results: Dict[str, float] = {}
    for label in unique_labels:
        mask = masks[label]
        points = coords[mask]
        count = int(np.count_nonzero(mask))
        if count == 0 or label not in centroids:
            results[label] = math.nan
            continue

        centroid = centroids[label]
        cohesion = float(np.linalg.norm(points - centroid, axis=1).mean()) if count > 1 else 0.0
        separation_candidates = [
            float(np.linalg.norm(centroid - centroids[other]))
            for other in unique_labels
            if other != label and other in centroids
        ]
        if not separation_candidates:
            results[label] = math.nan
            continue
        separation = min(separation_candidates)
        if cohesion == 0.0:
            results[label] = math.inf if separation > 0 else math.nan
        else:
            results[label] = separation / (cohesion + EPS)
    return results


def compute_davies_bouldin_component(coords: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    unique_labels = np.unique(labels)
    centroids: Dict[str, np.ndarray] = {}
    scatters: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for label in unique_labels:
        mask = labels == label
        points = coords[mask]
        count = int(np.count_nonzero(mask))
        counts[label] = count
        if count == 0:
            continue
        centroid = points.mean(axis=0)
        centroids[label] = centroid
        if count == 1:
            scatters[label] = 0.0
        else:
            scatters[label] = float(np.linalg.norm(points - centroid, axis=1).mean())

    results: Dict[str, float] = {}
    for label in unique_labels:
        if counts.get(label, 0) < 2:
            results[label] = math.nan
            continue
        centroid = centroids.get(label)
        scatter_i = scatters.get(label, math.nan)
        ratios: List[float] = []
        for other in unique_labels:
            if other == label or counts.get(other, 0) < 2:
                continue
            centroid_j = centroids.get(other)
            if centroid is None or centroid_j is None:
                continue
            distance = float(np.linalg.norm(centroid - centroid_j))
            if distance <= 0:
                continue
            scatter_j = scatters.get(other, math.nan)
            term_i = 0.0 if math.isnan(scatter_i) else scatter_i
            term_j = 0.0 if math.isnan(scatter_j) else scatter_j
            ratios.append((term_i + term_j) / distance)
        results[label] = max(ratios) if ratios else math.nan
    return results


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters in arbitrary strings."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = []
    for char in text:
        escaped.append(replacements.get(char, char))
    return "".join(escaped)


METRICS: List[MetricDefinition] = [
    MetricDefinition(
        name="silhouette",
        display_name="Silhouette",
        higher_is_better=True,
        precision=3,
        compute_fn=compute_silhouette_by_label,
    ),
    MetricDefinition(
        name="cohesion_separation",
        display_name="Separation / Cohesion",
        higher_is_better=True,
        precision=2,
        compute_fn=compute_cohesion_separation_ratio,
    ),
    MetricDefinition(
        name="davies_bouldin",
        display_name="Davies-Bouldin (max)",
        higher_is_better=False,
        precision=3,
        compute_fn=compute_davies_bouldin_component,
    ),
]


def compute_metrics(payload: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    required_columns = {"dataset_type", "tsne_x", "tsne_y"}
    missing = required_columns.difference(payload.columns)
    if missing:
        raise ValueError(f"Missing required columns in t-SNE CSV: {sorted(missing)}")
    coords = payload[["tsne_x", "tsne_y"]].to_numpy(dtype=np.float64)
    labels = payload["dataset_type"].astype(str).to_numpy()

    metrics: Dict[str, Dict[str, float]] = {}
    for metric in METRICS:
        metrics[metric.name] = metric.compute_fn(coords, labels)
    return metrics


def read_yaml(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if content is None:
        return {}
    if not isinstance(content, MutableMapping):
        raise ValueError(f"Expected YAML mapping at root of {path}")
    return content


def dig_mapping(payload: object, dotted_path: str) -> object:
    if not dotted_path:
        return payload
    current = payload
    for segment in dotted_path.split("."):
        segment = segment.strip()
        if not segment:
            continue
        if isinstance(current, Mapping):
            current = current.get(segment)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)) and segment.isdigit():
            index = int(segment)
            current = current[index]
        else:
            raise KeyError(f"Cannot navigate '{segment}' in path '{dotted_path}'")
        if current is None:
            break
    return current


def extract_model_names(
    *,
    config_path: Optional[Path],
    explicit_names: Optional[Sequence[str]],
    model_key: str,
    name_field: str,
) -> List[str]:
    if explicit_names:
        return [str(name) for name in explicit_names]
    if config_path is None:
        raise ValueError("Provide either --model-names or --model-config")
    config = read_yaml(config_path)
    container = dig_mapping(config, model_key)
    if container is None:
        raise KeyError(f"Section '{model_key}' not found in YAML config {config_path}")
    if not isinstance(container, Sequence):
        raise TypeError(f"Section '{model_key}' must be a list of model definitions")

    names: List[str] = []
    for entry in container:
        if isinstance(entry, Mapping):
            if name_field not in entry:
                raise KeyError(
                    f"Missing field '{name_field}' in model entry while parsing {config_path}"
                )
            names.append(str(entry[name_field]))
        else:
            names.append(str(entry))
    if not names:
        raise ValueError(f"No model names extracted from '{model_key}' in {config_path}")
    return names


def build_best_trackers(
    dataset_types: List[str],
    model_names: List[str],
    metric_values: Dict[str, Dict[str, Dict[str, float]]],
    metric_defs: Dict[str, MetricDefinition],
):
    best_lookup: Dict[str, Dict[str, Optional[str]]] = {ds: {} for ds in dataset_types}
    second_lookup: Dict[str, Dict[str, Optional[str]]] = {ds: {} for ds in dataset_types}

    for metric_name, metric_def in metric_defs.items():
        for dataset in dataset_types:
            values = []
            for model in model_names:
                value = (
                    metric_values.get(dataset, {})
                    .get(model, {})
                    .get(metric_name)
                )
                if value is None or math.isnan(value):
                    continue
                values.append((model, value))
            reverse = metric_def.higher_is_better
            values.sort(key=lambda item: item[1], reverse=reverse)
            best_model = values[0][0] if values else None
            second_model = values[1][0] if len(values) > 1 else None
            best_lookup[dataset][metric_name] = best_model
            second_lookup[dataset][metric_name] = second_model
    return best_lookup, second_lookup


def apply_emphasis(formatted: str, *, position: str) -> str:
    if position == "best":
        return f"\\textcolor{{red}}{{\\textbf{{{formatted}}}}}"
    if position == "second":
        return f"\\textcolor{{blue}}{{\\underline{{{formatted}}}}}"
    return formatted


def build_latex_table(
    *,
    dataset_types: List[str],
    model_names: List[str],
    metric_values: Dict[str, Dict[str, Dict[str, float]]],
    caption: str,
    label: Optional[str],
) -> str:
    metric_defs = {metric.name: metric for metric in METRICS}
    col_groups = len(model_names)
    metrics_per_model = len(METRICS)

    col_spec_parts = ["l"] + ["c"] * (col_groups * metrics_per_model)
    col_spec = " ".join(col_spec_parts)

    cmidrules = []
    start_col = 2
    for _model in model_names:
        end_col = start_col + metrics_per_model - 1
        cmidrules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1

    best_lookup, second_lookup = build_best_trackers(dataset_types, model_names, metric_values, metric_defs)

    lines: List[str] = []
    lines.append("\\begin{table*}[ht!]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\small")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header_cells = ["\\textbf{Dataset}"]
    for model in model_names:
        header_cells.append(
            f"\\multicolumn{{{metrics_per_model}}}{{c}}{{\\textbf{{{latex_escape(model)}}}}}"  # escape header names
        )
    lines.append(" & ".join(header_cells) + r" \\")

    lines.extend(cmidrules)
    metric_header = [" "]
    for _model in model_names:
        metric_header.extend(metric.display_name for metric in METRICS)
    lines.append(" & ".join(metric_header) + r" \\")
    lines.append("\\midrule")

    for dataset in dataset_types:
        row_cells = [f"\\textbf{{{latex_escape(dataset)}}}"]
        for model in model_names:
            for metric in METRICS:
                raw_value = (
                    metric_values.get(dataset, {})
                    .get(model, {})
                    .get(metric.name)
                )
                formatted = metric.format_value(raw_value)
                best = best_lookup[dataset].get(metric.name)
                second = second_lookup[dataset].get(metric.name)
                if best == model:
                    formatted = apply_emphasis(formatted, position="best")
                elif second == model:
                    formatted = apply_emphasis(formatted, position="second")
                row_cells.append(formatted)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table*}")
    if label:
        lines.insert(3, f"\\label{{{label}}}")
    return "\n".join(lines)


def main() -> None:
    config_path = determine_config_path()
    job_config = load_job_config(config_path)

    model_names = extract_model_names(
        config_path=job_config.model_config,
        explicit_names=job_config.model_names,
        model_key=job_config.model_key,
        name_field=job_config.model_name_field,
    )
    tsne_files = job_config.tsne_files
    if len(model_names) != len(tsne_files):
        raise ValueError(
            "Number of model names does not match number of provided CSV files "
            f"({len(model_names)} names vs {len(tsne_files)} files)."
        )

    dataset_types: List[str] = []
    metric_values: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model_name, csv_path in zip(model_names, tsne_files):
        frame = pd.read_csv(csv_path)
        metrics = compute_metrics(frame)
        present_types = sorted({str(label) for label in frame["dataset_type"].dropna().unique()})
        for dataset in present_types:
            if dataset not in dataset_types:
                dataset_types.append(dataset)
            metric_values.setdefault(dataset, {})
            metric_values[dataset].setdefault(model_name, {})
            for metric_name, per_label_values in metrics.items():
                metric_values[dataset][model_name][metric_name] = per_label_values.get(dataset, math.nan)

    dataset_types = sorted(dataset_types)
    if not dataset_types:
        raise ValueError("No dataset types detected in the provided t-SNE CSV files.")

    output_path = job_config.output_path
    if output_path is None:
        first_dir = tsne_files[0].parent
        output_path = first_dir / "tsne_clustering_metrics.tex"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table_body = build_latex_table(
        dataset_types=dataset_types,
        model_names=model_names,
        metric_values=metric_values,
        caption=job_config.table_caption,
        label=job_config.table_label,
    )
    output_path.write_text(table_body, encoding="utf-8")
    print(f"Loaded config: {config_path}")
    print(f"Saved LaTeX table to {output_path}")


if __name__ == "__main__":
    main()
