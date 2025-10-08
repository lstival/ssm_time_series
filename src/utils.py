
def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    required = {"experiment_name", "seed", "device", "model", "data", "training", "logging"}
    missing = required.difference(payload.keys())
    if missing:
        raise KeyError(f"Missing keys in configuration: {sorted(missing)}")
    return ExperimentConfig(
        experiment_name=str(payload["experiment_name"]),
        seed=int(payload.get("seed", 42)),
        device=str(payload.get("device", "auto")),
        model=dict(payload["model"]),
        data=dict(payload["data"]),
        training=dict(payload["training"]),
        logging=dict(payload["logging"]),
    )
