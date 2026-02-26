
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Literal, Iterable
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel


def time_series_2_recurrence_plot(x: torch.Tensor) -> torch.Tensor:
    """
    Native PyTorch implementation of Recurrence Plot.
    Computes distance matrix |x_i - x_j| via broadcasting.
    Input x: (B, L) or (B, C, L) or (L,)
    Returns: (B, L, L) or (B, C, L, L) or (L, L)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x).float()

    if x.ndim == 1:
        # (L,) -> (1, L, L)
        return torch.abs(x.unsqueeze(-1) - x.unsqueeze(-2))
    elif x.ndim == 2:
        # (B, L) -> (B, L, L)
        return torch.abs(x.unsqueeze(-1) - x.unsqueeze(-2))
    elif x.ndim == 3:
        # (B, C, L) -> (B, C, L, L)
        return torch.abs(x.unsqueeze(-1) - x.unsqueeze(-2))
    else:
        raise ValueError(f"Unsupported input shape for RP: {x.shape}")




@dataclass
class SDMambaDatasetConfig:
    """Configuration describing an SD-Mamba style dataset loader."""

    name: str
    data_key: str
    root_path: Path
    data_path: str
    features: str
    target: str
    freq: str
    seq_len: int
    label_len: int
    pred_len: int
    batch_size: int
    val_batch_size: int
    num_workers: int
    pin_memory: bool = True
    embed: str = "timeF"
    scale: bool = True
    scaler_type: str = "standard"
    train_shuffle: bool = True
    train_drop_last: bool = True
    val_shuffle: bool = False
    val_drop_last: bool = False


@dataclass
class ChronosForecastConfig:
    """Configuration class for Chronos forecasting."""
    # Model configuration
    model_config_path: Path
    overrides: Dict[str, object]
    
    # Training configuration
    horizons: List[int]
    max_horizon: int
    seed: int
    batch_size: int
    val_batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    mlp_hidden_dim: int
    num_workers: int
    pin_memory: bool
    grad_clip: float
    
    # Chronos dataset configuration
    datasets_to_load: List[str]
    split: str
    repo_id: str
    target_dtype: Optional[str]
    normalize_per_series: bool
    val_ratio: float
    context_length: int
    stride: int
    max_windows_per_series: Optional[int]
    max_series: Optional[int]
    load_kwargs: Dict[str, object]
    
    # Path configuration
    visual_mamba_checkpoint_path: Path
    encoder_checkpoint_path: Path
    checkpoint_dir: Path
    results_dir: Path
    
    # Configuration metadata
    config_path: Path
    config_dir: Path


def _parse_dataset_names(raw: Optional[Iterable[str] | str]) -> List[str]:
    """Parse dataset names from various input formats."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [entry.strip() for entry in raw.split(",") if entry.strip()]
    result: List[str] = []
    for entry in raw:
        if entry is None:
            continue
        name = str(entry).strip()
        if name:
            result.append(name)
    return result


def _load_yaml_config(path: Path) -> Dict[str, object]:
    """Load YAML configuration from file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file must define a mapping: {path}")
    return payload


def _resolve_optional_path(base: Path, candidate: Optional[object]) -> Optional[Path]:
    """Resolve optional path relative to base directory."""
    if candidate is None:
        return None
    candidate_path = Path(str(candidate))
    # Try to resolve relative to base first
    if not candidate_path.is_absolute():
        resolved = (base / candidate_path).resolve()
        if resolved.exists():
            return resolved
    return candidate_path.expanduser().resolve()


def _coerce_path(base: Path, candidate: object, *, must_exist: bool = False, description: str) -> Path:
    """Coerce and validate path."""
    resolved = _resolve_optional_path(base, candidate)
    if resolved is None:
        raise FileNotFoundError(f"{description} not found: {candidate}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _normalize_horizons(raw: object) -> List[int]:
    """Normalize horizon values from config."""
    if raw is None:
        raise ValueError("Configuration must supply 'training.horizons'.")
    if isinstance(raw, str):
        # Assume comma-separated string
        try:
            values = [int(v.strip()) for v in raw.split(",") if v.strip()]
        except ValueError as exc:
            raise ValueError("'training.horizons' string must be comma-separated integers.") from exc
    else:
        try:
            values = [int(value) for value in raw]  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("'training.horizons' must be a list of integers or a comma-separated string.") from exc
    return sorted(set(values))


def _ensure_hf_list_feature_registered() -> None:
    """Ensure HuggingFace List feature is registered."""
    feature_registry = getattr(datasets.features, "_FEATURE_TYPES", None)
    sequence_cls = getattr(datasets.features, "Sequence", None)
    if isinstance(feature_registry, dict) and "List" not in feature_registry and sequence_cls is not None:
        feature_registry["List"] = sequence_cls


def load_chronos_forecast_config(
    config_path: Optional[Path] = None,
    env_var: str = "CHRONOS_FORECAST_CONFIG",
    default_config_name: str = "chronos_forecast.yaml"
) -> ChronosForecastConfig:
    """Load and parse Chronos forecast configuration.
    
    Args:
        config_path: Path to configuration file (optional)
        env_var: Environment variable name for config override
        default_config_name: Default config filename to look for
        
    Returns:
        ChronosForecastConfig: Parsed configuration object
    """
    # Determine config path
    if config_path is None:
        env_override = os.getenv(env_var)
        if env_override:
            config_path = Path(env_override)
        else:
            # Look for default config in src/configs/
            src_dir = Path(__file__).resolve().parents[1]  # Go up from models to src
            config_path = src_dir / "configs" / default_config_name
    
    config_path = _resolve_optional_path(Path.cwd(), config_path)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Chronos forecast configuration not found: {config_path}")
    
    print(f"Using forecast configuration: {config_path}")
    
    forecast_cfg = _load_yaml_config(config_path)
    config_dir = config_path.parent
    
    # Load model configuration
    model_section = dict(forecast_cfg.get("model") or {})
    model_config_candidate = model_section.get("config")
    if model_config_candidate is None:
        raise ValueError("Configuration missing required key 'model.config'.")
    model_config_path = _coerce_path(
        config_dir,
        model_config_candidate,
        must_exist=True,
        description="Model configuration",
    )
    
    # Import training_utils to load base config
    src_dir = Path(__file__).resolve().parents[1]
    root_dir = src_dir.parent
    for path in (src_dir, root_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    
    base_config = tu.load_config(model_config_path)
    
    data_cfg = dict(base_config.data or {})
    logging_cfg = dict(base_config.logging or {})
    
    overrides = dict(model_section.get("overrides") or {})
    
    # Parse training configuration
    training_section = dict(forecast_cfg.get("training") or {})
    horizons = _normalize_horizons(training_section.get("horizons"))
    max_horizon = max(horizons)
    
    seed = int(training_section.get("seed", base_config.seed))
    batch_size = int(training_section.get("batch_size", data_cfg.get("batch_size", 16)))
    val_batch_size = int(training_section.get("val_batch_size", batch_size))
    epochs = int(training_section.get("epochs", 2))
    lr = float(training_section.get("lr", 3e-4))
    weight_decay = float(training_section.get("weight_decay", 1e-2))
    mlp_hidden_dim = int(training_section.get("mlp_hidden_dim", 512))
    num_workers = int(training_section.get("num_workers", data_cfg.get("num_workers", 0)))
    pin_memory = bool(training_section.get("pin_memory", data_cfg.get("pin_memory", True)))
    grad_clip = float(training_section.get("grad_clip", 1.0))
    
    # Parse Chronos configuration
    chronos_section = dict(forecast_cfg.get("chronos") or {})
    loader_config_candidate = chronos_section.get("config")
    chronos_cfg: Dict[str, object] = {}
    if loader_config_candidate is not None:
        loader_config_path = _coerce_path(
            config_dir,
            loader_config_candidate,
            must_exist=True,
            description="Chronos loader configuration",
        )
        with loader_config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Chronos loader configuration must be a mapping: {loader_config_path}")
        chronos_cfg.update(payload)
    
    # Parse dataset configuration
    datasets_value = chronos_section.get("datasets")
    if datasets_value is None:
        datasets_value = chronos_cfg.get("datasets_to_load") or chronos_cfg.get("datasets")
    datasets_to_load = _parse_dataset_names(datasets_value)
    if not datasets_to_load:
        raise ValueError("Chronos configuration must provide at least one dataset name.")
    
    split = str(chronos_section.get("split", chronos_cfg.get("split", "train")))
    repo_id = str(chronos_section.get("repo_id", chronos_cfg.get("repo_id", "autogluon/chronos_datasets")))
    target_dtype = chronos_section.get("target_dtype", chronos_cfg.get("target_dtype"))
    
    normalize_per_series = bool(chronos_section.get("normalize", chronos_cfg.get("normalize", True)))
    val_ratio = float(chronos_section.get("val_split", chronos_cfg.get("val_split", 0.2)))
    
    context_length = chronos_section.get("context_length")
    if context_length is None:
        context_length = chronos_cfg.get("context_length", chronos_cfg.get("patch_length", 384))
    context_length = int(context_length)
    
    stride_value = chronos_section.get("window_stride", chronos_cfg.get("window_stride"))
    stride = int(stride_value) if stride_value is not None else max_horizon
    stride = max(1, stride)
    
    max_windows_per_series = chronos_section.get("max_windows_per_series", chronos_cfg.get("max_windows_per_series"))
    if max_windows_per_series is not None:
        max_windows_per_series = int(max_windows_per_series)
    
    max_series = chronos_section.get("max_series", chronos_cfg.get("max_series"))
    if max_series is not None:
        max_series = int(max_series)
    
    load_kwargs: Dict[str, object] = dict(chronos_cfg.get("load_kwargs", {}) or {})
    section_load_kwargs = chronos_section.get("load_kwargs")
    if isinstance(section_load_kwargs, dict):
        load_kwargs.update(section_load_kwargs)
    
    force_offline = chronos_section.get("force_offline")
    if force_offline is not None:
        load_kwargs["force_offline"] = bool(force_offline)
    
    offline_cache_dir = load_kwargs.get("offline_cache_dir")
    if offline_cache_dir is not None:
        load_kwargs["offline_cache_dir"] = str(
            _resolve_optional_path(config_dir, offline_cache_dir)
        )
    if "offline_cache_dir" not in load_kwargs:
        load_kwargs["offline_cache_dir"] = str((root_dir / "data").resolve())
    
    # Parse paths configuration
    paths_section = dict(forecast_cfg.get("paths") or {})
    visual_candidate = paths_section.get(
        "visual_encoder_checkpoint",
        Path("../../checkpoints/ts_encoder_20251101_1100/visual_encoder_best.pt"),
    )
    visual_mamba_checkpoint_path = _coerce_path(
        config_dir,
        visual_candidate,
        must_exist=True,
        description="Visual Mamba encoder checkpoint",
    )
    
    temporal_encoder_checkpoint = _coerce_path(
        config_dir,
        visual_candidate,
        must_exist=True,
        description="Temporal encoder checkpoint",
    )

    encoder_candidate = paths_section.get(
        "encoder_checkpoint",
        Path("../../checkpoints/ts_encoder_20251101_1100/time_series_best.pt"),
    )
    encoder_checkpoint_path = _coerce_path(
        config_dir,
        encoder_candidate,
        must_exist=True,
        description="Encoder checkpoint",
    )
    
    checkpoint_candidate = paths_section.get("checkpoint_dir")
    if checkpoint_candidate is not None:
        checkpoint_base = config_dir
    else:
        checkpoint_candidate = logging_cfg.get("checkpoint_dir", root_dir / "checkpoints")
        checkpoint_base = root_dir
    checkpoint_dir = _coerce_path(
        checkpoint_base,
        checkpoint_candidate,
        must_exist=False,
        description="Checkpoint directory",
    )
    
    results_candidate = paths_section.get("results_dir")
    if results_candidate is not None:
        results_base = config_dir
    else:
        results_candidate = root_dir / "results"
        results_base = root_dir
    results_dir = _coerce_path(
        results_base,
        results_candidate,
        must_exist=False,
        description="Results directory",
    )
    
    # Ensure HF feature is registered
    _ensure_hf_list_feature_registered()
    
    return ChronosForecastConfig(
        model_config_path=model_config_path,
        overrides=overrides,
        horizons=horizons,
        max_horizon=max_horizon,
        seed=seed,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        mlp_hidden_dim=mlp_hidden_dim,
        num_workers=num_workers,
        pin_memory=pin_memory,
        datasets_to_load=datasets_to_load,
        split=split,
        repo_id=repo_id,
        target_dtype=target_dtype,
        normalize_per_series=normalize_per_series,
        val_ratio=val_ratio,
        context_length=context_length,
        stride=stride,
        max_windows_per_series=max_windows_per_series,
        max_series=max_series,
        load_kwargs=load_kwargs,
        visual_mamba_checkpoint_path=visual_mamba_checkpoint_path,
        encoder_checkpoint_path=encoder_checkpoint_path,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        config_path=config_path,
        config_dir=config_dir,
        grad_clip=grad_clip,
    )







def _canonical_kernel_size(kernel_size: int) -> int:
    if kernel_size < 1:
        raise ValueError("conv_kernel must be >= 1")
    return kernel_size


def hippo_legS_matrix(
    state_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return HiPPO-LegS initialization matrices (A, B, C)."""
    if state_dim <= 0:
        raise ValueError("state_dim must be positive for HiPPO initialisation")
    device = torch.device("cpu") if device is None else device
    A = torch.zeros((state_dim, state_dim), dtype=dtype, device=device)
    for n in range(state_dim):
        coeff_n = math.sqrt(2 * n + 1)
        for m in range(n + 1):
            coeff_m = math.sqrt(2 * m + 1)
            if n == m:
                A[n, m] = -(2 * n + 1)
            else:
                sign = -1.0 if (n - m) % 2 == 0 else 1.0
                A[n, m] = sign * coeff_n * coeff_m
    indices = torch.arange(state_dim, dtype=dtype, device=device)
    B = torch.sqrt(2.0 * indices + 1.0)
    C = B.clone()
    C[1::2] *= -1.0
    return A, B, C


class MambaBlock(nn.Module):
    """A minimal Mamba-style block with a selective scan core.

    Parameters
    ----------
    d_model: int
        Number of channels in the model (after the input projection).
    state_dim: int, default=16
        Internal state dimension for the selective scan. Smaller values keep the
        parameter count down.
    conv_kernel: int, default=3
        Depthwise convolution kernel size used as a light-weight local mixer.
    expand_factor: float, default=1.5
        Expansion ratio that controls the inner dimensionality of the block.
    dropout: float, default=0.0
        Dropout probability applied to the residual branch output.
    """

    def __init__(
        self,
        d_model: int = 128,
        *,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand_factor: float = 1.5,
        dropout: float = 0.0,
        input_dim: int = 32,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if expand_factor <= 1.0:
            raise ValueError("expand_factor must be greater than 1.0")

        self.hidden_model = 128
        self.input_dim = input_dim
        self.d_model = d_model
        self.state_dim = state_dim
        self.inner_dim = math.ceil(d_model * expand_factor)
        kernel_size = _canonical_kernel_size(conv_kernel)

        self.norm = nn.LayerNorm(d_model)
        # project from d_model to 3 * inner_dim
        # so that projected.chunk(3, dim=-1) yields tensors with last dim == inner_dim
        self.in_proj = nn.Linear(d_model, self.inner_dim * 3, bias=False)
        self.depthwise_conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=self.inner_dim,
            bias=False,
        )
        self.conv_activation = nn.SiLU()

        # HiPPO-LegS initialised SSM parameters (A, B, C).
        A_init, B_init, C_init = hippo_legS_matrix(state_dim)
        scale = 1.0 / math.sqrt(self.inner_dim)
        self.A = nn.Parameter(A_init)
        self.B = nn.Parameter(B_init.unsqueeze(1).repeat(1, self.inner_dim) * scale)
        self.C = nn.Parameter(C_init.unsqueeze(0).repeat(self.inner_dim, 1) * scale)

        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.delta_min = 1e-4
        self.delta_max = 3.0
        self.register_buffer("eye_state", torch.eye(state_dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to a `(batch, seq, channels)` tensor."""
        if x.ndim != 3:
            raise ValueError("Expected input of shape (batch, seq, channels)")

        residual = x
        x = self.norm(x)
        projected = self.in_proj(x)
        data, gate, delta_raw = projected.chunk(3, dim=-1)

        # Depth-wise conv expects (batch, channels, seq)
        data_conv = data.transpose(1, 2)
        data_conv = self.depthwise_conv(data_conv)
        seq_len = x.size(1)
        if data_conv.size(-1) > seq_len:
            data_conv = data_conv[..., :seq_len]
        data_conv = data_conv.transpose(1, 2)
        data_conv = self.conv_activation(data_conv)

        delta = F.softplus(delta_raw).mean(dim=-1, keepdim=True) + self.delta_min
        delta = delta.clamp(max=self.delta_max)

        selective = self._selective_scan(data_conv, delta)
        gated = torch.sigmoid(gate) * selective
        out = self.out_proj(gated)
        out = self.dropout(out)
        return residual + out

    def _selective_scan(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Selective scan driven by HiPPO-initialised SSM (A, B, C).
        Uses a zero-order hold discretisation per time step based on a
        data-dependent positive step size ``delta``.

        Optimized version: Vectorizes discretization and simplifies the recurrent loop.
        """
        batch, seq_len, _ = x.shape
        if delta.shape[0] != batch or delta.shape[1] != seq_len or delta.shape[2] != 1:
            raise ValueError(f"delta must have shape (batch, seq_len, 1), got {delta.shape}")

        # 1. Vectorized Discretization (O(1) matrix operations outside the loop)
        # delta: (batch, seq_len, 1) -> (batch, seq_len, 1, 1)
        # self.A: (state_dim, state_dim)
        # scaled_A: (batch, seq_len, state_dim, state_dim)
        scaled_A = delta.unsqueeze(-1) * self.A
        A_disc = torch.matrix_exp(scaled_A)

        # B_disc = A^-1 (exp(delta A) - I) B
        I_minus_A_expm = A_disc - self.eye_state

        # Solve A * integral = (exp(delta A) - I) for the whole batch/sequence at once
        integral = torch.linalg.solve(self.A, I_minus_A_expm)
        # B_disc: (batch, seq_len, state_dim, inner_dim)
        B_disc = torch.matmul(integral, self.B)

        # 2. Recurrent loop (now significantly faster as it only involves matrix multiplications)
        state = x.new_zeros(batch, self.state_dim, 1)
        outputs = x.new_empty(batch, seq_len, self.inner_dim)

        for t in range(seq_len):
            u_t = x[:, t, :].unsqueeze(-1)  # (batch, inner_dim, 1)

            # state_t = A_disc_t @ state_{t-1} + B_disc_t @ u_t
            state = torch.bmm(A_disc[:, t], state) + torch.bmm(B_disc[:, t], u_t)

            # output_t = C @ state_t
            # (inner, state) @ (batch, state, 1) -> (batch, inner, 1)
            outputs[:, t, :] = torch.matmul(self.C, state).squeeze(-1)

        return outputs
    
if __name__ == '__main__':
    # Example of use (model dim = 128)
    b,f,t = 4,1,384
    x = torch.randn(b,f,t)
    projection = nn.Linear(t, 128)(x)
    mamba_block = MambaBlock()
    out = mamba_block(projection)
    print(out.shape)

"""Lightweight Mamba-style encoder for 384-dim time series inputs.

This module implements a compact encoder inspired by the Mamba state-space
architecture (https://arxiv.org/abs/2312.00752). It is designed to keep the
parameter count low while still capturing temporal structure in fixed-length
(time-major) sequences. The core building block is a simplified selective scan
that mixes information along the sequence using inexpensive operations.

Example
-------
>>> import torch
>>> from mamba_encoder import MambaEncoder, temporal_tokenize_sequence
>>> encoder = MambaEncoder()
>>> x = torch.randn(8, 128, 384)  # (batch, time, features)
>>> # create non-overlapping tokens of length 16 (result: 8 tokens)
>>> xt = temporal_tokenize_sequence(x, token_size=16, stride=None, method="mean")
>>> embedding = encoder(xt)        # encoder expects (B, seq, features)
>>> embedding.shape
torch.Size([8, 8, 128])  # if pooling='mean' / projection dims, etc.
"""


Pooling = Literal["mean", "last", "cls"]


class TemporalTokenizer:
    """
    Convert a (B, T, F) sequence into a sequence of tokens (B, N, F) by
    windowing along the time dimension.

    Usage:
        tk = TemporalTokenizer(token_size=32, stride=None, method="values", pad=True)
        tokens = tk(x)  # where x is (B, T, F) or (B, F, T) — note: this implementation
                        # will swap axes at the start like the original function.

    Parameters
    - token_size: length of each token window (number of timesteps per token)
    - stride: step between token starts. If None, defaults to token_size (non-overlap).
    - method: how to aggregate values inside a token: "mean", "max", "first", or
              "values". "values" returns the raw window values with shape
              (batch, n_tokens, token_size, features).
    - pad: whether to pad the end with zeros so the final token is full length.
    """

    def __init__(
        self,
        *,
        token_size: int = 32,
        stride: Optional[int] = None,
        method: Literal["mean", "max", "first", "values"] = "values",
        pad: bool = True,
    ) -> None:
        if token_size <= 0:
            raise ValueError("token_size must be positive")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive")
        if method not in ("mean", "max", "first", "values"):
            raise ValueError(f"Unknown tokenization method: {method}")

        self.token_size = token_size
        self.stride = stride if stride is not None else token_size
        self.method = method
        self.pad = pad

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Swap axes from (B, F, T) to (B, T, F) as default of tokenizes (preserve original behavior)
        x = x.swapaxes(1, 2)
        if x.ndim != 3:
            raise ValueError("temporal_tokenize_sequence expects input of shape (batch, time, features)")

        B, T, F = x.shape
        token_size = self.token_size
        stride = self.stride

        # compute required padding so that unfold will include the final (possibly partial) block
        if T <= token_size:
            n_steps = 1
            required_total = token_size
        else:
            n_steps = math.ceil((T - token_size) / stride) + 1
            required_total = token_size + stride * (n_steps - 1)
        pad_len = max(0, required_total - T)

        if pad_len > 0:
            if self.pad:
                pad_tensor = x.new_zeros((B, pad_len, F))
                x_padded = torch.cat([x, pad_tensor], dim=1)
            else:
                raise ValueError(
                    "Sequence length is shorter than token_size and pad=False; cannot form full tokens"
                )
        else:
            x_padded = x

        # Create the patches for the number of tokens: (B, n_tokens, token_size, F)
        patches = x_padded.unfold(dimension=1, size=token_size, step=stride).contiguous()
        patches = patches.permute(0, 1, 3, 2)

        if self.method == "mean":
            tokens = patches.mean(dim=2)
        elif self.method == "max":
            tokens, _ = patches.max(dim=2)
        elif self.method == "first":
            tokens = patches[:, :, 0, :]
        elif self.method == "values":
            tokens = patches
        else:
            # should be unreachable due to check in __init__
            raise ValueError(f"Unknown tokenization method: {self.method}")

        return tokens


def temporal_tokenize_sequence(
    x: torch.Tensor,
    *,
    token_size: int = 32,
    stride: Optional[int] = None,
    method: Literal["mean", "max", "first", "values"] = "values",
    pad: bool = True,
) -> torch.Tensor:
    return TemporalTokenizer(token_size=token_size, stride=stride, method=method, pad=pad)(x)


class MambaEncoder(nn.Module):
    """Compact Mamba-style encoder for 384-dim time series inputs."""

    def __init__(
        self,
        *,
        input_dim: int = 32, token_size: int = 32, #Token size default as 16
        model_dim: int = 768,
        depth: int = 6,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand_factor: float = 1.5,
        embedding_dim: int = 128,
        pooling: Pooling = "mean",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.input_dim = input_dim
        self.token_size = token_size
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.pooling: Pooling = pooling

        self.input_proj = nn.Linear(input_dim, model_dim, bias=False)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    model_dim,
                    state_dim=state_dim,
                    conv_kernel=conv_kernel,
                    expand_factor=expand_factor,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a fixed-size embedding for a `(batch, seq, features)` tensor."""
        if x.ndim != 3:
            raise ValueError("Expected input of shape (batch, seq, features)")
        seq_len = x.size(2)
        if seq_len < self.input_dim:
            raise ValueError(
                f"Sequence length ({seq_len}) must be at least as large as the encoder token size ({self.input_dim})."
            )
        # if x.size(-1) != self.input_dim:
        #     raise ValueError(f"Expected final dimension {self.input_dim}, got {x.size(-1)}")

        features = self.forward_sequence(x)
        pooled = self._pool_sequence(features, x)
        return self.output_proj(pooled)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sequence of hidden states before pooling."""
        tokens = self.tokenizer(x)
        if tokens.ndim == 4:
            batch, windows, window_len, feat_dim = tokens.shape
            tokens = tokens.reshape(batch, windows * window_len, feat_dim)
        x = self.input_proj(tokens) # Change here to use pre_trained model to feature exctration
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
    
    def tokenizer(self, x):
        tokens = temporal_tokenize_sequence(x, token_size=self.token_size)
        return tokens

    def _pool_sequence(self, hidden: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        if self.pooling == "last":
            return hidden[:, -1, :]
        if self.pooling == "cls":
            # Expect the first timestep to act as a CLS token; fall back to mean if seq len == 1
            if hidden.size(1) == 1:
                return hidden[:, 0, :]
            return hidden[:, 0, :]
        raise ValueError(f"Unknown pooling mode: {self.pooling}")

    def count_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)


def create_default_encoder(**overrides: object) -> MambaEncoder:
    """Factory that mirrors the defaults while allowing overrides."""
    return MambaEncoder(**overrides)



"""Lightweight Mamba-style encoder for 384-dim time series inputs.

This module implements a compact encoder inspired by the Mamba state-space
architecture (https://arxiv.org/abs/2312.00752). It is designed to keep the
parameter count low while still capturing temporal structure in fixed-length
(time-major) sequences. The core building block is a simplified selective scan
that mixes information along the sequence using inexpensive operations.

"""



Pooling = Literal["mean", "last", "cls"]


class VisualTokenizer:
    """
    Convert a (B, T, F) sequence into a sequence of tokens (B, N, F) by
    windowing along the time dimension.

    Usage:
        tk = VisualTokenizer(token_size=32, stride=None, method="values", pad=True)
        tokens = tk(x)  # where x is (B, T, F) or (B, F, T) — note: this implementation
                        # will swap axes at the start like the original function.

    Parameters
    - token_size: length of each token window (number of timesteps per token)
    - stride: step between token starts. If None, defaults to token_size (non-overlap).
    - method: how to aggregate values inside a token: "mean", "max", "first", or
              "values". "values" returns the raw window values with shape
              (batch, n_tokens, token_size, features).
    - pad: whether to pad the end with zeros so the final token is full length.
    """

    def __init__(
        self,
        *,
        token_size: int = 32,
        stride: Optional[int] = None,
        method: Literal["mean", "max", "first", "values"] = "values",
        pad: bool = True,
    ) -> None:
        if token_size <= 0:
            raise ValueError("token_size must be positive")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive")
        if method not in ("mean", "max", "first", "values"):
            raise ValueError(f"Unknown tokenization method: {method}")

        self.token_size = token_size
        self.stride = stride if stride is not None else token_size
        self.method = method
        self.pad = pad

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Swap axes from (B, F, T) to (B, T, F) as default of tokenizes (preserve original behavior)
        # x = x.swapaxes(1, 2)
        if x.ndim != 3:
            raise ValueError("visual_tokenize_sequence expects input of shape (batch, time, features)")

        B, T, F = x.shape
        token_size = self.token_size
        stride = self.stride

        # compute required padding so that unfold will include the final (possibly partial) block
        if T <= token_size:
            n_steps = 1
            required_total = token_size
        else:
            n_steps = math.ceil((T - token_size) / stride) + 1
            required_total = token_size + stride * (n_steps - 1)
        pad_len = max(0, required_total - T)

        if pad_len > 0:
            if self.pad:
                pad_tensor = x.new_zeros((B, pad_len, F))
                x_padded = torch.cat([x, pad_tensor], dim=1)
            else:
                raise ValueError(
                    "Sequence length is shorter than token_size and pad=False; cannot form full tokens"
                )
        else:
            x_padded = x

        # Create the patches for the number of tokens: (B, n_tokens, token_size, F)
        patches = x_padded.unfold(dimension=1, size=token_size, step=stride).contiguous()
        patches = patches.permute(0, 1, 3, 2)

        if self.method == "values":
            tokens = patches
        elif self.method == "mean":
            tokens = patches.mean(dim=2)
        elif self.method == "max":
            tokens, _ = patches.max(dim=2)
        elif self.method == "first":
            tokens = patches[:, :, 0, :]
        else:
            # should be unreachable due to check in __init__
            raise ValueError(f"Unknown tokenization method: {self.method}")

        return tokens

def visual_tokenize_sequence(
    x: torch.Tensor,
    *,
    token_size: int = 32,
    stride: Optional[int] = None,
    method: Literal["mean", "max", "first", "values"] = "values",
    pad: bool = True,
) -> torch.Tensor:
    return VisualTokenizer(token_size=token_size, stride=stride, method=method, pad=pad)(x)

class _InputConv(nn.Module):
    """
    Accept input of shape (B, windows, window_size, window_size) and produce
    (B, windows, out_dim) by applying a Conv2d over each window and collapsing
    the spatial dimensions to a single vector per window.

    token_len here is the spatial window_size (H = W = token_len).
    """
    def __init__(self, token_len: int, out_dim: int):
        super().__init__()
        if token_len <= 0:
            raise ValueError("token_len must be positive")
        self.token_len = token_len
        self.out_dim = out_dim
        # single-channel input windows -> produce out_dim channels, kernel covers whole window
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_dim, kernel_size=(token_len, token_len), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # allow array-like inputs
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        residual = x
        # expect (B, windows, H, W)
        if x.ndim != 4:
            raise ValueError("Expected input of shape (B, windows, window_size, window_size)")

        b, windows, h, w = x.shape
        if h != self.token_len or w != self.token_len:
            raise ValueError(f"Window size mismatch: got {h}x{w}, expected {self.token_len}x{self.token_len}")

        # treat each window as a single-channel image: (B*windows, 1, H, W)
        x = x.view(b * windows, 1, h, w)
        x = self.conv(x)  # -> (B*windows, out_dim, 1, 1)
        x = x.view(b, windows, -1)  # -> (B, windows, out_dim)
        return x

class MambaVisualEncoder(nn.Module):
    """Compact Mamba-style encoder for 384-dim time series inputs."""

    def __init__(
        self,
        *,
        input_dim: int = 32, token_size: int = 32, #Token size default as 16
        model_dim: int = 768,
        depth: int = 6,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand_factor: float = 1.5,
        embedding_dim: int = 128,
        pooling: Pooling = "mean",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.input_dim = input_dim
        self.token_size = token_size
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.pooling: Pooling = pooling

        self.input_proj = _InputConv(token_len=self.input_dim, out_dim=model_dim)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    model_dim,
                    state_dim=state_dim,
                    conv_kernel=conv_kernel,
                    expand_factor=expand_factor,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, embedding_dim, bias=False)

    def _time_series_2_image(self, ts):
        # Tokens are (B, W, L_token, F)
        # We want (B, W, L_token, L_token)
        # Combine B and W to call the RP utility
        B, W, L, F = ts.shape
        ts_reshaped = ts.view(B * W, L, F)
        rp = time_series_2_recurrence_plot(ts_reshaped)  # (B*W, L, L)
        # Reshape back to (B, W, L, L)
        return rp.view(B, W, L, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_sequence(x)
        pooled = self._pool_sequence(features, x)
        return self.output_proj(pooled)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sequence of hidden states before pooling."""
        tokens = self.tokenizer(x)
        
        # Consistent with (B, windows, H, W) for input_proj.
        # Tokens from tokenizer(method="values") is (B, W, L_token, F).
        img_from_patches = self._time_series_2_image(tokens)
        
        # Ensure it's on the correct device
        if not isinstance(img_from_patches, torch.Tensor):
            img_from_patches = torch.from_numpy(img_from_patches).float()
        img_from_patches = img_from_patches.to(x.device)

        # If windows was 1, it might be squeezed.
        if img_from_patches.ndim == 3:
            img_from_patches = img_from_patches.unsqueeze(1)

        x = self.input_proj(img_from_patches) 

        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def tokenizer(self, x):
        # We use method="values" to get (B, W, L_token, F) for RP
        tokens = visual_tokenize_sequence(x, token_size=self.input_dim, method="values")
        return tokens

    def _pool_sequence(self, hidden: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        if self.pooling == "last":
            return hidden[:, -1, :]
        if self.pooling == "cls":
            # Expect the first timestep to act as a CLS token; fall back to mean if seq len == 1
            if hidden.size(1) == 1:
                return hidden[:, 0, :]
            return hidden[:, 0, :]
        raise ValueError(f"Unknown pooling mode: {self.pooling}")

def create_default_encoder(**overrides: object) -> MambaVisualEncoder:
    """Factory that mirrors the defaults while allowing overrides."""
    return MambaVisualEncoder(**overrides)




class MultiHorizonForecastMLP(nn.Module):
    """MLP that predicts up to the maximum configured horizon in one pass."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizons: List[int],
        target_features: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not horizons:
            raise ValueError("At least one horizon must be provided")
        self.horizons = sorted(set(int(h) for h in horizons))
        self.target_features = int(target_features)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_horizon = max(self.horizons)

        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.prediction_head = nn.Linear(
            self.hidden_dim,
            self.max_horizon * self.target_features,
        )

    def forward(self, x: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        shared_out = self.shared_layers(x)
        output = self.prediction_head(shared_out)
        output = output.view(batch_size, self.max_horizon, self.target_features)
        if horizon is None:
            return output
        if horizon not in self.horizons:
            raise ValueError(
                f"Requested horizon {horizon} not configured; available {self.horizons}"
            )
        return output[:, :horizon, :]

    def get_max_horizon(self) -> int:
        return self.max_horizon


class ForecastRegressor(nn.Module):
	"""Frozen encoder followed by a trainable forecasting head."""

	def __init__(
		self,
		*,
		encoder: nn.Module,
		head: MultiHorizonForecastMLP,
		freeze_encoder: bool = True,
	) -> None:
		super().__init__()
		self.encoder = encoder
		self.head = head
		self.freeze_encoder = freeze_encoder

		if freeze_encoder:
			for param in self.encoder.parameters():
				param.requires_grad = False
			self.encoder.eval()

	def forward(self, x: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
		with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
			embeddings = self.encoder(x)
		return self.head(embeddings, horizon)

	def train(self, mode: bool = True):
		super().train(mode)
		if self.freeze_encoder:
			self.encoder.eval()
		return self

	@property
	def max_horizon(self) -> int:
		return self.head.get_max_horizon()

"""Dual encoder forecasting models."""




class DualEncoderForecastMLP(nn.Module):
    """MLP that takes concatenated outputs from both encoder and visual_encoder."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizons: List[int],
        target_features: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not horizons:
            raise ValueError("At least one horizon must be provided")
        self.horizons = sorted(set(int(h) for h in horizons))
        self.target_features = int(target_features)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_horizon = max(self.horizons)

        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.prediction_head = nn.Linear(
            self.hidden_dim,
            self.max_horizon * self.target_features,
        )

    def forward(self, x: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        shared_out = self.shared_layers(x)
        output = self.prediction_head(shared_out)
        output = output.view(batch_size, self.max_horizon, self.target_features)
        if horizon is None:
            return output
        if horizon not in self.horizons:
            raise ValueError(f"Requested horizon {horizon} not configured; available {self.horizons}")
        return output[:, :horizon, :]

    def get_max_horizon(self) -> int:
        return self.max_horizon


class DualEncoderForecastRegressor(nn.Module):
    """Forecast regressor using both encoder and visual_encoder with frozen weights."""
    
    def __init__(
        self,
        encoder: nn.Module,
        visual_encoder: nn.Module,
        head: DualEncoderForecastMLP,
        freeze_encoders: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.visual_encoder = visual_encoder  
        self.head = head
        
        if freeze_encoders:
            # Freeze encoders
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
                
        self.encoder.eval()
        self.visual_encoder.eval()

    def forward(self, x: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        # Transpose input for encoders (from [batch, seq, features] to [batch, features, seq])
        x_transposed = x.transpose(1, 2)
        
        # Encode with both encoders
        encoder_embedding = self.encoder(x_transposed)
        visual_embedding = self.visual_encoder(x_transposed)
        
        # Concatenate embeddings
        combined_embedding = torch.cat([encoder_embedding, visual_embedding], dim=-1)
        
        # Pass through forecast head
        return self.head(combined_embedding, horizon=horizon)

"""Hugging Face forecasting interface for CM-Mamba models."""




class CM_MambaForecastConfig(PretrainedConfig):
    """Configuration for CM-Mamba forecasting models.

    Attributes:
        input_dim: Input feature dimension (F).
        token_size: Token window size.
        model_dim: Hidden model dimension.
        embedding_dim: Encoder embedding dimension.
        depth: Number of Mamba blocks.
        state_dim: State dimension for Mamba blocks.
        conv_kernel: Convolution kernel size.
        expand_factor: Mamba expansion factor.
        dropout: Encoder dropout.
        pooling: Pooling strategy (mean/last/cls).
        encoder_type: "temporal" or "visual" for single-encoder models.
        use_dual_encoder: Whether to use both temporal and visual encoders.
        horizons: Forecast horizons (sorted unique list).
        target_features: Number of target features.
        mlp_hidden_dim: Hidden dimension of the forecasting MLP head.
        head_dropout: Dropout for the forecasting head.
        freeze_encoder: Whether to freeze encoder weights.
    """

    model_type = "cm_mamba_forecast"

    def __init__(
        self,
        *,
        input_dim: int = 32,
        token_size: int = 32,
        model_dim: int = 128,
        embedding_dim: int = 128,
        depth: int = 8,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand_factor: float = 1.5,
        dropout: float = 0.05,
        pooling: str = "mean",
        encoder_type: str = "temporal",
        use_dual_encoder: bool = False,
        horizons: Optional[Sequence[int]] = None,
        target_features: int = 1,
        mlp_hidden_dim: int = 512,
        head_dropout: float = 0.1,
        freeze_encoder: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.token_size = int(token_size)
        self.model_dim = int(model_dim)
        self.embedding_dim = int(embedding_dim)
        self.depth = int(depth)
        self.state_dim = int(state_dim)
        self.conv_kernel = int(conv_kernel)
        self.expand_factor = float(expand_factor)
        self.dropout = float(dropout)
        self.pooling = str(pooling)
        self.encoder_type = str(encoder_type)
        self.use_dual_encoder = bool(use_dual_encoder)
        self.horizons = sorted({int(h) for h in (horizons or [96])})
        self.target_features = int(target_features)
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.head_dropout = float(head_dropout)
        self.freeze_encoder = bool(freeze_encoder)
        self.auto_map = {
            "AutoConfig": "forecasting.CM_MambaForecastConfig",
            "AutoModel": "forecasting.CM_MambaForecastModel",
        }


class CM_MambaForecastModel(PreTrainedModel):
    """Forecasting model with optional dual encoders.

    Input shape: [B, T, F]
    Output shape: [B, H, C] where H is max horizon and C is target_features.
    """

    config_class = CM_MambaForecastConfig
    base_model_prefix = "cm_mamba_forecast"

    def __init__(self, config: CM_MambaForecastConfig) -> None:
        super().__init__(config)
        self.encoder_type = config.encoder_type
        self.use_dual_encoder = config.use_dual_encoder

        self.encoder = self._build_temporal_encoder(config)
        self.visual_encoder: Optional[nn.Module] = None
        if self.use_dual_encoder:
            self.visual_encoder = self._build_visual_encoder(config)
            head_input_dim = int(self.encoder.embedding_dim) + int(self.visual_encoder.embedding_dim)
            self.head = DualEncoderForecastMLP(
                input_dim=head_input_dim,
                hidden_dim=config.mlp_hidden_dim,
                horizons=config.horizons,
                target_features=config.target_features,
                dropout=config.head_dropout,
            )
        else:
            head_input_dim = int(self.encoder.embedding_dim)
            self.head = MultiHorizonForecastMLP(
                input_dim=head_input_dim,
                hidden_dim=config.mlp_hidden_dim,
                horizons=config.horizons,
                target_features=config.target_features,
                dropout=config.head_dropout,
            )

        if config.freeze_encoder:
            self._freeze_encoders()
        self.post_init()

    def _build_temporal_encoder(self, config: CM_MambaForecastConfig) -> MambaEncoder:
        return MambaEncoder(
            input_dim=config.input_dim,
            token_size=config.token_size,
            model_dim=config.model_dim,
            depth=config.depth,
            state_dim=config.state_dim,
            conv_kernel=config.conv_kernel,
            expand_factor=config.expand_factor,
            embedding_dim=config.embedding_dim,
            pooling=config.pooling,
            dropout=config.dropout,
        )

    def _build_visual_encoder(self, config: CM_MambaForecastConfig) -> MambaVisualEncoder:
        return MambaVisualEncoder(
            input_dim=config.input_dim,
            model_dim=config.model_dim,
            depth=config.depth,
            state_dim=config.state_dim,
            conv_kernel=config.conv_kernel,
            expand_factor=config.expand_factor,
            embedding_dim=config.embedding_dim,
            pooling=config.pooling,
            dropout=config.dropout,
        )

    def _freeze_encoders(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        if self.visual_encoder is not None:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        self.encoder.eval()
        if self.visual_encoder is not None:
            self.visual_encoder.eval()

    def get_encoder_only(self) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        """Return encoder module(s) without the forecast head."""
        if self.visual_encoder is None:
            return self.encoder
        return self.encoder, self.visual_encoder

    def forward(self, x: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        """Run forecasting inference.

        Args:
            x: Input tensor with shape [B, T, F].
            horizon: Optional horizon to slice outputs.

        Returns:
            Forecast predictions with shape [B, H, C].
        """
        if self.visual_encoder is None:
            embeddings = self.encoder(x)
        else:
            temporal = self.encoder(x)
            visual = self.visual_encoder(x)
            embeddings = torch.cat([temporal, visual], dim=-1)
        return self.head(embeddings, horizon=horizon)

    @staticmethod
    def _extract_state_dict(payload: Dict[str, torch.Tensor] | Dict[str, object]) -> Dict[str, torch.Tensor]:
        if "model_state_dict" in payload:
            return payload["model_state_dict"]  # type: ignore[return-value]
        if "state_dict" in payload:
            return payload["state_dict"]  # type: ignore[return-value]
        return payload  # type: ignore[return-value]

    def load_from_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        device: Optional[Union[str, torch.device]] = None,
        load_encoder: bool = True,
        load_head: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """Load model weights from a training checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            device: Optional device for loading the checkpoint.
            load_encoder: Whether to load encoder weights.
            load_head: Whether to load forecasting head weights.

        Returns:
            Tuple of (missing_keys, unexpected_keys) from ``load_state_dict``.
        """
        map_location = device if device is not None else "cpu"
        payload = torch.load(checkpoint_path, map_location=map_location)
        state_dict = self._extract_state_dict(payload)

        filtered: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("encoder.") and load_encoder:
                filtered[key] = value
            elif key.startswith("visual_encoder.") and load_encoder:
                filtered[key] = value
            elif key.startswith("head.") and load_head:
                filtered[key] = value
            elif not (key.startswith("encoder.") or key.startswith("visual_encoder.") or key.startswith("head.")):
                if load_encoder and load_head:
                    filtered[key] = value

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        return list(missing), list(unexpected)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        config: CM_MambaForecastConfig,
        checkpoint_path: Path,
        device: Optional[Union[str, torch.device]] = None,
        load_encoder: bool = True,
        load_head: bool = True,
    ) -> "CM_MambaForecastModel":
        """Create a model and load weights from a checkpoint."""
        model = cls(config)
        model.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            load_encoder=load_encoder,
            load_head=load_head,
        )
        if device is not None:
            model.to(device)
        return model

    @classmethod
    def from_checkpoint_encoder_only(
        cls,
        *,
        config: CM_MambaForecastConfig,
        checkpoint_path: Path,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        """Load encoder weights only and return the encoder module(s)."""
        model = cls.from_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            device=device,
            load_encoder=True,
            load_head=False,
        )
        return model.get_encoder_only()


@dataclass
class CM_MambaForecastExportSpec:
    """Configuration container for HF export scripts."""

    config: CM_MambaForecastConfig
    checkpoint_path: Path
    output_dir: Path
    model_card_template: Optional[Path] = None
    model_id: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "CM_MambaForecastExportSpec":
        model_cfg = payload.get("model") or {}
        forecast_cfg = payload.get("forecast") or {}
        paths_cfg = payload.get("paths") or {}

        config = CM_MambaForecastConfig(
            input_dim=int(model_cfg.get("input_dim", 32)),
            token_size=int(model_cfg.get("token_size", 32)),
            model_dim=int(model_cfg.get("model_dim", 128)),
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            depth=int(model_cfg.get("depth", 8)),
            state_dim=int(model_cfg.get("state_dim", 16)),
            conv_kernel=int(model_cfg.get("conv_kernel", 4)),
            expand_factor=float(model_cfg.get("expand_factor", 1.5)),
            dropout=float(model_cfg.get("dropout", 0.05)),
            pooling=str(model_cfg.get("pooling", "mean")),
            encoder_type=str(model_cfg.get("encoder_type", "temporal")),
            use_dual_encoder=bool(model_cfg.get("use_dual_encoder", False)),
            horizons=forecast_cfg.get("horizons", [96]),
            target_features=int(forecast_cfg.get("target_features", 1)),
            mlp_hidden_dim=int(forecast_cfg.get("mlp_hidden_dim", 512)),
            head_dropout=float(forecast_cfg.get("head_dropout", 0.1)),
            freeze_encoder=bool(forecast_cfg.get("freeze_encoder", True)),
        )

        checkpoint_value = paths_cfg.get("checkpoint")
        output_value = paths_cfg.get("output_dir")
        model_card_template = paths_cfg.get("model_card_template")

        if checkpoint_value is None or output_value is None:
            raise ValueError("paths.checkpoint and paths.output_dir must be provided")

        checkpoint_path = Path(str(checkpoint_value)).expanduser().resolve()
        output_dir = Path(str(output_value)).expanduser().resolve()
        template_path = (
            Path(model_card_template).expanduser().resolve() if model_card_template else None
        )
        model_id = paths_cfg.get("model_id")

        if not str(checkpoint_path):
            raise ValueError("paths.checkpoint is required in the export config")
        if not str(output_dir):
            raise ValueError("paths.output_dir is required in the export config")

        return cls(
            config=config,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            model_card_template=template_path,
            model_id=str(model_id) if model_id is not None else None,
        )
