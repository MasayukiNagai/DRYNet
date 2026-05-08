from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "input_length": 2048,
        "output_length": 1000,
    },
    "model": {
        "input_channels": 4,
        "input_length": 2048,
        "output_length": 1000,
        "dna_embedder_channels": 64,
        "num_groups": 8,
        "encoder_channels": [128, 192, 256],
        "pool_size": 2,
        "decoder_channels": [256, 192, 128, 64],
        "output_embedder_channels": 256,
        "embedding_projector": {"enabled": True},
        "norm_type": "batch",
        "activation": "gelu",
        "dropout": 0.1,
        "profile_head": {
            "source": "decoder",
            "num_outputs": 2,
            "kernel_size": 75,
        },
        "count_head": {
            "type": "ynet",
            "source": None,
            "conv_hidden_dims": [128, 64],
            "mlp_hidden_dims": [64],
            "kernel_size": 5,
            "pool_size": 2,
            "adaptive_pool_size": 4,
            "dropout": 0.1,
            "norm_type": "batch",
            "mlp_norm_type": "layer",
            "num_groups": 8,
        },
        "bottleneck": {
            "type": "hybrid_attention",
            "depth": 2,
            "kernel_size": 5,
            "dropout": 0.1,
            "norm_type": "batch",
            "activation": "gelu",
            "num_heads": 4,
            "attn_dropout": 0.1,
            "mlp_ratio": 2.0,
        },
    },
}

VALID_HEAD_SOURCES = {"encoder", "bottleneck", "decoder"}
VALID_COUNT_HEAD_TYPES = {"ynet", "unet"}
VALID_GLOBAL_POOL_TYPES = {"avg", "max", "avgmax"}
VALID_BOTTLENECK_TYPES = {"residual_conv", "hybrid_attention"}
VALID_NORM_TYPES = {"batch", "layer", "group", "none"}
MODEL_KEYS = {
    "input_channels",
    "input_length",
    "output_length",
    "n_outputs",
    "dna_embedder_channels",
    "num_groups",
    "encoder_channels",
    "pool_size",
    "decoder_channels",
    "output_embedder_channels",
    "embedding_projector",
    "norm_type",
    "activation",
    "dropout",
    "profile_head",
    "count_head",
    "bottleneck",
}
PROJECTOR_KEYS = {"enabled"}
PROFILE_HEAD_KEYS = {"source", "num_outputs", "kernel_size"}
COUNT_HEAD_KEYS = {
    "type",
    "source",
    "conv_hidden_dims",
    "mlp_hidden_dims",
    "kernel_size",
    "pool_size",
    "adaptive_pool_size",
    "global_pool",
    "dropout",
    "norm_type",
    "mlp_norm_type",
    "num_groups",
}
BOTTLENECK_KEYS = {
    "type",
    "depth",
    "kernel_size",
    "dropout",
    "num_groups",
    "norm_type",
    "activation",
    "num_heads",
    "attn_dropout",
    "mlp_ratio",
}


def default_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a normalized config dictionary.

    PyYAML is optional and only required when this function is used.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Loading YAML configs requires PyYAML. Install with `pip install pyyaml`.") from exc

    path = Path(path)
    with path.open("r") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must contain a mapping at the top level: {path}")
    return normalize_config(config)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _ensure_list(value: int | list[int] | tuple[int, ...], *, name: str, allow_empty: bool = False) -> list[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        if value:
            return [int(v) for v in value]
        if allow_empty:
            return []
    requirement = "an int, list[int], or tuple[int, ...]"
    if not allow_empty:
        requirement = f"a non-empty {requirement}"
    raise ValueError(f"{name} must be {requirement}.")


def _reject_unknown_keys(section: dict[str, Any], allowed: set[str], *, name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"Unsupported {name} config key(s): {unknown}")


def _validate_config_keys(config: dict[str, Any]) -> None:
    model_cfg = config.get("model", {})
    _reject_unknown_keys(model_cfg, MODEL_KEYS, name="model")
    _reject_unknown_keys(model_cfg.get("embedding_projector", {}), PROJECTOR_KEYS, name="model.embedding_projector")
    _reject_unknown_keys(model_cfg.get("profile_head", {}), PROFILE_HEAD_KEYS, name="model.profile_head")
    _reject_unknown_keys(model_cfg.get("count_head", {}), COUNT_HEAD_KEYS, name="model.count_head")
    _reject_unknown_keys(model_cfg.get("bottleneck", {}), BOTTLENECK_KEYS, name="model.bottleneck")


def normalize_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is not None:
        _validate_config_keys(config)
    config = _deep_update(DEFAULT_CONFIG, config or {})
    _validate_config_keys(config)

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    model_cfg["input_length"] = int(model_cfg.get("input_length", dataset_cfg["input_length"]))
    model_cfg["output_length"] = int(model_cfg.get("output_length", dataset_cfg["output_length"]))
    model_cfg["input_channels"] = int(model_cfg["input_channels"])
    model_cfg["dna_embedder_channels"] = int(model_cfg["dna_embedder_channels"])
    model_cfg["num_groups"] = int(model_cfg.get("num_groups", 8))
    if model_cfg["num_groups"] < 1:
        raise ValueError("model.num_groups must be >= 1.")
    model_cfg["pool_size"] = int(model_cfg["pool_size"])
    model_cfg["output_embedder_channels"] = int(model_cfg["output_embedder_channels"])
    model_cfg["encoder_channels"] = _ensure_list(model_cfg["encoder_channels"], name="encoder_channels")
    model_cfg["decoder_channels"] = _ensure_list(model_cfg["decoder_channels"], name="decoder_channels")

    expected_decoder_len = len(model_cfg["encoder_channels"]) + 1
    if len(model_cfg["decoder_channels"]) != expected_decoder_len:
        raise ValueError(
            "decoder_channels must specify one output channel count per UpResBlock. "
            f"Expected {expected_decoder_len}, got {len(model_cfg['decoder_channels'])}."
        )

    model_cfg["embedding_projector"]["enabled"] = bool(model_cfg["embedding_projector"]["enabled"])
    model_cfg["norm_type"] = str(model_cfg["norm_type"]).lower()
    if model_cfg["norm_type"] not in VALID_NORM_TYPES:
        raise ValueError(f"Unsupported model.norm_type: {model_cfg['norm_type']}")
    model_cfg["dropout"] = float(model_cfg["dropout"])

    profile_head_cfg = model_cfg["profile_head"]
    profile_head_cfg["source"] = str(profile_head_cfg["source"]).lower()
    if profile_head_cfg["source"] not in VALID_HEAD_SOURCES:
        raise ValueError(f"Unsupported model.profile_head.source: {profile_head_cfg['source']}")
    profile_head_cfg["num_outputs"] = int(profile_head_cfg.get("num_outputs", model_cfg.get("n_outputs", 2)))
    profile_head_cfg["kernel_size"] = int(profile_head_cfg["kernel_size"])

    count_head_cfg = model_cfg["count_head"]
    count_head_cfg["type"] = str(count_head_cfg.get("type", "ynet")).lower()
    if count_head_cfg["type"] not in VALID_COUNT_HEAD_TYPES:
        raise ValueError(f"Unsupported model.count_head.type: {count_head_cfg['type']}")
    source_value = count_head_cfg.get("source")
    if source_value is None:
        count_head_cfg["source"] = "bottleneck" if count_head_cfg["type"] == "ynet" else "decoder"
    else:
        count_head_cfg["source"] = str(source_value).lower()
    if count_head_cfg["source"] not in VALID_HEAD_SOURCES:
        raise ValueError(f"Unsupported model.count_head.source: {count_head_cfg['source']}")

    valid_count_sources = {"encoder", "bottleneck"} if count_head_cfg["type"] == "ynet" else {"decoder"}
    if count_head_cfg["source"] not in valid_count_sources:
        raise ValueError(
            f"model.count_head.type='{count_head_cfg['type']}' requires source in "
            f"{sorted(valid_count_sources)}, got '{count_head_cfg['source']}'."
        )

    allow_empty_for_unet = count_head_cfg["type"] == "unet"
    count_head_cfg["conv_hidden_dims"] = _ensure_list(
        count_head_cfg["conv_hidden_dims"],
        name="count_head.conv_hidden_dims",
        allow_empty=allow_empty_for_unet,
    )
    count_head_cfg["mlp_hidden_dims"] = _ensure_list(
        count_head_cfg["mlp_hidden_dims"],
        name="count_head.mlp_hidden_dims",
        allow_empty=allow_empty_for_unet,
    )
    count_head_cfg["kernel_size"] = int(count_head_cfg["kernel_size"])
    count_head_cfg["pool_size"] = int(count_head_cfg["pool_size"])
    count_head_cfg["adaptive_pool_size"] = int(count_head_cfg["adaptive_pool_size"])
    count_head_cfg["global_pool"] = str(count_head_cfg.get("global_pool", "avgmax")).lower()
    if count_head_cfg["global_pool"] not in VALID_GLOBAL_POOL_TYPES:
        raise ValueError(f"Unsupported model.count_head.global_pool: {count_head_cfg['global_pool']}")
    count_head_cfg["dropout"] = float(count_head_cfg["dropout"])
    count_head_cfg["norm_type"] = str(count_head_cfg.get("norm_type", model_cfg["norm_type"])).lower()
    count_head_cfg["mlp_norm_type"] = str(count_head_cfg.get("mlp_norm_type", "layer")).lower()
    for key in ("norm_type", "mlp_norm_type"):
        if count_head_cfg[key] not in VALID_NORM_TYPES:
            raise ValueError(f"Unsupported model.count_head.{key}: {count_head_cfg[key]}")
    count_head_cfg["num_groups"] = int(count_head_cfg.get("num_groups", model_cfg["num_groups"]))

    bottleneck_cfg = model_cfg["bottleneck"]
    bottleneck_cfg["type"] = str(bottleneck_cfg["type"]).lower()
    if bottleneck_cfg["type"] not in VALID_BOTTLENECK_TYPES:
        raise ValueError(f"Unsupported model.bottleneck.type: {bottleneck_cfg['type']}")
    bottleneck_cfg["depth"] = int(bottleneck_cfg["depth"])
    bottleneck_cfg["kernel_size"] = int(bottleneck_cfg["kernel_size"])
    bottleneck_cfg["dropout"] = float(bottleneck_cfg["dropout"])
    bottleneck_cfg["num_groups"] = int(bottleneck_cfg.get("num_groups", model_cfg["num_groups"]))
    bottleneck_cfg["norm_type"] = str(bottleneck_cfg["norm_type"]).lower()
    if bottleneck_cfg["norm_type"] not in VALID_NORM_TYPES:
        raise ValueError(f"Unsupported model.bottleneck.norm_type: {bottleneck_cfg['norm_type']}")
    bottleneck_cfg["num_heads"] = int(bottleneck_cfg["num_heads"])
    bottleneck_cfg["attn_dropout"] = float(bottleneck_cfg["attn_dropout"])
    bottleneck_cfg["mlp_ratio"] = float(bottleneck_cfg["mlp_ratio"])

    return config
