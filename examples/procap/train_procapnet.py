from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import ProCapDataModule
from file_config import FoldFilesConfig
from procapnet import ProCapNet
from train_utils import (
    configure_count_finetune_parameters,
    fine_tune_timestamp,
    finetune_count_head,
    make_count_finetune_params,
    read_yaml,
    require_training_dependencies,
    select_device,
    train_model,
    trainable_parameter_count,
    validate_training_stage,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ProCapNet on processed PRO-cap data.")
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=REPO_ROOT / "configs" / "default_procapnet.yaml")
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--data_type", type=str, default="procap")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--stage", choices=["train", "finetune", "both"], default="both")
    parser.add_argument("--device", type=str, default="gpu", help="Device: gpu, cpu, auto, or a torch device string.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging for this run.")
    return parser.parse_args()


def make_files(args: argparse.Namespace, params: dict[str, Any], timestamp: str | None) -> FoldFilesConfig:
    dataset_params = params["dataset"]
    return FoldFilesConfig.create(
        proj_dir=args.proj_dir,
        cell_type=args.cell_type,
        data_type=args.data_type,
        fold=args.fold,
        model_name="procapnet",
        timestamp=timestamp,
        use_unmappability_mask=bool(dataset_params["use_unmappability_mask"]),
    )


def build_datamodule(
    *,
    files: FoldFilesConfig,
    params: dict[str, Any],
    batch_size: int,
    num_workers: int,
    verbose: bool,
) -> ProCapDataModule:
    dataset_params = params["dataset"]
    config_dict = files.as_dict()
    data_config = {
        **config_dict,
        "input_length": int(dataset_params["input_length"]),
        "output_length": int(dataset_params["output_length"]),
        "max_jitter": int(dataset_params["max_jitter"]),
        "reverse_complement": bool(dataset_params["reverse_complement"]),
        "use_dnase": bool(dataset_params["use_dnase"]),
        "source_fracs": list(dataset_params["source_fracs"]),
        "random_seed": dataset_params.get("seed"),
    }
    return ProCapDataModule(
        config=data_config,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=params["dataloader"].get("prefetch_factor", 2),
        pin_memory=bool(params["dataloader"].get("pin_memory", True)),
        persistent_workers=bool(params["dataloader"].get("persistent_workers", True)),
        verbose=verbose,
    )


def load_source_model(params: dict[str, Any], files: FoldFilesConfig, device: torch.device) -> nn.Module:
    model = ProCapNet(**params["model"])
    checkpoint = torch.load(files.best_checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain model_state_dict: {files.best_checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def run_train_stage(
    *,
    args: argparse.Namespace,
    params: dict[str, Any],
    device: torch.device,
) -> FoldFilesConfig:
    files = make_files(args, params, args.timestamp)
    files.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dict = files.as_dict()
    write_yaml(files.params_path, params)
    write_yaml(files.config_path, config_dict)

    datamodule = build_datamodule(
        files=files,
        params=params,
        batch_size=int(params["train"]["batch_size"]),
        num_workers=int(params["dataloader"]["num_workers"]),
        verbose=args.verbose,
    )

    metadata = {
        "model_name": "procapnet",
        "params": params,
        "file_config": config_dict,
    }
    print(f"Training ProCapNet on {device}; outputs: {files.model_dir}", flush=True)
    train_model(
        model=ProCapNet(**params["model"]),
        datamodule=datamodule,
        output_paths=config_dict,
        params=params,
        device=device,
        metadata=metadata,
    )
    return files


def run_finetune_stage(
    *,
    args: argparse.Namespace,
    requested_params: dict[str, Any],
    source_files: FoldFilesConfig,
    device: torch.device,
) -> None:
    if not source_files.params_path.exists():
        raise FileNotFoundError(f"Missing saved params: {source_files.params_path}")
    if not source_files.best_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing source best checkpoint: {source_files.best_checkpoint_path}")

    source_params = read_yaml(source_files.params_path)
    tuned_params = make_count_finetune_params(
        source_params,
        requested_params.get("fine_tune"),
        source_files.timestamp,
        no_wandb=args.no_wandb,
    )
    target_files = make_files(args, tuned_params, fine_tune_timestamp(source_files.timestamp))
    target_files.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dict = target_files.as_dict()
    write_yaml(target_files.params_path, tuned_params)
    write_yaml(target_files.config_path, config_dict)

    datamodule = build_datamodule(
        files=target_files,
        params=tuned_params,
        batch_size=int(tuned_params["train"]["batch_size"]),
        num_workers=int(tuned_params["dataloader"]["num_workers"]),
        verbose=args.verbose,
    )

    fine_tune_cfg = tuned_params["fine_tune"]
    model = load_source_model(source_params, source_files, device)
    trainable_names = configure_count_finetune_parameters(model, "procapnet", fine_tune_cfg["mode"])
    trainable_count = trainable_parameter_count(model)
    alias_message = ""
    if fine_tune_cfg["mode"] == "count_head":
        alias_message = " (alias for model.linear in ProCapNet)"
    metadata = {
        "model_name": "procapnet",
        "params": tuned_params,
        "file_config": config_dict,
        "finetune_count": {
            "source_model_dir": str(source_files.model_dir),
            "source_checkpoint_path": str(source_files.best_checkpoint_path),
            "mode": fine_tune_cfg["mode"],
            "trainable_names": trainable_names,
        },
    }

    print(f"Source checkpoint: {source_files.best_checkpoint_path}", flush=True)
    print(f"Fine-tuning mode: {fine_tune_cfg['mode']}{alias_message}", flush=True)
    print(f"Tuned outputs: {target_files.model_dir}", flush=True)
    print(f"Trainable parameters: {trainable_count}", flush=True)
    best_metrics = finetune_count_head(
        model=model,
        datamodule=datamodule,
        output_paths=config_dict,
        params=tuned_params,
        device=device,
        metadata=metadata,
        mode=fine_tune_cfg["mode"],
    )
    print(
        "Best validation checkpoint: "
        f"epoch={best_metrics['best_epoch']} "
        f"loss={best_metrics['best_valid_count_loss']:.6f} "
        f"pearson={best_metrics['valid_count_pearson']:.4f} "
        f"spearman={best_metrics['valid_count_spearman']:.4f} "
        f"r2={best_metrics['valid_count_r2']:.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()

    require_training_dependencies()
    stage = validate_training_stage(args.stage)
    if stage == "finetune" and args.timestamp is None:
        raise ValueError("--timestamp is required when --stage finetune.")

    params = read_yaml(args.params)
    if args.no_wandb:
        params.setdefault("wandb", {})["enabled"] = False

    device = select_device(args.device)
    source_files = None
    if stage in {"train", "both"}:
        source_files = run_train_stage(args=args, params=params, device=device)
    else:
        source_files = make_files(args, params, args.timestamp)

    if stage in {"finetune", "both"}:
        run_finetune_stage(
            args=args,
            requested_params=params,
            source_files=source_files,
            device=device,
        )


if __name__ == "__main__":
    main()
