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

from capybara import CAPY, load_config
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
    resolved_count_finetune_config,
    select_device,
    trainable_parameter_count,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune count-specific CAPY/ProCapNet parameters.")
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--model_name", choices=["capy", "procapnet"], required=True)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--data_type", type=str, default="procap")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--mode", choices=["count_head", "final_layer"], default="final_layer")
    parser.add_argument("--learning_rate", type=float, default=5.0e-7)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--validation_iter",
        type=int,
        default=None,
        help="Validate every N training iterations. If omitted, inherit the source training config.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="gpu", help="Device: gpu, cpu, auto, or a torch device string.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging for this run.")
    return parser.parse_args()


def derived_timestamp(timestamp: str, mode: str) -> str:
    return fine_tune_timestamp(timestamp)


def make_files(
    *,
    args: argparse.Namespace,
    params: dict[str, Any],
    timestamp: str,
) -> FoldFilesConfig:
    return FoldFilesConfig.create(
        proj_dir=args.proj_dir,
        cell_type=args.cell_type,
        data_type=args.data_type,
        fold=args.fold,
        model_name=args.model_name,
        timestamp=timestamp,
        use_unmappability_mask=bool(params["dataset"]["use_unmappability_mask"]),
    )


def load_source_model(
    *,
    args: argparse.Namespace,
    params: dict[str, Any],
    source_files: FoldFilesConfig,
    device: torch.device,
) -> nn.Module:
    if args.model_name == "capy":
        model = CAPY(load_config(source_files.params_path))
    else:
        model = ProCapNet(**params["model"])

    checkpoint = torch.load(source_files.best_checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain model_state_dict: {source_files.best_checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def build_datamodule(
    *,
    files: FoldFilesConfig,
    params: dict[str, Any],
    batch_size: int,
    num_workers: int,
    verbose: bool,
) -> ProCapDataModule:
    dataset_params = params["dataset"]
    data_config = {
        **files.as_dict(),
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


def run(args: argparse.Namespace) -> None:
    require_training_dependencies()
    source_probe = FoldFilesConfig.create(
        proj_dir=args.proj_dir,
        cell_type=args.cell_type,
        data_type=args.data_type,
        fold=args.fold,
        model_name=args.model_name,
        timestamp=args.timestamp,
        use_unmappability_mask=False,
    )
    if not source_probe.params_path.exists():
        raise FileNotFoundError(f"Missing saved params: {source_probe.params_path}")
    source_params = read_yaml(source_probe.params_path)

    source_files = make_files(args=args, params=source_params, timestamp=args.timestamp)
    if not source_files.best_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing source best checkpoint: {source_files.best_checkpoint_path}")

    fine_tune_cfg = resolved_count_finetune_config(
        {
            "mode": args.mode,
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "validation_iter": args.validation_iter,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }
    )
    tuned_params = make_count_finetune_params(
        source_params,
        fine_tune_cfg,
        args.timestamp,
        no_wandb=args.no_wandb,
    )
    target_files = make_files(args=args, params=tuned_params, timestamp=derived_timestamp(args.timestamp, args.mode))
    target_files.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dict = target_files.as_dict()
    write_yaml(target_files.params_path, tuned_params)
    write_yaml(target_files.config_path, config_dict)

    batch_size = int(tuned_params["train"]["batch_size"])
    num_workers = int(tuned_params["dataloader"]["num_workers"])
    datamodule = build_datamodule(
        files=target_files,
        params=tuned_params,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=args.verbose,
    )

    device = select_device(args.device)
    model = load_source_model(args=args, params=source_params, source_files=source_files, device=device)
    trainable_names = configure_count_finetune_parameters(model, args.model_name, args.mode)
    trainable_count = trainable_parameter_count(model)
    alias_message = ""
    if args.model_name == "procapnet" and args.mode == "count_head":
        alias_message = " (alias for model.linear in ProCapNet)"

    metadata = {
        "model_name": args.model_name,
        "params": tuned_params,
        "file_config": config_dict,
        "finetune_count": {
            "source_model_dir": str(source_files.model_dir),
            "source_checkpoint_path": str(source_files.best_checkpoint_path),
            "mode": args.mode,
            "trainable_names": trainable_names,
        },
    }

    print(f"Source checkpoint: {source_files.best_checkpoint_path}", flush=True)
    print(f"Fine-tuning mode: {args.mode}{alias_message}", flush=True)
    print(f"Tuned outputs: {target_files.model_dir}", flush=True)
    print(f"Trainable parameters: {trainable_count}", flush=True)

    best_metrics = finetune_count_head(
        model=model,
        datamodule=datamodule,
        output_paths=config_dict,
        params=tuned_params,
        device=device,
        metadata=metadata,
        mode=args.mode,
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
    run(parse_args())


if __name__ == "__main__":
    main()
