from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import ProCapDataModule
from file_config import FoldFilesConfig
from procapnet import ProCapNet
from train_utils import require_training_dependencies, select_device, train_model, write_yaml


PROCAPNET_PARAMS = {
    "project": "PRO-cap",
    "monitor": "valid_loss",
    "mode": "min",
    "model": {
        "n_filters": 512,
        "n_layers": 8,
        "n_outputs": 2,
        "trimming": 557,
    },
    "dataset": {
        "input_length": 2114,
        "output_length": 1000,
        "max_jitter": 200,
        "reverse_complement": True,
        "use_dnase": True,
        "source_fracs": [0.875, 0.125],
        "use_unmappability_mask": True,
    },
    "train": {
        "batch_size": 32,
        "learning_rate": 0.0005,
        "profile_loss_fn": "MNLLLoss",
        "count_loss_fn": "log1pMSELoss",
        "counts_weight": 100.0,
        "max_epochs": 500,
        "validation_iter": 100,
        "precision": 32,
    },
    "early_stopping": {
        "patience": 10,
    },
    "checkpoint": {
        "save_top_k": 1,
        "filename": "{epoch:02d}-{valid_loss:.2f}",
    },
    "dataloader": {
        "num_workers": 4,
        "prefetch_factor": 2,
        "pin_memory": True,
        "persistent_workers": True,
    },
    "lr_scheduler": {
        "scheduler_name": None,
        "scheduler_kwargs": None,
        "lr_scheduler_kwargs": None,
    },
    "wandb": {
        "enabled": True,
        "project": "capybara-procap",
        "entity": None,
        "mode": "online",
        "tags": [],
        "group": None,
        "name": None,
        "notes": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ProCapNet on processed PRO-cap data.")
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--data_type", type=str, default="procap")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--device", type=str, default="gpu", help="Device: gpu, cpu, auto, or a torch device string.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging for this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    require_training_dependencies()
    params = copy.deepcopy(PROCAPNET_PARAMS)
    if args.no_wandb:
        params.setdefault("wandb", {})["enabled"] = False
    dataset_params = params["dataset"]

    files = FoldFilesConfig.create(
        proj_dir=args.proj_dir,
        cell_type=args.cell_type,
        data_type=args.data_type,
        fold=args.fold,
        model_name="procapnet",
        timestamp=args.timestamp,
        use_unmappability_mask=bool(dataset_params["use_unmappability_mask"]),
    )
    files.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dict = files.as_dict()
    write_yaml(files.params_path, params)
    write_yaml(files.config_path, config_dict)

    data_config = {
        **config_dict,
        "input_length": int(dataset_params["input_length"]),
        "output_length": int(dataset_params["output_length"]),
        "max_jitter": int(dataset_params["max_jitter"]),
        "reverse_complement": bool(dataset_params["reverse_complement"]),
        "use_dnase": bool(dataset_params["use_dnase"]),
        "source_fracs": list(dataset_params["source_fracs"]),
        "random_seed": 0,
    }
    datamodule = ProCapDataModule(
        config=data_config,
        batch_size=int(params["train"]["batch_size"]),
        num_workers=int(params["dataloader"]["num_workers"]),
        prefetch_factor=params["dataloader"].get("prefetch_factor", 2),
        pin_memory=bool(params["dataloader"].get("pin_memory", True)),
        persistent_workers=bool(params["dataloader"].get("persistent_workers", True)),
        verbose=args.verbose,
    )

    model = ProCapNet(**params["model"])
    metadata = {
        "model_name": "procapnet",
        "params": params,
        "file_config": config_dict,
    }
    device = select_device(args.device)
    print(f"Training ProCapNet on {device}; outputs: {files.model_dir}", flush=True)
    train_model(
        model=model,
        datamodule=datamodule,
        output_paths=config_dict,
        params=params,
        device=device,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
