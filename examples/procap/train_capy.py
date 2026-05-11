from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara import CAPY, load_config
from data import ProCapDataModule
from file_config import FoldFilesConfig
from train_utils import read_yaml, require_training_dependencies, select_device, train_model, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CAPY on processed PRO-cap data.")
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=REPO_ROOT / "configs" / "default_procap.yaml")
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
    params = read_yaml(args.params)
    if args.no_wandb:
        params.setdefault("wandb", {})["enabled"] = False
    model_cfg = load_config(args.params)

    dataset_params = params["dataset"]
    files = FoldFilesConfig.create(
        proj_dir=args.proj_dir,
        cell_type=args.cell_type,
        data_type=args.data_type,
        fold=args.fold,
        model_name="capy",
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
        "random_seed": dataset_params.get("seed"),
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

    model = CAPY(model_cfg)
    metadata = {
        "model_name": "capy",
        "params": params,
        "file_config": config_dict,
    }
    device = select_device(args.device)
    print(f"Training CAPY on {device}; outputs: {files.model_dir}", flush=True)
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
