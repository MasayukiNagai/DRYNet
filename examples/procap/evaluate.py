from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara import CAPY
from data import ProfileDataset, extract_loci, load_chrom_names
from file_config import FoldFilesConfig
from performance_metrics import compute_performance_metrics
from procapnet import ProCapNet
from train_utils import read_yaml, require_training_dependencies, select_device


PROFILE_METRIC_COLUMNS = [
    "nll",
    "cross_ent",
    "jsd",
    "profile_pearson",
    "profile_spearman",
    "profile_mse",
]
SUMMARY_METRIC_COLUMNS = [
    *PROFILE_METRIC_COLUMNS,
    "count_pearson",
    "count_spearman",
    "count_mse",
    "count_r2",
]
SPLITS = ["train", "val", "test", "all", "dnase_train", "dnase_val", "dnase_test"]


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained CAPY/ProCapNet models on PRO-cap splits.")
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--model_name", choices=["capy", "procapnet"], required=True)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--data_type", type=str, default="procap")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--reverse_complement", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="gpu", help="Device: gpu, cpu, auto, or a torch device string.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def split_peak_path(files: FoldFilesConfig, split: str) -> Path:
    if split == "train":
        return files.train_peak_path
    if split == "val":
        return files.val_peak_path
    if split == "test":
        return files.test_peak_path
    if split == "all":
        return files.all_peak_path
    if split == "dnase_train":
        return files.dnase_train_path
    if split == "dnase_val":
        return files.dnase_val_path
    if split == "dnase_test":
        return files.dnase_test_path
    raise ValueError(f"Unsupported split: {split}")


def model_params_for_name(params: dict[str, Any], model_name: str) -> dict[str, Any]:
    if model_name == "procapnet":
        return dict(params["model"])
    if model_name == "capy":
        return params
    raise ValueError(f"Unsupported model_name: {model_name}")


def load_model(model_name: str, params: dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if model_name == "procapnet":
        model = ProCapNet(**model_params_for_name(params, model_name))
    elif model_name == "capy":
        model = CAPY(model_params_for_name(params, model_name))
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def make_eval_loader(
    *,
    files: FoldFilesConfig,
    params: dict[str, Any],
    split: str,
    batch_size: int,
    num_workers: int,
    verbose: bool,
) -> DataLoader:
    dataset_params = params["dataset"]
    peak_path = split_peak_path(files, split)
    chroms = load_chrom_names(files.chrom_size_path)
    seqs, signals, _ = extract_loci(
        genome_path=files.genome_path,
        chroms=chroms,
        bw_paths=[files.plus_bw_path, files.minus_bw_path],
        bed_path=peak_path,
        mask_bw_path=None,
        input_length=int(dataset_params["input_length"]),
        output_length=int(dataset_params["output_length"]),
        max_jitter=0,
        verbose=verbose,
    )
    dataset = ProfileDataset(
        sequences=seqs,
        signals=signals,
        masks=None,
        input_length=int(dataset_params["input_length"]),
        output_length=int(dataset_params["output_length"]),
        max_jitter=0,
        reverse_complement=False,
        random_seed=dataset_params.get("seed"),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)


def _log_softmax_profiles(logits: torch.Tensor) -> torch.Tensor:
    flat = logits.reshape(logits.shape[0], -1)
    return torch.nn.functional.log_softmax(flat, dim=-1).reshape_as(logits)


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    reverse_complement: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, log_counts = model(x)
    log_probs = _log_softmax_profiles(logits)
    if not reverse_complement:
        return log_probs, log_counts

    rc_x = torch.flip(x, dims=(1, 2))
    rc_logits, rc_log_counts = model(rc_x)
    rc_logits = torch.flip(rc_logits, dims=(1, 2))
    rc_log_probs = _log_softmax_profiles(rc_logits)

    merged_probs = 0.5 * (torch.exp(log_probs) + torch.exp(rc_log_probs))
    merged_log_probs = torch.log(torch.clamp(merged_probs, min=torch.finfo(merged_probs.dtype).tiny))
    merged_log_counts = 0.5 * (log_counts + rc_log_counts)
    return merged_log_probs, merged_log_counts


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    reverse_complement: bool,
) -> dict[str, np.ndarray]:
    true_profiles = []
    pred_log_profiles = []
    pred_log_counts = []

    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"]
        log_probs, log_counts = predict_batch(model, x, reverse_complement=reverse_complement)
        true_profiles.append(y.detach().cpu().numpy())
        pred_log_profiles.append(log_probs.detach().cpu().numpy())
        pred_log_counts.append(log_counts.detach().cpu().numpy())

    if not true_profiles:
        raise RuntimeError("Evaluation dataloader produced no batches.")

    y_true = np.concatenate(true_profiles, axis=0)
    y_pred_log_profiles = np.concatenate(pred_log_profiles, axis=0)
    y_pred_log_counts = np.concatenate(pred_log_counts, axis=0)

    true_flat = y_true.reshape(y_true.shape[0], -1)
    true_metrics = np.expand_dims(true_flat, (1, 3))
    true_counts = true_metrics.sum(axis=2)
    pred_flat = y_pred_log_profiles.reshape(y_pred_log_profiles.shape[0], -1)
    pred_metrics = np.expand_dims(pred_flat, (1, 3))
    pred_counts = y_pred_log_counts.reshape(y_pred_log_counts.shape[0], -1)
    if pred_counts.shape[1] != 1:
        raise ValueError(f"Expected one predicted count per example, got shape {pred_counts.shape}")
    pred_counts = np.expand_dims(pred_counts, 1)

    metrics = compute_performance_metrics(
        true_metrics,
        pred_metrics,
        true_counts,
        pred_counts,
        smooth_true_profs=False,
        smooth_pred_profs=False,
    )
    return {
        "true_profiles": y_true,
        "true_log_counts": np.log1p(y_true.sum(axis=(1, 2))),
        "pred_log_profiles": y_pred_log_profiles,
        "pred_log_counts": y_pred_log_counts,
        "metrics": metrics,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_outputs(
    *,
    results: dict[str, np.ndarray],
    files: FoldFilesConfig,
    args: argparse.Namespace,
    params: dict[str, Any],
) -> dict[str, str]:
    eval_dir = files.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.cell_type
    split = args.split
    rc_suffix = "_rc" if args.reverse_complement else ""

    metrics = results["metrics"]
    summary = {key: float(np.nanmean(metrics[key])) for key in SUMMARY_METRIC_COLUMNS}
    summary.update(
        {
            "model_name": args.model_name,
            "cell_type": args.cell_type,
            "data_type": args.data_type,
            "fold": int(args.fold),
            "timestamp": args.timestamp,
            "split": split,
            "reverse_complement": bool(args.reverse_complement),
            "num_examples": int(results["true_profiles"].shape[0]),
        }
    )

    profile_rows = []
    num_examples = int(results["true_profiles"].shape[0])
    for i in range(num_examples):
        row = {"example_index": i}
        for key in PROFILE_METRIC_COLUMNS:
            row[key] = float(np.ravel(metrics[key])[i])
        profile_rows.append(row)

    summary_path = eval_dir / f"{prefix}_metrics_summary{rc_suffix}_{split}.csv"
    profile_path = eval_dir / f"{prefix}_metrics_profile{rc_suffix}_{split}.csv"
    log_path = eval_dir / f"{prefix}_eval_log{rc_suffix}_{split}.txt"
    write_csv(summary_path, [summary], list(summary.keys()))
    write_csv(profile_path, profile_rows, ["example_index", *PROFILE_METRIC_COLUMNS])

    saved_paths = {
        "metrics_summary": str(summary_path),
        "metrics_profile": str(profile_path),
        "eval_log": str(log_path),
    }
    if args.save_predictions:
        pred_profiles_path = eval_dir / f"{prefix}_log_pred_profiles{rc_suffix}_{split}.npy"
        pred_counts_path = eval_dir / f"{prefix}_log_pred_counts{rc_suffix}_{split}.npy"
        true_counts_path = eval_dir / f"{prefix}_log_true_counts{rc_suffix}_{split}.npy"
        np.save(pred_profiles_path, results["pred_log_profiles"])
        np.save(pred_counts_path, results["pred_log_counts"].reshape(num_examples, -1).squeeze(axis=1))
        np.save(true_counts_path, results["true_log_counts"])
        saved_paths["log_pred_profiles"] = str(pred_profiles_path)
        saved_paths["log_pred_counts"] = str(pred_counts_path)
        saved_paths["log_true_counts"] = str(true_counts_path)

    log_payload = {
        "args": vars(args),
        "checkpoint_path": str(files.best_checkpoint_path),
        "params_path": str(files.params_path),
        "peak_path": str(split_peak_path(files, split)),
        "output_length": int(params["dataset"]["output_length"]),
        "outputs": saved_paths,
        "summary": summary,
    }
    with log_path.open("w") as handle:
        handle.write(json.dumps(log_payload, default=json_default, indent=2, sort_keys=True) + "\n")
    return saved_paths


def run(args: argparse.Namespace) -> None:
    require_training_dependencies()
    files = FoldFilesConfig.create(
        proj_dir=args.proj_dir,
        cell_type=args.cell_type,
        data_type=args.data_type,
        fold=args.fold,
        model_name=args.model_name,
        timestamp=args.timestamp,
        use_unmappability_mask=False,
    )
    if not files.best_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {files.best_checkpoint_path}")
    if not files.params_path.exists():
        raise FileNotFoundError(f"Missing saved params: {files.params_path}")

    params = read_yaml(files.params_path)
    batch_size = int(args.batch_size or params["train"]["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else params["dataloader"]["num_workers"])
    device = select_device(args.device)

    print(f"Loading {args.model_name} checkpoint: {files.best_checkpoint_path}", flush=True)
    model = load_model(args.model_name, params, files.best_checkpoint_path, device)
    print(f"Loading {args.split} split data.", flush=True)
    dataloader = make_eval_loader(
        files=files,
        params=params,
        split=args.split,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=args.verbose,
    )
    print(f"Evaluating on {device}; reverse_complement={args.reverse_complement}", flush=True)
    results = evaluate_model(model, dataloader, device, reverse_complement=args.reverse_complement)
    saved_paths = save_outputs(results=results, files=files, args=args, params=params)
    print("Saved evaluation outputs:", flush=True)
    for name, path in saved_paths.items():
        print(f"  {name}: {path}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
