from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import load_model, make_eval_loader
from file_config import FoldFilesConfig
from train_utils import read_yaml, require_training_dependencies, select_device


CALIBRATION_MODES = ("affine", "offset_only")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitData:
    split: str
    true_log_counts: np.ndarray
    pred_log_counts: np.ndarray
    prediction_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count-only split metrics and affine calibration analysis for CAPY/ProCapNet."
    )
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--model_name", choices=["capy", "procapnet"], required=True)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--data_type", type=str, default="procap")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--reverse_complement", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--recompute_saved_predictions",
        action="store_true",
        help="Ignore saved train/val/test count .npy files and recompute predictions from the checkpoint.",
    )
    parser.add_argument(
        "--save_generated_predictions",
        action="store_true",
        help="Save count predictions that had to be generated because no artifact was available.",
    )
    return parser.parse_args()


def count_prediction_path(files: FoldFilesConfig, split: str, reverse_complement: bool) -> Path:
    rc_suffix = "_rc" if reverse_complement else ""
    return files.eval_dir / f"{files.cell_type}_log_pred_counts{rc_suffix}_{split}.npy"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def predict_counts_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    reverse_complement: bool,
) -> torch.Tensor:
    _, log_counts = model(x)
    if not reverse_complement:
        return log_counts

    rc_x = torch.flip(x, dims=(1, 2))
    _, rc_log_counts = model(rc_x)
    return 0.5 * (log_counts + rc_log_counts)


@torch.no_grad()
def evaluate_counts(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    *,
    reverse_complement: bool,
) -> tuple[np.ndarray, np.ndarray]:
    true_log_counts = []
    pred_log_counts = []
    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"]
        log_counts = predict_counts_batch(model, x, reverse_complement=reverse_complement)
        true_total_counts = y.sum(dim=(1, 2))
        true_log_counts.append(torch.log1p(true_total_counts).detach().cpu().numpy())
        pred_log_counts.append(log_counts.reshape(-1).detach().cpu().numpy())

    if not true_log_counts:
        raise RuntimeError("Evaluation dataloader produced no batches.")
    return np.concatenate(true_log_counts), np.concatenate(pred_log_counts)


def collect_split_data(
    *,
    split: str,
    files: FoldFilesConfig,
    params: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    reverse_complement: bool,
    recompute_saved_predictions: bool,
    save_generated_predictions: bool,
    verbose: bool,
) -> SplitData:
    dataloader = make_eval_loader(
        files=files,
        params=params,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=verbose,
    )
    pred_path = count_prediction_path(files, split, reverse_complement)
    saved_pred = None
    if pred_path.exists() and not recompute_saved_predictions:
        saved_pred = np.asarray(np.load(pred_path)).reshape(-1).astype(np.float64)

    true_log_counts, generated_pred = evaluate_counts(
        model,
        dataloader,
        device,
        reverse_complement=reverse_complement,
    )
    true_log_counts = true_log_counts.astype(np.float64)
    generated_pred = generated_pred.astype(np.float64)

    if saved_pred is not None:
        if saved_pred.shape != true_log_counts.shape:
            raise ValueError(
                f"Saved prediction length mismatch for split={split}: "
                f"saved={saved_pred.shape}, true={true_log_counts.shape}"
            )
        pred_log_counts = saved_pred
        prediction_source = "saved"
    else:
        pred_log_counts = generated_pred
        prediction_source = "generated"
        if save_generated_predictions:
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(pred_path, pred_log_counts)

    if pred_log_counts.shape != true_log_counts.shape:
        raise ValueError(
            f"Prediction length mismatch for split={split}: "
            f"pred={pred_log_counts.shape}, true={true_log_counts.shape}"
        )
    return SplitData(
        split=split,
        true_log_counts=true_log_counts,
        pred_log_counts=pred_log_counts,
        prediction_source=prediction_source,
    )


def compute_count_metrics(true_log_counts: np.ndarray, pred_log_counts: np.ndarray) -> dict[str, float]:
    if true_log_counts.shape != pred_log_counts.shape:
        raise ValueError(f"Metric shape mismatch: true={true_log_counts.shape}, pred={pred_log_counts.shape}")
    finite = np.isfinite(true_log_counts) & np.isfinite(pred_log_counts)
    if not np.any(finite):
        raise ValueError("No finite true/pred count pairs are available.")

    truth = true_log_counts[finite]
    pred = pred_log_counts[finite]
    mse = float(np.mean((truth - pred) ** 2))
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "pearson": float(np.corrcoef(pred, truth)[0, 1]),
        "spearman": float(spearmanr(pred, truth).correlation),
        "mse": mse,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "num_examples": int(truth.size),
    }


def fit_calibration(true_log_counts: np.ndarray, pred_log_counts: np.ndarray, mode: str) -> tuple[float, float]:
    if true_log_counts.shape != pred_log_counts.shape:
        raise ValueError(
            f"Calibration shape mismatch: true={true_log_counts.shape}, pred={pred_log_counts.shape}"
        )
    if mode == "offset_only":
        return 1.0, float(np.mean(true_log_counts - pred_log_counts))
    if mode == "affine":
        design = np.column_stack([pred_log_counts, np.ones_like(pred_log_counts)])
        a, b = np.linalg.lstsq(design, true_log_counts, rcond=None)[0]
        return float(a), float(b)
    raise ValueError(f"Unsupported calibration mode: {mode}")


def metric_row(
    *,
    args: argparse.Namespace,
    split: str,
    prediction_source: str,
    prediction_kind: str,
    fit_split: str | None,
    mode: str | None,
    a: float | None,
    b: float | None,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "cell_type": args.cell_type,
        "data_type": args.data_type,
        "fold": int(args.fold),
        "timestamp": args.timestamp,
        "reverse_complement": bool(args.reverse_complement),
        "split": split,
        "prediction_source": prediction_source,
        "prediction_kind": prediction_kind,
        "fit_split": fit_split,
        "mode": mode,
        "a": a,
        "b": b,
        "count_pearson": metrics["pearson"],
        "count_spearman": metrics["spearman"],
        "count_mse": metrics["mse"],
        "count_r2": metrics["r2"],
        "num_examples": metrics["num_examples"],
    }


def print_calibration_table(rows: list[dict[str, Any]]) -> None:
    calibrated_rows = [row for row in rows if row["prediction_kind"] == "calibrated"]
    if not calibrated_rows:
        return

    headers = ("fit", "eval", "mode", "a", "b", "pearson", "spearman", "mse", "r2")
    line = (
        f"{headers[0]:>7}  {headers[1]:>7}  {headers[2]:>11}  "
        f"{headers[3]:>9}  {headers[4]:>9}  {headers[5]:>9}  "
        f"{headers[6]:>9}  {headers[7]:>9}  {headers[8]:>9}"
    )
    print("Calibrated count metrics:", flush=True)
    print(line, flush=True)
    print("-" * len(line), flush=True)
    for row in calibrated_rows:
        print(
            f"{str(row['fit_split']):>7}  {str(row['split']):>7}  {str(row['mode']):>11}  "
            f"{float(row['a']):>9.4f}  {float(row['b']):>9.4f}  "
            f"{float(row['count_pearson']):>9.4f}  {float(row['count_spearman']):>9.4f}  "
            f"{float(row['count_mse']):>9.4f}  {float(row['count_r2']):>9.4f}",
            flush=True,
        )

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
    model = load_model(args.model_name, params, files.best_checkpoint_path, device)

    split_data = {
        split: collect_split_data(
            split=split,
            files=files,
            params=params,
            model=model,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            reverse_complement=args.reverse_complement,
            recompute_saved_predictions=args.recompute_saved_predictions,
            save_generated_predictions=args.save_generated_predictions,
            verbose=args.verbose,
        )
        for split in SPLITS
    }

    rows: list[dict[str, Any]] = []
    raw_metrics = {}
    for split, data in split_data.items():
        metrics = compute_count_metrics(data.true_log_counts, data.pred_log_counts)
        raw_metrics[split] = metrics
        rows.append(
            metric_row(
                args=args,
                split=split,
                prediction_source=data.prediction_source,
                prediction_kind="raw",
                fit_split=None,
                mode=None,
                a=None,
                b=None,
                metrics=metrics,
            )
        )

    calibration_specs = [
        ("train", ("train", "val", "test")),
        ("val", ("val", "test")),
    ]
    calibration_params = []
    for fit_split, eval_splits in calibration_specs:
        fit_data = split_data[fit_split]
        for mode in CALIBRATION_MODES:
            a, b = fit_calibration(fit_data.true_log_counts, fit_data.pred_log_counts, mode)
            calibration_params.append({"fit_split": fit_split, "mode": mode, "a": a, "b": b})
            for eval_split in eval_splits:
                eval_data = split_data[eval_split]
                calibrated_pred = a * eval_data.pred_log_counts + b
                metrics = compute_count_metrics(eval_data.true_log_counts, calibrated_pred)
                rows.append(
                    metric_row(
                        args=args,
                        split=eval_split,
                        prediction_source=eval_data.prediction_source,
                        prediction_kind="calibrated",
                        fit_split=fit_split,
                        mode=mode,
                        a=a,
                        b=b,
                        metrics=metrics,
                    )
                )

    files.eval_dir.mkdir(parents=True, exist_ok=True)
    rc_suffix = "_rc" if args.reverse_complement else ""
    metrics_path = files.eval_dir / f"{args.cell_type}_count_calibration_metrics{rc_suffix}.csv"
    params_path = files.eval_dir / f"{args.cell_type}_count_calibration_params{rc_suffix}.json"
    write_csv(metrics_path, rows)
    params_payload = {
        "args": vars(args),
        "checkpoint_path": str(files.best_checkpoint_path),
        "params_path": str(files.params_path),
        "split_prediction_sources": {split: data.prediction_source for split, data in split_data.items()},
        "raw_metrics": raw_metrics,
        "calibration_params": calibration_params,
        "metrics_csv": str(metrics_path),
    }
    with params_path.open("w") as handle:
        json.dump(params_payload, handle, default=json_default, indent=2, sort_keys=True)
        handle.write("\n")

    print("Count calibration analysis complete.", flush=True)
    print(f"  metrics: {metrics_path}", flush=True)
    print(f"  params:  {params_path}", flush=True)
    for split in SPLITS:
        metrics = raw_metrics[split]
        print(
            f"  raw {split:>5}: pearson={metrics['pearson']:.4f} "
            f"spearman={metrics['spearman']:.4f} mse={metrics['mse']:.4f} r2={metrics['r2']:.4f}",
            flush=True,
        )


    print_calibration_table(rows)

def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
