from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from capybara.losses import count_log1p_mse_loss, profile_mnll_loss
from performance_metrics import compute_performance_metrics

METRICS_COLUMNS = [
    "row_type",
    "epoch",
    "iteration",
    "train_loss",
    "train_profile_loss",
    "train_count_loss",
    "train_loss_epoch",
    "train_profile_loss_epoch",
    "train_count_loss_epoch",
    "valid_loss",
    "valid_profile_loss",
    "valid_count_loss",
    "valid_jsd",
    "valid_profile_pearson",
    "valid_count_pearson",
    "lr",
    "elapsed_seconds",
    "saved_best",
    "epoch_complete",
]

def require_training_dependencies() -> None:
    missing = []
    for module_name in ("pyBigWig", "pyfaidx", "scipy", "tqdm", "yaml"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise ImportError(
            "Missing required PRO-cap training dependencies: "
            + ", ".join(missing)
            + ". Run these scripts in the GPU/Torch environment that has the PRO-cap dependencies installed."
        )

def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open() as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data

def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

def to_wandb_config(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_wandb_config(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_wandb_config(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

class WandbLogger:
    def __init__(
        self,
        *,
        params: dict[str, Any],
        output_paths: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        self.run = None
        self.wandb = None
        self.disabled = False

        wandb_cfg = params.get("wandb", {})
        if not wandb_cfg.get("enabled", False):
            self.disabled = True
            return

        try:
            import wandb

            self.wandb = wandb
            file_config = metadata.get("file_config", {})
            model_name = str(metadata.get("model_name", file_config.get("model_name", "model")))
            cell_type = str(file_config.get("cell_type", "cell"))
            fold = file_config.get("fold", "fold")
            timestamp = str(file_config.get("timestamp", "run"))

            run_name = wandb_cfg.get("name") or f"{model_name}-{cell_type}-fold{fold}-{timestamp}"
            group = wandb_cfg.get("group") or f"{model_name}-{cell_type}"
            config = {
                "model_name": model_name,
                "cell_type": cell_type,
                "data_type": file_config.get("data_type"),
                "fold": fold,
                "timestamp": timestamp,
                "params": params,
                "file_config": file_config,
                "output_paths": output_paths,
            }

            self.run = wandb.init(
                project=wandb_cfg.get("project", "capybara-procap"),
                entity=wandb_cfg.get("entity"),
                name=run_name,
                group=group,
                tags=wandb_cfg.get("tags", []),
                notes=wandb_cfg.get("notes"),
                mode=wandb_cfg.get("mode", "online"),
                config=to_wandb_config(config),
            )
        except Exception as exc:
            self.disabled = True
            print(f"Warning: wandb logging disabled because initialization failed: {exc}", flush=True)

    def log(self, metrics: dict[str, Any]) -> None:
        if self.disabled or self.wandb is None or self.run is None:
            return
        try:
            step = metrics.get("iteration")
            self.wandb.log(to_wandb_config(metrics), step=int(step) if step is not None else None)
        except Exception as exc:
            self.disabled = True
            print(f"Warning: wandb logging disabled because log failed: {exc}", flush=True)

    def finish(self) -> None:
        if self.wandb is None or self.run is None:
            return
        try:
            self.wandb.finish()
        except Exception as exc:
            print(f"Warning: wandb finish failed: {exc}", flush=True)

def append_metrics(path: str | Path, row: dict[str, Any], *, write_header: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_keys = [key for key in row if key not in METRICS_COLUMNS]
    keys = METRICS_COLUMNS + extra_keys
    with path.open("a") as handle:
        if write_header:
            handle.write("\t".join(keys) + "\n")
        handle.write("\t".join("" if row.get(key) is None else str(row.get(key, "")) for key in keys) + "\n")
def select_device(device: str = "gpu") -> torch.device:
    device = device.lower()
    if device in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device gpu, but CUDA is not available.")
        return torch.device("cuda")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    return torch.device(device)

def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

def compute_losses(
    model: nn.Module,
    batch: dict[str, Tensor],
    counts_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    pred_profile_logits, pred_log_counts = model(batch["x"])
    loss_mask = batch.get("mask")
    profile_loss = profile_mnll_loss(pred_profile_logits, batch["y"], loss_mask=loss_mask)
    count_loss = count_log1p_mse_loss(pred_log_counts, batch["y"])
    total_loss = profile_loss + float(counts_weight) * count_loss
    return total_loss, profile_loss, count_loss

@torch.no_grad()
def validate(model: nn.Module, dataloader, device: torch.device, counts_weight: float) -> dict[str, float]:
    model.eval()
    true_profiles = []
    pred_log_profiles = []
    pred_log_counts = []
    for batch in dataloader:
        batch = move_batch(batch, device)
        logits, log_counts = model(batch["x"])
        log_probs = torch.nn.functional.log_softmax(logits.reshape(logits.shape[0], -1), dim=-1)

        true_profiles.append(batch["y"].detach().cpu().numpy())
        pred_log_profiles.append(log_probs.detach().cpu().numpy())
        pred_log_counts.append(log_counts.detach().cpu().numpy())

    if not true_profiles:
        raise RuntimeError("Validation dataloader produced no batches.")

    y_valid = np.concatenate(true_profiles)
    y_pred_log_probs = np.concatenate(pred_log_profiles)
    y_pred_log_counts = np.concatenate(pred_log_counts)

    y_valid = y_valid.reshape(y_valid.shape[0], -1)
    y_valid = np.expand_dims(y_valid, (1, 3))
    y_valid_counts = y_valid.sum(axis=2)
    y_pred_log_probs = np.expand_dims(y_pred_log_probs, (1, 3))
    y_pred_log_counts = np.expand_dims(y_pred_log_counts, 1)

    measures = compute_performance_metrics(y_valid, y_pred_log_probs, y_valid_counts, y_pred_log_counts, 7, 81)
    valid_profile_loss = float(measures["nll"].mean())
    valid_count_loss = float(measures["count_mse"].mean())
    return {
        "valid_loss": valid_profile_loss + float(counts_weight) * valid_count_loss,
        "valid_profile_loss": valid_profile_loss,
        "valid_jsd": float(measures["jsd"].mean()),
        "valid_profile_pearson": float(measures["profile_pearson"].mean()),
        "valid_count_pearson": float(measures["count_pearson"].mean()),
        "valid_count_loss": valid_count_loss,
    }

def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_valid_loss: float,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_valid_loss": best_valid_loss,
            "metadata": metadata,
        },
        path,
    )

def train_model(
    *,
    model: nn.Module,
    datamodule,
    output_paths: dict[str, str],
    params: dict[str, Any],
    device: torch.device,
    metadata: dict[str, Any],
) -> None:
    train_params = params["train"]
    counts_weight = float(train_params["counts_weight"])
    learning_rate = float(train_params["learning_rate"])
    max_epochs = int(train_params["max_epochs"])
    validation_iter = train_params.get("validation_iter")
    validation_iter = int(validation_iter) if validation_iter is not None else None
    patience = int(params.get("early_stopping", {}).get("patience", 10))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = None
    scheduler_cfg = params.get("lr_scheduler", {})
    if scheduler_cfg.get("scheduler_name") == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **scheduler_cfg.get("scheduler_kwargs", {}),
        )

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.val_dataloader()

    best_valid_loss = float("inf")
    best_epoch = -1
    metrics_path = Path(output_paths["metrics_path"])
    if metrics_path.exists():
        metrics_path.unlink()
    wandb_logger = WandbLogger(params=params, output_paths=output_paths, metadata=metadata)

    start = time.time()
    iteration = 0
    try:
        for epoch in range(max_epochs):
            model.train()
            epoch_total = 0.0
            epoch_profile = 0.0
            epoch_count = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                total_loss, profile_loss, count_loss = compute_losses(model, batch, counts_weight)
                total_loss.backward()
                optimizer.step()

                epoch_total += float(total_loss.detach().cpu())
                epoch_profile += float(profile_loss.detach().cpu())
                epoch_count += float(count_loss.detach().cpu())
                n_batches += 1

                should_validate = validation_iter is not None and iteration % validation_iter == 0
                if should_validate:
                    valid_metrics = validate(model, valid_loader, device, counts_weight)
                    if scheduler is not None:
                        scheduler.step(valid_metrics["valid_loss"])

                    train_metrics = {
                        "row_type": "validation",
                        "epoch": epoch,
                        "iteration": iteration,
                        "train_loss": float(total_loss.detach().cpu()),
                        "train_profile_loss": float(profile_loss.detach().cpu()),
                        "train_count_loss": float(count_loss.detach().cpu()),
                        **valid_metrics,
                        "lr": optimizer.param_groups[0]["lr"],
                        "elapsed_seconds": round(time.time() - start, 3),
                        "saved_best": valid_metrics["valid_loss"] < best_valid_loss,
                    }

                    if train_metrics["saved_best"]:
                        best_valid_loss = valid_metrics["valid_loss"]
                        best_epoch = epoch
                        save_checkpoint(
                            output_paths["best_checkpoint_path"],
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            best_valid_loss=best_valid_loss,
                            metadata=metadata,
                        )

                    append_metrics(metrics_path, train_metrics, write_header=not metrics_path.exists())
                    wandb_logger.log(train_metrics)
                    print(json.dumps(train_metrics), flush=True)

                    if best_epoch <= epoch - patience:
                        print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.", flush=True)
                        break

                iteration += 1

            if validation_iter is not None and best_epoch <= epoch - patience:
                save_checkpoint(
                    output_paths["last_checkpoint_path"],
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_valid_loss=best_valid_loss,
                    metadata=metadata,
                )
                break

            if n_batches == 0:
                raise RuntimeError("Training dataloader produced no batches.")
            epoch_metrics = {
                "row_type": "epoch",
                "epoch": epoch,
                "iteration": iteration,
                "train_loss_epoch": epoch_total / n_batches,
                "train_profile_loss_epoch": epoch_profile / n_batches,
                "train_count_loss_epoch": epoch_count / n_batches,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": round(time.time() - start, 3),
                "epoch_complete": True,
            }
            append_metrics(metrics_path, epoch_metrics, write_header=not metrics_path.exists())
            wandb_logger.log(epoch_metrics)
            print(json.dumps(epoch_metrics), flush=True)

            if validation_iter is None:
                valid_metrics = validate(model, valid_loader, device, counts_weight)
                if scheduler is not None:
                    scheduler.step(valid_metrics["valid_loss"])

                train_metrics = {
                    "row_type": "validation",
                    "epoch": epoch,
                    "iteration": iteration,
                    "train_loss": epoch_total / n_batches,
                    "train_profile_loss": epoch_profile / n_batches,
                    "train_count_loss": epoch_count / n_batches,
                    **valid_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": round(time.time() - start, 3),
                    "saved_best": valid_metrics["valid_loss"] < best_valid_loss,
                }

                if train_metrics["saved_best"]:
                    best_valid_loss = valid_metrics["valid_loss"]
                    best_epoch = epoch
                    save_checkpoint(
                        output_paths["best_checkpoint_path"],
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_valid_loss=best_valid_loss,
                        metadata=metadata,
                    )

                append_metrics(metrics_path, train_metrics, write_header=not metrics_path.exists())
                wandb_logger.log(train_metrics)
                print(json.dumps(train_metrics), flush=True)

            save_checkpoint(
                output_paths["last_checkpoint_path"],
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_valid_loss=best_valid_loss,
                metadata=metadata,
            )

            if validation_iter is None and best_epoch <= epoch - patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.", flush=True)
                break
    finally:
        wandb_logger.finish()
