from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
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
    "valid_count_spearman",
    "valid_count_r2",
    "lr",
    "elapsed_seconds",
    "saved_best",
    "epoch_complete",
]

COUNT_FINETUNE_METRICS_COLUMNS = [
    "row_type",
    "epoch",
    "iteration",
    "train_count_loss",
    "train_count_loss_epoch",
    "valid_count_loss",
    "valid_count_pearson",
    "valid_count_spearman",
    "valid_count_r2",
    "lr",
    "elapsed_seconds",
    "saved_best",
    "epoch_complete",
]

VALID_TRAINING_STAGES = {"train", "finetune", "both"}
COUNT_FINETUNE_DEFAULTS = {
    "mode": "final_layer",
    "learning_rate": 5.0e-7,
    "max_epochs": 50,
    "patience": 10,
    "weight_decay": 0.0,
    "validation_iter": 300,
    "batch_size": None,
    "num_workers": None,
}


def validate_training_stage(stage: str) -> str:
    stage = str(stage).lower()
    if stage not in VALID_TRAINING_STAGES:
        raise ValueError(f"Unsupported training stage: {stage!r}. Expected one of {sorted(VALID_TRAINING_STAGES)}.")
    return stage


def fine_tune_timestamp(timestamp: str) -> str:
    return f"{timestamp}_ft"


def resolved_count_finetune_config(fine_tune_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = copy.deepcopy(COUNT_FINETUNE_DEFAULTS)
    cfg.update(copy.deepcopy(fine_tune_cfg or {}))
    cfg["mode"] = str(cfg["mode"])
    if cfg["mode"] not in {"count_head", "final_layer"}:
        raise ValueError("fine_tune.mode must be 'count_head' or 'final_layer'.")
    return cfg


def make_count_finetune_params(
    source_params: dict[str, Any],
    fine_tune_cfg: dict[str, Any] | None,
    source_timestamp: str,
    *,
    no_wandb: bool = False,
) -> dict[str, Any]:
    cfg = resolved_count_finetune_config(fine_tune_cfg)
    tuned_params = copy.deepcopy(source_params)
    if no_wandb:
        tuned_params.setdefault("wandb", {})["enabled"] = False

    tuned_params.setdefault("train", {})
    tuned_params["train"]["learning_rate"] = float(cfg["learning_rate"])
    tuned_params["train"]["max_epochs"] = int(cfg["max_epochs"])
    tuned_params["train"]["weight_decay"] = float(cfg["weight_decay"])
    tuned_params["train"]["validation_iter"] = None if cfg["validation_iter"] is None else int(cfg["validation_iter"])
    if cfg["batch_size"] is not None:
        tuned_params["train"]["batch_size"] = int(cfg["batch_size"])
    if cfg["num_workers"] is not None:
        tuned_params.setdefault("dataloader", {})["num_workers"] = int(cfg["num_workers"])

    tuned_params.setdefault("early_stopping", {})["patience"] = int(cfg["patience"])
    tuned_params["fine_tune"] = cfg
    tuned_params["finetune_count"] = {
        "source_timestamp": source_timestamp,
        "mode": cfg["mode"],
        "learning_rate": float(cfg["learning_rate"]),
        "max_epochs": int(cfg["max_epochs"]),
        "patience": int(cfg["patience"]),
        "weight_decay": float(cfg["weight_decay"]),
        "validation_iter": tuned_params["train"].get("validation_iter"),
    }
    return tuned_params


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
            print(
                f"Warning: wandb logging disabled because initialization failed: {exc}", flush=True
            )

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


def append_metrics_row(
    path: str | Path,
    row: dict[str, Any],
    *,
    write_header: bool,
    columns: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_keys = [key for key in row if key not in columns]
    keys = columns + extra_keys
    with path.open("a") as handle:
        if write_header:
            handle.write("\t".join(keys) + "\n")
        handle.write(
            "\t".join("" if row.get(key) is None else str(row.get(key, "")) for key in keys) + "\n"
        )


def append_metrics(path: str | Path, row: dict[str, Any], *, write_header: bool) -> None:
    append_metrics_row(path, row, write_header=write_header, columns=METRICS_COLUMNS)


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
def validate(
    model: nn.Module, dataloader, device: torch.device, counts_weight: float
) -> dict[str, float]:
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

    measures = compute_performance_metrics(
        y_valid, y_pred_log_probs, y_valid_counts, y_pred_log_counts, 7, 81
    )
    valid_profile_loss = float(measures["nll"].mean())
    valid_count_loss = float(measures["count_mse"].mean())
    return {
        "valid_loss": valid_profile_loss + float(counts_weight) * valid_count_loss,
        "valid_profile_loss": valid_profile_loss,
        "valid_jsd": float(measures["jsd"].mean()),
        "valid_profile_pearson": float(measures["profile_pearson"].mean()),
        "valid_count_pearson": float(measures["count_pearson"].mean()),
        "valid_count_spearman": float(measures["count_spearman"].mean()),
        "valid_count_loss": valid_count_loss,
        "valid_count_r2": float(measures["count_r2"].mean()),
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
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_metric_name="best_valid_loss",
        best_metric_value=best_valid_loss,
        metadata=metadata,
    )


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric_name: str,
    best_metric_value: float,
    metadata: dict[str, Any],
    extra_values: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        best_metric_name: best_metric_value,
        "metadata": metadata,
    }
    if extra_values:
        checkpoint.update(extra_values)
    torch.save(checkpoint, path)


def final_count_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "procapnet":
        if not hasattr(model, "linear"):
            raise AttributeError("ProCapNet model does not expose linear count layer.")
        return model.linear

    if not hasattr(model, "count_head") or not hasattr(model.count_head, "mlp"):
        raise AttributeError("CAPY model does not expose count_head.mlp.")
    last_layer = model.count_head.mlp[-1]
    if not isinstance(last_layer, nn.Linear) or last_layer.out_features != 1:
        raise TypeError("Expected CAPY count_head.mlp[-1] to be nn.Linear(..., 1).")
    return last_layer


def configure_count_finetune_parameters(model: nn.Module, model_name: str, mode: str) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if model_name == "capy" and mode == "count_head":
        modules = [model.count_head]
    elif mode in {"count_head", "final_layer"}:
        modules = [final_count_layer(model, model_name)]
    else:
        raise ValueError(f"Unsupported count fine-tuning mode: {mode}")

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names:
        raise RuntimeError("No trainable parameters were selected.")
    return trainable_names


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def set_count_finetune_train_mode(model: nn.Module, model_name: str, mode: str) -> None:
    model.eval()
    if model_name == "capy" and mode == "count_head":
        model.count_head.train()
    else:
        final_count_layer(model, model_name).train()


def count_only_metrics(true_log_counts: np.ndarray, pred_log_counts: np.ndarray) -> dict[str, float]:
    if true_log_counts.shape != pred_log_counts.shape:
        raise ValueError(f"Metric shape mismatch: true={true_log_counts.shape}, pred={pred_log_counts.shape}")
    finite = np.isfinite(true_log_counts) & np.isfinite(pred_log_counts)
    if not np.any(finite):
        raise ValueError("No finite validation count pairs are available.")

    truth = true_log_counts[finite]
    pred = pred_log_counts[finite]
    mse = float(np.mean((truth - pred) ** 2))
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "valid_count_loss": mse,
        "valid_count_pearson": float(np.corrcoef(pred, truth)[0, 1]),
        "valid_count_spearman": float(spearmanr(pred, truth).correlation),
        "valid_count_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


@torch.no_grad()
def validate_count_only(model: nn.Module, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    true_log_counts = []
    pred_log_counts = []
    for batch in dataloader:
        batch = move_batch(batch, device)
        _, log_counts = model(batch["x"])
        true_counts = batch["y"].sum(dim=(1, 2))
        true_log_counts.append(torch.log1p(true_counts).detach().cpu().numpy())
        pred_log_counts.append(log_counts.reshape(-1).detach().cpu().numpy())
    if not true_log_counts:
        raise RuntimeError("Validation dataloader produced no batches.")
    return count_only_metrics(np.concatenate(true_log_counts), np.concatenate(pred_log_counts))


def save_count_finetune_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_valid_count_loss: float,
    metadata: dict[str, Any],
) -> None:
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_metric_name="best_valid_count_loss",
        best_metric_value=best_valid_count_loss,
        metadata=metadata,
        extra_values={"best_valid_loss": best_valid_count_loss},
    )


def append_count_finetune_metrics(path: str | Path, row: dict[str, Any], *, write_header: bool) -> None:
    append_metrics_row(
        path,
        row,
        write_header=write_header,
        columns=COUNT_FINETUNE_METRICS_COLUMNS,
    )


def run_training_loop(
    *,
    model: nn.Module,
    datamodule,
    output_paths: dict[str, str],
    params: dict[str, Any],
    device: torch.device,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    train_step_fn,
    validate_fn,
    set_train_mode_fn,
    metrics_columns: list[str],
    selection_metric: str,
    best_metric_name: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    checkpoint_extra_values_fn=None,
) -> dict[str, Any]:
    train_params = params["train"]
    max_epochs = int(train_params["max_epochs"])
    validation_iter = train_params.get("validation_iter")
    validation_iter = int(validation_iter) if validation_iter is not None else None
    patience = int(params.get("early_stopping", {}).get("patience", 10))

    model.to(device)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.val_dataloader()

    best_metric_value = float("inf")
    best_epoch = -1
    best_metrics: dict[str, float] | None = None
    metrics_path = Path(output_paths["metrics_path"])
    if metrics_path.exists():
        metrics_path.unlink()
    wandb_logger = WandbLogger(params=params, output_paths=output_paths, metadata=metadata)

    def save_loop_checkpoint(path: str | Path, epoch: int) -> None:
        extra_values = None
        if checkpoint_extra_values_fn is not None:
            extra_values = checkpoint_extra_values_fn(best_metric_value)
        save_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            metadata=metadata,
            extra_values=extra_values,
        )

    def write_row(row: dict[str, Any]) -> None:
        append_metrics_row(
            metrics_path,
            row,
            write_header=not metrics_path.exists(),
            columns=metrics_columns,
        )
        wandb_logger.log(row)
        print(json.dumps(row), flush=True)

    def validate_and_log(
        *,
        epoch: int,
        iteration: int,
        train_metrics: dict[str, float],
        start: float,
    ) -> None:
        nonlocal best_metric_value, best_epoch, best_metrics

        valid_metrics = validate_fn(model, valid_loader, device)
        if scheduler is not None:
            scheduler.step(valid_metrics[selection_metric])

        saved_best = valid_metrics[selection_metric] < best_metric_value
        if saved_best:
            best_metric_value = valid_metrics[selection_metric]
            best_epoch = epoch
            best_metrics = dict(valid_metrics)
            save_loop_checkpoint(output_paths["best_checkpoint_path"], epoch)

        row = {
            "row_type": "validation",
            "epoch": epoch,
            "iteration": iteration,
            **train_metrics,
            **valid_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": round(time.time() - start, 3),
            "saved_best": saved_best,
        }
        write_row(row)

    start = time.time()
    iteration = 0
    try:
        for epoch in range(max_epochs):
            set_train_mode_fn(model)
            epoch_sums: dict[str, float] = {}
            last_train_metrics: dict[str, float] = {}
            n_batches = 0

            for batch in train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, train_metrics = train_step_fn(model, batch)
                loss.backward()
                optimizer.step()

                last_train_metrics = {
                    key: float(value.detach().cpu()) if isinstance(value, Tensor) else float(value)
                    for key, value in train_metrics.items()
                }
                for key, value in last_train_metrics.items():
                    epoch_sums[key] = epoch_sums.get(key, 0.0) + value
                n_batches += 1

                should_validate = validation_iter is not None and iteration % validation_iter == 0
                if should_validate:
                    validate_and_log(
                        epoch=epoch,
                        iteration=iteration,
                        train_metrics=last_train_metrics,
                        start=start,
                    )

                    if best_epoch <= epoch - patience:
                        print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.", flush=True)
                        break
                    set_train_mode_fn(model)

                iteration += 1

            if validation_iter is not None and best_epoch <= epoch - patience:
                save_loop_checkpoint(output_paths["last_checkpoint_path"], epoch)
                break

            if n_batches == 0:
                raise RuntimeError("Training dataloader produced no batches.")
            epoch_averages = {key: value / n_batches for key, value in epoch_sums.items()}
            epoch_metrics = {
                "row_type": "epoch",
                "epoch": epoch,
                "iteration": iteration,
                **{f"{key}_epoch": value for key, value in epoch_averages.items()},
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": round(time.time() - start, 3),
                "epoch_complete": True,
            }
            write_row(epoch_metrics)

            if validation_iter is None:
                validate_and_log(
                    epoch=epoch,
                    iteration=iteration,
                    train_metrics=epoch_averages,
                    start=start,
                )

            save_loop_checkpoint(output_paths["last_checkpoint_path"], epoch)

            if validation_iter is None and best_epoch <= epoch - patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.", flush=True)
                break
    finally:
        if best_metrics is not None:
            summary_metrics = {
                "row_type": "summary",
                "iteration": int(iteration),
                "best_epoch": int(best_epoch),
                best_metric_name: best_metric_value,
            }
            for key, value in best_metrics.items():
                prefixed_key = f"best_{key}"
                if prefixed_key not in summary_metrics:
                    summary_metrics[prefixed_key] = value
            wandb_logger.log(summary_metrics)
        wandb_logger.finish()

    if best_metrics is None:
        raise RuntimeError("No best validation checkpoint was recorded.")
    return {
        "best_epoch": int(best_epoch),
        best_metric_name: best_metric_value,
        **best_metrics,
    }


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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = None
    scheduler_cfg = params.get("lr_scheduler", {})
    if scheduler_cfg.get("scheduler_name") == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **scheduler_cfg.get("scheduler_kwargs", {}),
        )

    def train_joint_step(current_model: nn.Module, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        total_loss, profile_loss, count_loss = compute_losses(current_model, batch, counts_weight)
        return total_loss, {
            "train_loss": total_loss,
            "train_profile_loss": profile_loss,
            "train_count_loss": count_loss,
        }

    run_training_loop(
        model=model,
        datamodule=datamodule,
        output_paths=output_paths,
        params=params,
        device=device,
        metadata=metadata,
        optimizer=optimizer,
        train_step_fn=train_joint_step,
        validate_fn=lambda current_model, valid_loader, current_device: validate(
            current_model, valid_loader, current_device, counts_weight
        ),
        set_train_mode_fn=lambda current_model: current_model.train(),
        metrics_columns=METRICS_COLUMNS,
        selection_metric="valid_loss",
        best_metric_name="best_valid_loss",
        scheduler=scheduler,
    )


def finetune_count_head(
    *,
    model: nn.Module,
    datamodule,
    output_paths: dict[str, str],
    params: dict[str, Any],
    device: torch.device,
    metadata: dict[str, Any],
    mode: str,
) -> dict[str, float]:
    train_params = params["train"]
    learning_rate = float(train_params["learning_rate"])
    weight_decay = float(train_params.get("weight_decay", 0.0))
    model_name = str(metadata["model_name"])

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def train_count_step(current_model: nn.Module, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        _, log_counts = current_model(batch["x"])
        count_loss = count_log1p_mse_loss(log_counts, batch["y"])
        return count_loss, {"train_count_loss": count_loss}

    return run_training_loop(
        model=model,
        datamodule=datamodule,
        output_paths=output_paths,
        params=params,
        device=device,
        metadata=metadata,
        optimizer=optimizer,
        train_step_fn=train_count_step,
        validate_fn=lambda current_model, valid_loader, current_device: validate_count_only(
            current_model, valid_loader, current_device
        ),
        set_train_mode_fn=lambda current_model: set_count_finetune_train_mode(
            current_model, model_name, mode
        ),
        metrics_columns=COUNT_FINETUNE_METRICS_COLUMNS,
        selection_metric="valid_count_loss",
        best_metric_name="best_valid_count_loss",
        checkpoint_extra_values_fn=lambda best_value: {"best_valid_loss": best_value},
    )
