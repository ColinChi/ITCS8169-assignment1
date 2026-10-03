"""Supervised training and evaluation engine."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import ClassificationAccumulator
from .utils import save_history_csv, save_json


class ExponentialMovingAverage:
    """Maintain an evaluation-only exponential moving average of model state."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be strictly between 0 and 1")
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        averaged_state = self.module.state_dict()
        current_state = model.state_dict()
        if averaged_state.keys() != current_state.keys():
            raise ValueError("EMA and training model states do not match")
        for key, averaged_value in averaged_state.items():
            current_value = current_state[key].detach()
            if torch.is_floating_point(averaged_value):
                averaged_value.mul_(self.decay).add_(
                    current_value, alpha=1.0 - self.decay
                )
            else:
                averaged_value.copy_(current_value)


def mixup_batch(
    images: torch.Tensor, labels: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Mix a batch and return the paired targets and dominant mixing weight."""
    if alpha <= 0.0:
        raise ValueError("MixUp alpha must be positive")
    if images.shape[0] != labels.shape[0]:
        raise ValueError("Images and labels must have the same batch size")
    mixing_weight = float(torch.distributions.Beta(alpha, alpha).sample().item())
    # Keep the original target dominant so the reported training accuracy remains
    # interpretable; the expected MixUp loss is unchanged by this symmetry.
    mixing_weight = max(mixing_weight, 1.0 - mixing_weight)
    permutation = torch.randperm(images.shape[0], device=images.device)
    mixed_images = (
        mixing_weight * images + (1.0 - mixing_weight) * images[permutation]
    )
    return mixed_images, labels[permutation], mixing_weight


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = True,
    grad_clip_norm: float | None = None,
    auxiliary_weight: float = 0.0,
    auxiliary_mask_ratio: float = 0.40,
    ema: ExponentialMovingAverage | None = None,
    mixup_alpha: float = 0.0,
    mixup_probability: float = 1.0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    accumulator = ClassificationAccumulator(num_classes=model_num_classes(model))
    use_amp = bool(amp and device.type == "cuda")
    auxiliary_loss_sum = 0.0

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            model_inputs = images
            paired_labels = None
            mixing_weight = 1.0
            if (
                training
                and mixup_alpha > 0.0
                and float(torch.rand(1).item()) < mixup_probability
            ):
                model_inputs, paired_labels, mixing_weight = mixup_batch(
                    images, labels, alpha=mixup_alpha
                )

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(model_inputs)
                if paired_labels is None:
                    loss = criterion(logits, labels)
                else:
                    loss = mixing_weight * criterion(logits, labels) + (
                        1.0 - mixing_weight
                    ) * criterion(logits, paired_labels)
                auxiliary_loss = None
                if training and auxiliary_weight > 0.0:
                    auxiliary_method = getattr(
                        model, "masked_patch_prediction_loss", None
                    )
                    if auxiliary_method is None:
                        raise ValueError(
                            "Auxiliary loss is enabled but the model has no "
                            "masked_patch_prediction_loss method"
                        )
                    auxiliary_loss = auxiliary_method(
                        images, mask_ratio=auxiliary_mask_ratio
                    )
                    loss = loss + auxiliary_weight * auxiliary_loss

            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                if ema is not None:
                    ema.update(model)

            accumulator.update(logits, labels, loss)
            if auxiliary_loss is not None:
                auxiliary_loss_sum += float(auxiliary_loss.detach().item()) * labels.numel()
    result = accumulator.compute(include_confusion=False)
    if training and auxiliary_weight > 0.0:
        result["auxiliary_loss"] = auxiliary_loss_sum / accumulator.sample_count
    return result


def model_num_classes(model: nn.Module) -> int:
    """Infer output count without a dummy forward pass."""
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return module.out_features
    raise ValueError("Model does not contain a Linear classification layer")


@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    include_confusion: bool = True,
) -> dict[str, Any]:
    model.eval()
    accumulator = ClassificationAccumulator(num_classes=model_num_classes(model))
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        accumulator.update(logits, labels, loss)
    return accumulator.compute(include_confusion=include_confusion)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: nn.Module,
    device: torch.device,
    config: dict[str, Any],
    class_names: list[str],
    output_dir: Path,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    training_config = config["training"]
    epochs = int(training_config["epochs"])
    patience = int(training_config.get("early_stopping_patience", epochs))
    grad_clip = training_config.get("grad_clip_norm")
    amp = bool(training_config.get("amp", True))
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    auxiliary_config = config.get("auxiliary", {})
    auxiliary_weight = (
        float(auxiliary_config.get("weight", 0.0))
        if auxiliary_config.get("enabled", False)
        else 0.0
    )
    auxiliary_mask_ratio = float(auxiliary_config.get("mask_ratio", 0.40))
    model.to(device)
    ema_config = config.get("ema", {})
    ema = (
        ExponentialMovingAverage(model, decay=float(ema_config.get("decay", 0.995)))
        if ema_config.get("enabled", False)
        else None
    )
    mixup_config = config.get("mixup", {})
    mixup_alpha = (
        float(mixup_config.get("alpha", 0.0))
        if mixup_config.get("enabled", False)
        else 0.0
    )
    mixup_probability = float(mixup_config.get("probability", 1.0))
    if not 0.0 <= mixup_probability <= 1.0:
        raise ValueError("MixUp probability must be between 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best_accuracy = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            grad_clip_norm=float(grad_clip) if grad_clip is not None else None,
            auxiliary_weight=auxiliary_weight,
            auxiliary_mask_ratio=auxiliary_mask_ratio,
            ema=ema,
            mixup_alpha=mixup_alpha,
            mixup_probability=mixup_probability,
        )
        raw_val_metrics = evaluate_classifier(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            include_confusion=False,
        )
        val_metrics = (
            evaluate_classifier(
                model=ema.module,
                loader=val_loader,
                criterion=criterion,
                device=device,
                include_confusion=False,
            )
            if ema is not None
            else raw_val_metrics
        )
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - start_time,
        }
        if ema is not None:
            row.update(
                {
                    "raw_val_loss": raw_val_metrics["loss"],
                    "raw_val_accuracy": raw_val_metrics["accuracy"],
                    "raw_val_macro_f1": raw_val_metrics["macro_f1"],
                }
            )
        if "auxiliary_loss" in train_metrics:
            row["train_auxiliary_loss"] = train_metrics["auxiliary_loss"]
        history.append(row)
        save_history_csv(history, output_dir / "history.csv")
        save_json(history, output_dir / "history.json")

        improved = val_metrics["accuracy"] > best_accuracy
        if improved:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "epoch": epoch,
                "model_state": copy.deepcopy(
                    ema.module.state_dict() if ema is not None else model.state_dict()
                ),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
                "class_names": class_names,
                "weight_source": "ema" if ema is not None else "raw",
            }
            temporary_path = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(checkpoint_path)
        else:
            epochs_without_improvement += 1

        validation_label = "EMA val" if ema is not None else "val"
        raw_label = (
            f"raw val acc {raw_val_metrics['accuracy']:.4f} | "
            if ema is not None
            else ""
        )
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
            f"{raw_label}{validation_label} loss {val_metrics['loss']:.4f} "
            f"acc {val_metrics['accuracy']:.4f} macro-F1 {val_metrics['macro_f1']:.4f} | "
            f"best {best_accuracy:.4f} @ {best_epoch}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    best_metrics = evaluate_classifier(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        include_confusion=True,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "epochs_completed": len(history),
        "elapsed_seconds": time.time() - start_time,
        "checkpoint": str(checkpoint_path),
        "weight_source": checkpoint.get("weight_source", "raw"),
    }
    save_json(summary, output_dir / "summary.json")
    return model, history, summary
