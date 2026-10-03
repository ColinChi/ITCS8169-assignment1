#!/usr/bin/env python3
"""Self-supervised convolutional masked-reconstruction pretraining."""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from src.data import IMAGENET_MEAN, IMAGENET_STD, build_dataloaders
from src.masked_autoencoder import ConvMaskedAutoencoder, masked_reconstruction_loss
from src.utils import (
    environment_summary,
    load_json,
    resolve_device,
    save_history_csv,
    save_json,
    set_random_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    return config


def save_reconstruction_figure(
    model: ConvMaskedAutoencoder,
    loader: Any,
    device: torch.device,
    output_path: Path,
    examples: int = 4,
) -> None:
    model.eval()
    images, _labels = next(iter(loader))
    images = images[:examples].to(device)
    with torch.inference_mode():
        reconstruction, mask, masked_images = model(images)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def display(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor * std + mean).clamp(0.0, 1.0).cpu()

    original = display(images)
    masked = display(masked_images)
    reconstructed = display(reconstruction)
    # Preserve known pixels to make the quality of predicted masked regions easy to inspect.
    composite = original * (1.0 - mask.cpu()) + reconstructed * mask.cpu()

    figure, axes = plt.subplots(examples, 4, figsize=(12, 3 * examples))
    columns = ("Original", "Masked input", "Raw reconstruction", "Composite")
    rows = (original, masked, reconstructed, composite)
    for row_index in range(examples):
        for column_index, batch in enumerate(rows):
            axes[row_index, column_index].imshow(batch[row_index].permute(1, 2, 0))
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(columns[column_index])
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = apply_overrides(load_json(config_path), args)
    device = resolve_device(args.device)
    config.setdefault("runtime", {})["resolved_device"] = device.type
    training_config = config["training"]
    model_config = config["model"]
    seed = int(training_config["seed"])
    set_random_seed(seed, deterministic=bool(training_config.get("deterministic", False)))

    if config["data"]["input_mode"] != "rgb":
        raise ValueError("Masked reconstruction currently requires RGB input")
    data = build_dataloaders(config, PROJECT_ROOT)
    model = ConvMaskedAutoencoder(
        pretrained=bool(model_config.get("pretrained", True)),
        patch_size=int(model_config["patch_size"]),
        mask_ratio=float(model_config["mask_ratio"]),
    ).to(device)

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": float(optimizer_config["encoder_lr"]),
                "name": "encoder",
            },
            {
                "params": model.decoder.parameters(),
                "lr": float(optimizer_config["decoder_lr"]),
                "name": "decoder",
            },
        ],
        weight_decay=float(optimizer_config.get("weight_decay", 0.0001)),
    )
    epochs = int(training_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(config["scheduler"].get("min_lr", 1e-6))
    )
    use_amp = bool(training_config.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = PROJECT_ROOT / "runs" / config["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.json")
    save_json(environment_summary(), output_dir / "environment.json")
    checkpoint_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    start_time = time.time()

    print(f"Run: {config['run_name']}")
    print(f"Device: {device} | Self-supervised train images: {data.train_size}")
    print("Validation and test images are not used during masked reconstruction.")
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for images, _labels in data.train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                reconstruction, pixel_mask, _masked_images = model(images)
                loss = masked_reconstruction_loss(reconstruction, images, pixel_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config.get("grad_clip_norm", 5.0))
            )
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().item()) * images.shape[0]
            sample_count += images.shape[0]

        epoch_loss = loss_sum / sample_count
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_reconstruction_loss": epoch_loss,
            "encoder_learning_rate": optimizer.param_groups[0]["lr"],
            "decoder_learning_rate": optimizer.param_groups[1]["lr"],
            "elapsed_seconds": time.time() - start_time,
        }
        history.append(row)
        save_history_csv(history, output_dir / "history.csv")
        save_json(history, output_dir / "history.json")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "epoch": epoch,
                "train_reconstruction_loss": epoch_loss,
                "model_state": model.state_dict(),
                "encoder_state": model.transferable_encoder_state(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
            }
            temporary_path = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(checkpoint_path)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | masked reconstruction loss "
            f"{epoch_loss:.6f} | best {best_loss:.6f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    save_reconstruction_figure(
        model, data.train_loader, device, output_dir / "reconstructions.png"
    )
    save_json(
        {
            "best_epoch": int(checkpoint["epoch"]),
            "best_train_reconstruction_loss": float(best_loss),
            "train_images": data.train_size,
            "validation_images_used": 0,
            "test_images_used": 0,
            "checkpoint": str(checkpoint_path),
        },
        output_dir / "summary.json",
    )
    print(f"Best reconstruction loss: {best_loss:.6f}")
    print(f"Saved transferable encoder: {checkpoint_path}")


if __name__ == "__main__":
    main()
