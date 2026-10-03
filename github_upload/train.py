#!/usr/bin/env python3
"""Train one supervised experiment without touching the test split."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.data import build_dataloaders
from src.engine import fit
from src.models import (
    build_model,
    initialize_from_encoder_checkpoint,
    optimizer_parameter_groups,
)
from src.utils import (
    count_trainable_parameters,
    environment_summary,
    load_json,
    resolve_device,
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
    parser.add_argument("--encoder-checkpoint", type=str)
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if args.epochs is not None:
        resolved["training"]["epochs"] = args.epochs
    if args.num_workers is not None:
        resolved["data"]["num_workers"] = args.num_workers
    if args.run_name is not None:
        resolved["run_name"] = args.run_name
    if args.seed is not None:
        resolved["training"]["seed"] = args.seed
    if args.encoder_checkpoint is not None:
        resolved["model"]["encoder_checkpoint"] = args.encoder_checkpoint
    return resolved


def build_optimizer(
    model: nn.Module, config: dict[str, Any]
) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    groups = optimizer_parameter_groups(model, config)
    name = optimizer_config["name"].lower()
    kwargs = {
        "lr": float(optimizer_config["lr"]),
        "weight_decay": float(optimizer_config.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return torch.optim.Adam(groups, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(groups, **kwargs)
    if name == "sgd":
        return torch.optim.SGD(
            groups, momentum=float(optimizer_config.get("momentum", 0.9)), **kwargs
        )
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any]
) -> Any:
    scheduler_config = config.get("scheduler", {"name": "none"})
    name = scheduler_config.get("name", "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"]["epochs"]),
            eta_min=float(scheduler_config.get("min_lr", 1e-6)),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = apply_overrides(load_json(config_path), args)
    device = resolve_device(args.device)
    config.setdefault("runtime", {})["resolved_device"] = device.type

    seed = int(config["training"]["seed"])
    set_random_seed(seed, deterministic=bool(config["training"].get("deterministic", False)))
    data = build_dataloaders(config, PROJECT_ROOT)
    model = build_model(config, num_classes=len(data.class_names))
    encoder_checkpoint = config["model"].get("encoder_checkpoint")
    if encoder_checkpoint:
        checkpoint_path = PROJECT_ROOT / encoder_checkpoint
        initialize_from_encoder_checkpoint(model, checkpoint_path)
        print(f"Initialized encoder from: {checkpoint_path}")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["training"].get("label_smoothing", 0.0))
    )

    output_dir = PROJECT_ROOT / "runs" / config["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.json")
    save_json(environment_summary(), output_dir / "environment.json")

    print(f"Run: {config['run_name']}")
    print(f"Device: {device}")
    print(
        f"Data: train={data.train_size} validation={data.val_size} "
        f"test={data.test_size} classes={len(data.class_names)}"
    )
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print("The test loader will not be evaluated by train.py.")

    _model, _history, summary = fit(
        model=model,
        train_loader=data.train_loader,
        val_loader=data.val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        class_names=data.class_names,
        output_dir=output_dir,
    )
    print(
        f"Best validation accuracy: {summary['best_validation']['accuracy']:.4f} "
        f"at epoch {summary['best_epoch']}"
    )


if __name__ == "__main__":
    main()
