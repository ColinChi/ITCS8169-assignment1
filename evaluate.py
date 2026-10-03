#!/usr/bin/env python3
"""Evaluate a saved checkpoint; test evaluation requires an explicit confirmation flag."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn

from src.data import build_dataloaders
from src.engine import evaluate_classifier
from src.models import build_model
from src.utils import resolve_device, save_json


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.confirm_final_test:
        raise SystemExit(
            "Refusing test evaluation without --confirm-final-test. "
            "Select the model using validation results first."
        )

    checkpoint_path = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else PROJECT_ROOT / args.checkpoint
    )
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = copy.deepcopy(checkpoint["config"])
    config.setdefault("runtime", {})["resolved_device"] = device.type
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    data = build_dataloaders(config, PROJECT_ROOT)
    expected_names = checkpoint["class_names"]
    if data.class_names != expected_names:
        raise ValueError(
            f"Checkpoint classes {expected_names} differ from dataset {data.class_names}"
        )

    # The checkpoint contains all weights, so do not download pretrained weights again.
    config["model"]["pretrained"] = False
    model = build_model(config, num_classes=len(data.class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["training"].get("label_smoothing", 0.0))
    )
    loader = data.val_loader if args.split == "validation" else data.test_loader
    metrics = evaluate_classifier(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        include_confusion=True,
    )
    metrics["split"] = args.split
    metrics["class_names"] = data.class_names
    output_path = checkpoint_path.parent / f"evaluation_{args.split}.json"
    save_json(metrics, output_path)
    print(
        f"{args.split.title()} loss={metrics['loss']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}"
    )
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()

