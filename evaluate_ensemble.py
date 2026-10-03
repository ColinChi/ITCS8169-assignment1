#!/usr/bin/env python3
"""Evaluate one or more CNN checkpoints with probability averaging and optional TTA."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.data import build_dataloaders
from src.metrics import ClassificationAccumulator
from src.models import build_model
from src.utils import resolve_device, save_json


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--horizontal-flip-tta", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _resolved(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_models(
    paths: list[Path], device: torch.device, num_workers: int | None
) -> tuple[list[torch.nn.Module], dict[str, Any], list[str]]:
    models: list[torch.nn.Module] = []
    reference_config: dict[str, Any] | None = None
    reference_classes: list[str] | None = None
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu")
        config = copy.deepcopy(checkpoint["config"])
        config.setdefault("runtime", {})["resolved_device"] = device.type
        if num_workers is not None:
            config["data"]["num_workers"] = num_workers
        classes = checkpoint["class_names"]

        if reference_config is None:
            # Keep an untouched comparison copy; model construction below disables
            # pretrained downloads in its own mutable config.
            reference_config = copy.deepcopy(config)
            reference_classes = classes
        else:
            comparable_keys = ("model", "data")
            for key in comparable_keys:
                left = copy.deepcopy(reference_config[key])
                right = copy.deepcopy(config[key])
                if key == "data":
                    # Loader details and training seed do not affect deterministic evaluation.
                    left.pop("num_workers", None)
                    right.pop("num_workers", None)
                    left.pop("batch_size", None)
                    right.pop("batch_size", None)
                if left != right:
                    raise ValueError(f"Incompatible checkpoint {key} settings: {path}")
            if classes != reference_classes:
                raise ValueError(f"Incompatible class order in checkpoint: {path}")

        config["model"]["pretrained"] = False
        model = build_model(config, num_classes=len(classes)).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)

    if reference_config is None or reference_classes is None:
        raise ValueError("At least one checkpoint is required")
    return models, reference_config, reference_classes


@torch.inference_mode()
def evaluate(
    models: list[torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    horizontal_flip_tta: bool,
    num_classes: int,
) -> dict[str, Any]:
    accumulator = ClassificationAccumulator(num_classes=num_classes)
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        probability_sum = torch.zeros(
            (inputs.shape[0], num_classes), device=device, dtype=torch.float32
        )
        prediction_count = 0
        for model in models:
            probability_sum += F.softmax(model(inputs), dim=1)
            prediction_count += 1
            if horizontal_flip_tta:
                probability_sum += F.softmax(model(torch.flip(inputs, dims=(3,))), dim=1)
                prediction_count += 1
        probabilities = probability_sum / prediction_count
        log_probabilities = probabilities.clamp_min(1e-12).log()
        loss = F.nll_loss(log_probabilities, targets)
        accumulator.update(log_probabilities, targets, loss)
    return accumulator.compute(include_confusion=True)


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.confirm_final_test:
        raise SystemExit(
            "Refusing test evaluation without --confirm-final-test. "
            "Select the recipe using validation results first."
        )

    device = resolve_device(args.device)
    checkpoint_paths = [_resolved(path) for path in args.checkpoint]
    models, config, class_names = _load_models(
        checkpoint_paths, device=device, num_workers=args.num_workers
    )
    data = build_dataloaders(config, PROJECT_ROOT)
    if data.class_names != class_names:
        raise ValueError("Dataset class order differs from checkpoint class order")
    loader = data.val_loader if args.split == "validation" else data.test_loader
    metrics = evaluate(
        models=models,
        loader=loader,
        device=device,
        horizontal_flip_tta=args.horizontal_flip_tta,
        num_classes=len(class_names),
    )
    metrics.update(
        {
            "split": args.split,
            "class_names": class_names,
            "checkpoints": [
                str(path.relative_to(PROJECT_ROOT)) for path in checkpoint_paths
            ],
            "ensemble_size": len(models),
            "horizontal_flip_tta": args.horizontal_flip_tta,
            "aggregation": "arithmetic mean of softmax probabilities",
        }
    )
    output_path = _resolved(args.output)
    save_json(metrics, output_path)
    print(
        f"{args.split.title()} accuracy={metrics['accuracy']:.4f} "
        f"macro-F1={metrics['macro_f1']:.4f} NLL={metrics['loss']:.4f}"
    )
    print(
        f"Models={len(models)} flip-TTA={args.horizontal_flip_tta}; "
        f"saved to {output_path}"
    )


if __name__ == "__main__":
    main()
