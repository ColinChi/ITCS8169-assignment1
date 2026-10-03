#!/usr/bin/env python3
"""Export a compact inference-only checkpoint suitable for repository release."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    source = resolve(args.source)
    output = resolve(args.output)
    checkpoint = torch.load(source, map_location="cpu")
    state = {
        key: value.detach().to(dtype=torch.float16)
        if torch.is_floating_point(value)
        else value.detach()
        for key, value in checkpoint["model_state"].items()
    }
    config = copy.deepcopy(checkpoint["config"])
    config["model"]["pretrained"] = False
    exported = {
        "epoch": checkpoint["epoch"],
        "model_state": state,
        "val_metrics": checkpoint["val_metrics"],
        "config": config,
        "class_names": checkpoint["class_names"],
        "weight_source": checkpoint.get("weight_source", "raw"),
        "export": {
            "format": "inference-only float16 state dict",
            "source": str(source.relative_to(PROJECT_ROOT)),
            "optimizer_removed": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(exported, temporary)
    temporary.replace(output)
    size_mb = output.stat().st_size / 1_000_000
    print(f"Exported {output} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
