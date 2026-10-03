#!/usr/bin/env python3
"""Generate the fixed train/validation split without requiring PyTorch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.splits import create_split_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--train-root", type=Path, default=Path("data/train"))
    parser.add_argument(
        "--output", type=Path, default=Path("splits/seed0_val20.json")
    )
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_from_project(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    train_root = resolve_from_project(args.train_root, project_root)
    output_path = resolve_from_project(args.output, project_root)
    manifest = create_split_manifest(
        train_root=train_root,
        output_path=output_path,
        project_root=project_root,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"Wrote split manifest: {output_path}")
    print(f"Train: {manifest['train_count']} | Validation: {manifest['val_count']}")
    print(f"Classes: {len(manifest['class_to_idx'])}")
    print(f"Duplicate groups kept intact: {len(manifest['duplicate_groups'])}")
    for class_name, counts in manifest["class_counts"].items():
        print(
            f"  {class_name:14s} total={counts['total']:3d} "
            f"train={counts['train']:3d} val={counts['val']:2d}"
        )


if __name__ == "__main__":
    main()

