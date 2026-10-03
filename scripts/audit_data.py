#!/usr/bin/env python3
"""Audit image counts, modes, dimensions, corruption, and exact duplicates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(data_root: Path) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    modes: dict[str, dict[str, dict[str, int]]] = {}
    sizes: Counter[str] = Counter()
    corrupt: list[dict[str, str]] = []
    ignored: list[str] = []
    hashes: dict[str, list[str]] = defaultdict(list)

    for split in ("train", "test"):
        split_root = data_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing dataset split: {split_root}")
        counts[split] = {}
        modes[split] = {}
        for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            class_count = 0
            class_modes: Counter[str] = Counter()
            for path in sorted(class_dir.iterdir()):
                relative_path = path.relative_to(PROJECT_ROOT).as_posix()
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    ignored.append(relative_path)
                    continue
                try:
                    with Image.open(path) as image:
                        class_modes[image.mode] += 1
                        sizes[f"{image.width}x{image.height}"] += 1
                        image.verify()
                    digest = sha256_file(path)
                    hashes[digest].append(relative_path)
                    class_count += 1
                except Exception as exc:  # report every malformed input in one pass
                    corrupt.append({"path": relative_path, "error": repr(exc)})
            counts[split][class_dir.name] = class_count
            modes[split][class_dir.name] = dict(sorted(class_modes.items()))

    duplicate_groups = sorted(sorted(paths) for paths in hashes.values() if len(paths) > 1)
    cross_split_duplicates = [
        paths
        for paths in duplicate_groups
        if any(path.startswith("data/train/") for path in paths)
        and any(path.startswith("data/test/") for path in paths)
    ]
    return {
        "counts": counts,
        "modes_by_class": modes,
        "image_size_counts": dict(sorted(sizes.items())),
        "valid_image_count": sum(sum(values.values()) for values in counts.values()),
        "corrupt": corrupt,
        "ignored_non_images": sorted(ignored),
        "duplicate_groups": duplicate_groups,
        "cross_split_duplicates": cross_split_duplicates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/data_audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root if args.data_root.is_absolute() else PROJECT_ROOT / args.data_root
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    result = audit(data_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Valid images: {result['valid_image_count']}")
    print(f"Corrupt images: {len(result['corrupt'])}")
    print(f"Duplicate groups: {len(result['duplicate_groups'])}")
    print(f"Cross-split duplicates: {len(result['cross_split_duplicates'])}")
    print(f"Saved audit to: {output_path}")


if __name__ == "__main__":
    main()

