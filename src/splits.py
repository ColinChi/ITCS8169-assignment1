"""Deterministic, duplicate-aware dataset split utilities.

This module intentionally uses only the Python standard library so that the
train/validation manifest can be generated before PyTorch is installed.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_records(train_root: Path, project_root: Path) -> list[dict[str, Any]]:
    """Return a stable manifest of images under ``train_root/class_name``."""
    train_root = train_root.resolve()
    project_root = project_root.resolve()
    if not train_root.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {train_root}")

    records: list[dict[str, Any]] = []
    class_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found under {train_root}")

    for class_dir in class_dirs:
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                relative_path = path.resolve().relative_to(project_root).as_posix()
            except ValueError as exc:
                raise ValueError(f"Image {path} is outside project root {project_root}") from exc
            records.append(
                {
                    "path": relative_path,
                    "class_name": class_dir.name,
                    "sha256": sha256_file(path),
                }
            )

    if not records:
        raise ValueError(f"No supported image files found under {train_root}")
    return records


def _choose_groups_for_target(
    groups: list[list[dict[str, Any]]], target: int, seed: int
) -> set[int]:
    """Choose whole duplicate groups whose total size is closest to target.

    Dynamic programming finds an exact target whenever one exists. The seeded
    shuffle prevents filename ordering from systematically biasing validation.
    """
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    reachable: dict[int, tuple[int, ...]] = {0: ()}

    for group_index in order:
        group_size = len(groups[group_index])
        for subtotal, chosen in sorted(list(reachable.items()), reverse=True):
            candidate = subtotal + group_size
            if candidate <= target and candidate not in reachable:
                reachable[candidate] = chosen + (group_index,)

    best_total = min(reachable, key=lambda value: (abs(target - value), -value))
    return set(reachable[best_total])


def create_split_manifest(
    train_root: Path,
    output_path: Path,
    project_root: Path,
    val_fraction: float = 0.20,
    seed: int = 0,
) -> dict[str, Any]:
    """Create and atomically save a class-stratified, duplicate-aware split."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")

    records = discover_records(train_root, project_root)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hash_classes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_class[record["class_name"]].append(record)
        hash_classes[record["sha256"]].add(record["class_name"])

    cross_class = {
        digest: sorted(class_names)
        for digest, class_names in hash_classes.items()
        if len(class_names) > 1
    }
    if cross_class:
        raise ValueError(f"Identical image content has conflicting labels: {cross_class}")

    train_paths: list[str] = []
    val_paths: list[str] = []
    duplicate_groups: list[list[str]] = []
    class_counts: dict[str, dict[str, int]] = {}

    for class_name in sorted(by_class):
        hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in by_class[class_name]:
            hash_groups[record["sha256"]].append(record)
        groups = list(hash_groups.values())
        groups.sort(key=lambda group: group[0]["path"])

        for group in groups:
            if len(group) > 1:
                duplicate_groups.append(sorted(item["path"] for item in group))

        target = round(len(by_class[class_name]) * val_fraction)
        class_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{class_name}".encode()).digest()[:8], "big"
        )
        selected = _choose_groups_for_target(groups, target, class_seed)

        class_train: list[str] = []
        class_val: list[str] = []
        for group_index, group in enumerate(groups):
            destination = class_val if group_index in selected else class_train
            destination.extend(item["path"] for item in group)

        train_paths.extend(sorted(class_train))
        val_paths.extend(sorted(class_val))
        class_counts[class_name] = {
            "total": len(by_class[class_name]),
            "train": len(class_train),
            "val": len(class_val),
        }

    fingerprint_source = [
        (record["path"], record["class_name"], record["sha256"])
        for record in sorted(records, key=lambda item: item["path"])
    ]
    dataset_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_source, separators=(",", ":")).encode()
    ).hexdigest()

    class_names = sorted(by_class)
    manifest: dict[str, Any] = {
        "version": 1,
        "seed": seed,
        "val_fraction": val_fraction,
        "dataset_fingerprint": dataset_fingerprint,
        "class_to_idx": {name: index for index, name in enumerate(class_names)},
        "class_counts": class_counts,
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "train_paths": sorted(train_paths),
        "val_paths": sorted(val_paths),
        "duplicate_groups": sorted(duplicate_groups),
    }

    validate_manifest(manifest, project_root=project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    return manifest


def validate_manifest(manifest: dict[str, Any], project_root: Path | None = None) -> None:
    train_paths = manifest.get("train_paths", [])
    val_paths = manifest.get("val_paths", [])
    if len(train_paths) != len(set(train_paths)):
        raise ValueError("Duplicate paths found within the training split")
    if len(val_paths) != len(set(val_paths)):
        raise ValueError("Duplicate paths found within the validation split")
    overlap = set(train_paths) & set(val_paths)
    if overlap:
        raise ValueError(f"Train/validation overlap detected: {sorted(overlap)[:5]}")
    if len(train_paths) != manifest.get("train_count"):
        raise ValueError("train_count does not match train_paths")
    if len(val_paths) != manifest.get("val_count"):
        raise ValueError("val_count does not match val_paths")

    assignments = {path: "train" for path in train_paths}
    assignments.update({path: "val" for path in val_paths})
    for group in manifest.get("duplicate_groups", []):
        sides = {assignments.get(path) for path in group}
        if len(sides) != 1 or None in sides:
            raise ValueError(f"Duplicate group crosses or is missing from splits: {group}")

    expected_class_counts = Counter()
    for path in train_paths + val_paths:
        path_parts = Path(path).parts
        if len(path_parts) < 3:
            raise ValueError(f"Unexpected image path in manifest: {path}")
        expected_class_counts[path_parts[-2]] += 1
        if project_root is not None and not (project_root / path).is_file():
            raise FileNotFoundError(f"Manifest image is missing: {project_root / path}")

    for class_name, counts in manifest.get("class_counts", {}).items():
        if expected_class_counts[class_name] != counts["total"]:
            raise ValueError(f"Class count mismatch for {class_name}")


def load_split_manifest(path: Path, project_root: Path | None = None) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest, project_root=project_root)
    return manifest


def relative_paths(paths: Iterable[Path], project_root: Path) -> list[str]:
    return [path.resolve().relative_to(project_root.resolve()).as_posix() for path in paths]

