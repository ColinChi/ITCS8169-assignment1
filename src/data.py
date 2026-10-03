"""Dataset, transformation, and DataLoader construction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .splits import load_split_manifest


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvertRGB:
    """Pickle-safe PIL conversion used by multi-worker DataLoaders."""

    def __call__(self, image: Any) -> Any:
        return image.convert("RGB")


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    class_to_idx: dict[str, int]
    train_size: int
    val_size: int
    test_size: int


def _channel_transform(input_mode: str) -> Any:
    if input_mode == "rgb":
        return ConvertRGB()
    if input_mode == "gray3":
        return transforms.Grayscale(num_output_channels=3)
    if input_mode == "gray1":
        return transforms.Grayscale(num_output_channels=1)
    raise ValueError(f"Unsupported input_mode={input_mode!r}; use rgb, gray3, or gray1")


def build_transforms(
    image_size: int,
    input_mode: str,
    augment: bool,
    profile: str = "standard",
    augmentation_policy: str = "standard",
) -> tuple[Any, Any]:
    """Build separate stochastic training and deterministic evaluation transforms."""
    channel_transform = _channel_transform(input_mode)
    if profile == "starter":
        if input_mode != "gray1":
            raise ValueError("The starter profile must use input_mode='gray1'")
        transform = transforms.Compose(
            [
                channel_transform,
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )
        return transform, transform
    if profile != "standard":
        raise ValueError(f"Unknown transform profile: {profile}")

    if input_mode == "gray1":
        mean, std = (0.5,), (0.5,)
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    train_steps: list[Any] = [channel_transform]
    if augment:
        if augmentation_policy == "standard":
            crop_scale = (0.65, 1.0)
            crop_ratio = (0.75, 1.3333)
            rotation = 8
            jitter = (0.15, 0.15, 0.10)
            use_randaugment = False
            erasing_probability = 0.0
        elif augmentation_policy == "conservative":
            # Scene labels depend on global layout, so retain more of each image.
            crop_scale = (0.82, 1.0)
            crop_ratio = (0.85, 1.1765)
            rotation = 5
            jitter = (0.10, 0.10, 0.05)
            use_randaugment = False
            erasing_probability = 0.10
        elif augmentation_policy == "randaugment":
            crop_scale = (0.78, 1.0)
            crop_ratio = (0.80, 1.25)
            rotation = 0
            jitter = (0.0, 0.0, 0.0)
            use_randaugment = True
            erasing_probability = 0.15
        else:
            raise ValueError(f"Unknown augmentation_policy: {augmentation_policy}")

        train_steps.extend(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=crop_scale, ratio=crop_ratio, antialias=True
                ),
                transforms.RandomHorizontalFlip(),
            ]
        )
        if rotation:
            train_steps.append(transforms.RandomRotation(degrees=rotation))
        if input_mode == "rgb" and any(jitter):
            train_steps.append(
                transforms.ColorJitter(
                    brightness=jitter[0], contrast=jitter[1], saturation=jitter[2]
                )
            )
        if use_randaugment:
            train_steps.append(transforms.RandAugment(num_ops=2, magnitude=7))
    else:
        train_steps.extend(
            [
                transforms.Resize(round(image_size * 1.14), antialias=True),
                transforms.CenterCrop(image_size),
            ]
        )
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if augment and erasing_probability:
        train_steps.append(
            transforms.RandomErasing(
                p=erasing_probability,
                scale=(0.02, 0.12),
                ratio=(0.5, 2.0),
                value="random",
            )
        )

    eval_transform = transforms.Compose(
        [
            channel_transform,
            transforms.Resize(round(image_size * 1.14), antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(train_steps), eval_transform


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _indices_from_manifest(
    dataset: datasets.ImageFolder,
    paths: list[str],
    project_root: Path,
) -> list[int]:
    path_to_index = {
        Path(sample_path).resolve(): index
        for index, (sample_path, _target) in enumerate(dataset.samples)
    }
    indices: list[int] = []
    missing: list[str] = []
    for relative_path in paths:
        resolved = (project_root / relative_path).resolve()
        if resolved not in path_to_index:
            missing.append(relative_path)
        else:
            indices.append(path_to_index[resolved])
    if missing:
        raise ValueError(f"Manifest paths absent from ImageFolder: {missing[:5]}")
    return indices


def build_dataloaders(config: dict[str, Any], project_root: Path) -> DataBundle:
    data_config = config["data"]
    train_root = (project_root / data_config["train_dir"]).resolve()
    test_root = (project_root / data_config["test_dir"]).resolve()
    split_path = (project_root / data_config["split_file"]).resolve()
    manifest = load_split_manifest(split_path, project_root=project_root)

    train_transform, eval_transform = build_transforms(
        image_size=int(data_config["image_size"]),
        input_mode=data_config["input_mode"],
        augment=bool(data_config.get("augment", True)),
        profile=data_config.get("transform_profile", "standard"),
        augmentation_policy=data_config.get("augmentation_policy", "standard"),
    )

    train_view = datasets.ImageFolder(train_root, transform=train_transform)
    val_view = datasets.ImageFolder(train_root, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_root, transform=eval_transform)

    expected_mapping = manifest["class_to_idx"]
    if train_view.class_to_idx != expected_mapping:
        raise ValueError(
            f"Training class mapping changed: {train_view.class_to_idx} != {expected_mapping}"
        )
    if test_dataset.class_to_idx != expected_mapping:
        raise ValueError(
            f"Test class mapping differs from training: {test_dataset.class_to_idx}"
        )

    discovered_paths = {
        Path(path).resolve() for path, _target in train_view.samples
    }
    manifest_paths = {
        (project_root / path).resolve()
        for path in manifest["train_paths"] + manifest["val_paths"]
    }
    if discovered_paths != manifest_paths:
        extra = sorted(str(path) for path in discovered_paths - manifest_paths)
        missing = sorted(str(path) for path in manifest_paths - discovered_paths)
        raise ValueError(
            "Dataset contents differ from the fixed split manifest. "
            f"Extra={extra[:3]} Missing={missing[:3]}. Regenerate the split intentionally."
        )

    train_indices = _indices_from_manifest(
        train_view, manifest["train_paths"], project_root
    )
    val_indices = _indices_from_manifest(val_view, manifest["val_paths"], project_root)
    train_dataset = Subset(train_view, train_indices)
    val_dataset = Subset(val_view, val_indices)

    seed = int(config["training"]["seed"])
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(data_config["batch_size"])
    num_workers = int(data_config.get("num_workers", 2))
    use_cuda = config.get("runtime", {}).get("resolved_device") == "cuda"
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _seed_worker,
    }

    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=train_view.classes,
        class_to_idx=train_view.class_to_idx,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
    )
