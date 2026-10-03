"""CNN classifiers used by the supervised baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """The deliberately weak starter model, kept for an exact E0 baseline."""

    def __init__(self, num_classes: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 15 * 15, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


class SceneCNN(nn.Module):
    """A compact residual CNN trained from scratch for E1."""

    def __init__(
        self,
        num_classes: int = 16,
        in_channels: int = 3,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.features(self.stem(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


class ConvNeXtMaskedAuxiliary(nn.Module):
    """ConvNeXt classifier with a training-only masked patch prediction head.

    The auxiliary target is the mean normalized RGB value in each stride-32
    feature cell. The head is ignored by ``forward`` and therefore adds no
    classification-time computation.
    """

    def __init__(self, backbone: nn.Module, feature_channels: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.auxiliary_head = nn.Conv2d(feature_channels, 3, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def masked_patch_prediction_loss(
        self, inputs: torch.Tensor, mask_ratio: float = 0.40
    ) -> torch.Tensor:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be strictly between 0 and 1")
        grid_height = inputs.shape[-2] // 32
        grid_width = inputs.shape[-1] // 32
        if grid_height < 1 or grid_width < 1:
            raise ValueError("ConvNeXt auxiliary input must be at least 32x32")

        patch_mask = (
            torch.rand(
                inputs.shape[0], 1, grid_height, grid_width, device=inputs.device
            )
            < mask_ratio
        )
        empty_samples = patch_mask.flatten(1).sum(dim=1) == 0
        patch_mask[empty_samples, :, 0, 0] = True
        pixel_mask = F.interpolate(
            patch_mask.to(dtype=inputs.dtype),
            size=inputs.shape[-2:],
            mode="nearest",
        )
        masked_inputs = inputs * (1.0 - pixel_mask)
        features = self.backbone.features(masked_inputs)
        predictions = self.auxiliary_head(features)
        targets = F.adaptive_avg_pool2d(inputs.detach(), predictions.shape[-2:])
        feature_mask = F.interpolate(
            patch_mask.to(dtype=predictions.dtype),
            size=predictions.shape[-2:],
            mode="nearest",
        )
        squared_error = (predictions - targets).pow(2) * feature_mask
        denominator = feature_mask.sum() * predictions.shape[1]
        return squared_error.sum() / denominator.clamp_min(1.0)


def _build_resnet18(
    num_classes: int, dropout: float, pretrained: bool
) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError("torchvision with ResNet18 support is required") from exc

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "Could not load ImageNet ResNet18 weights. Ensure network access or "
                "pre-populate the torch model cache; do not silently continue with "
                "random initialization."
            ) from exc
        raise
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def _build_resnet50(
    num_classes: int, dropout: float, pretrained: bool
) -> nn.Module:
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def _build_convnext(
    variant: str,
    num_classes: int,
    dropout: float,
    pretrained: bool,
    masked_auxiliary: bool = False,
) -> nn.Module:
    from torchvision.models import (
        ConvNeXt_Base_Weights,
        ConvNeXt_Small_Weights,
        ConvNeXt_Tiny_Weights,
        convnext_base,
        convnext_small,
        convnext_tiny,
    )

    constructors = {
        "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
        "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT),
        "convnext_base": (convnext_base, ConvNeXt_Base_Weights.DEFAULT),
    }
    constructor, default_weights = constructors[variant]
    model = constructor(weights=default_weights if pretrained else None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(dropout), nn.Linear(in_features, num_classes)
    )
    if masked_auxiliary:
        return ConvNeXtMaskedAuxiliary(model, feature_channels=in_features)
    return model


def build_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    model_config = config["model"]
    data_config = config["data"]
    name = model_config["name"].lower()
    input_mode = data_config["input_mode"]
    in_channels = 1 if input_mode == "gray1" else 3

    if name == "tnet":
        if in_channels != 1 or int(data_config["image_size"]) != 64:
            raise ValueError("TNet requires gray1 input at exactly 64x64")
        return TNet(num_classes=num_classes)
    if name == "scene_cnn":
        return SceneCNN(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=float(model_config.get("dropout", 0.30)),
        )
    if name == "resnet18":
        if in_channels != 3:
            raise ValueError("ResNet18 experiments require rgb or gray3 input")
        return _build_resnet18(
            num_classes=num_classes,
            dropout=float(model_config.get("dropout", 0.20)),
            pretrained=bool(model_config.get("pretrained", True)),
        )
    if name == "resnet50":
        if in_channels != 3:
            raise ValueError("ResNet50 experiments require rgb or gray3 input")
        return _build_resnet50(
            num_classes=num_classes,
            dropout=float(model_config.get("dropout", 0.20)),
            pretrained=bool(model_config.get("pretrained", True)),
        )
    if name in {"convnext_tiny", "convnext_small", "convnext_base"}:
        if in_channels != 3:
            raise ValueError("ConvNeXt experiments require rgb or gray3 input")
        return _build_convnext(
            variant=name,
            num_classes=num_classes,
            dropout=float(model_config.get("dropout", 0.20)),
            pretrained=bool(model_config.get("pretrained", True)),
            masked_auxiliary=bool(
                config.get("auxiliary", {}).get("enabled", False)
            ),
        )
    raise ValueError(f"Unknown model name: {name}")


def optimizer_parameter_groups(
    model: nn.Module, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Use a lower learning rate for a pretrained ResNet backbone."""
    model_name = config["model"]["name"].lower()
    optimizer_config = config["optimizer"]
    head_lr = float(optimizer_config["lr"])
    if model_name not in {
        "resnet18",
        "resnet50",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
    }:
        return [{"params": model.parameters(), "lr": head_lr}]

    backbone_lr = float(optimizer_config.get("backbone_lr", head_lr))
    if isinstance(model, ConvNeXtMaskedAuxiliary):
        head_parameters = list(model.backbone.classifier.parameters())
        head_parameters.extend(model.auxiliary_head.parameters())
    else:
        head_module = model.classifier if model_name.startswith("convnext_") else model.fc
        head_parameters = list(head_module.parameters())
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_parameter_ids
    ]
    return [
        {"params": backbone_parameters, "lr": backbone_lr, "name": "backbone"},
        {"params": head_parameters, "lr": head_lr, "name": "head"},
    ]


def initialize_from_encoder_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Load a self-supervised ResNet encoder while leaving the classifier new."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "encoder_state" not in checkpoint:
        raise ValueError(f"Checkpoint has no encoder_state: {checkpoint_path}")
    incompatible = model.load_state_dict(checkpoint["encoder_state"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if not key.startswith("fc.")]
    if unexpected or missing:
        raise ValueError(
            "Self-supervised checkpoint is incompatible with the classifier. "
            f"Missing={missing} Unexpected={unexpected}"
        )
