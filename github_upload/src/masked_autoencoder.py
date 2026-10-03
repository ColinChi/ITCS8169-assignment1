"""CNN-compatible masked image reconstruction for self-supervised adaptation.

This is deliberately convolutional so that the transferable encoder and final
classifier comply with the assignment's CNN-only primary-model restriction.
Unlike the original ViT MAE, masked blocks are zeroed in image space and passed
through a ResNet18 encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_block_mask(
    images: torch.Tensor,
    patch_size: int,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero-masked images and a Bx1xHxW binary pixel mask."""
    if images.ndim != 4:
        raise ValueError(f"Expected BCHW images, got shape {tuple(images.shape)}")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be strictly between 0 and 1")
    batch_size, _channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Image size {(height, width)} must be divisible by patch_size={patch_size}"
        )

    grid_height = height // patch_size
    grid_width = width // patch_size
    patch_count = grid_height * grid_width
    masked_count = max(1, min(patch_count - 1, round(patch_count * mask_ratio)))
    noise = torch.rand(batch_size, patch_count, device=images.device)
    masked_indices = noise.argsort(dim=1)[:, :masked_count]
    patch_mask = torch.zeros(
        batch_size, patch_count, device=images.device, dtype=images.dtype
    )
    patch_mask.scatter_(1, masked_indices, 1.0)
    pixel_mask = patch_mask.reshape(batch_size, 1, grid_height, grid_width)
    pixel_mask = pixel_mask.repeat_interleave(patch_size, dim=2).repeat_interleave(
        patch_size, dim=3
    )
    masked_images = images * (1.0 - pixel_mask)
    return masked_images, pixel_mask


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
) -> torch.Tensor:
    if reconstruction.shape != target.shape:
        raise ValueError(
            f"Reconstruction shape {tuple(reconstruction.shape)} != target {tuple(target.shape)}"
        )
    if pixel_mask.shape != (target.shape[0], 1, target.shape[2], target.shape[3]):
        raise ValueError(f"Unexpected mask shape: {tuple(pixel_mask.shape)}")
    squared_error = (reconstruction - target).pow(2) * pixel_mask
    denominator = pixel_mask.sum() * target.shape[1]
    return squared_error.sum() / denominator.clamp_min(1.0)


class ReconstructionDecoder(nn.Module):
    """Upsample a stride-32 ResNet feature map back to RGB pixels."""

    def __init__(self, in_channels: int = 512, out_channels: int = 3) -> None:
        super().__init__()
        channels = (in_channels, 256, 128, 64, 32, 16)
        blocks: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.ConvTranspose2d(
                        input_channels,
                        output_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.GELU(),
                ]
            )
        blocks.append(nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1))
        self.network = nn.Sequential(*blocks)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ConvMaskedAutoencoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        patch_size: int = 16,
        mask_ratio: float = 0.50,
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.encoder = resnet18(weights=weights)
        self.decoder = ReconstructionDecoder(in_channels=512, out_channels=3)
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def encode_spatial(self, images: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        features = encoder.conv1(images)
        features = encoder.bn1(features)
        features = encoder.relu(features)
        features = encoder.maxpool(features)
        features = encoder.layer1(features)
        features = encoder.layer2(features)
        features = encoder.layer3(features)
        return encoder.layer4(features)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masked_images, pixel_mask = random_block_mask(
            images, patch_size=self.patch_size, mask_ratio=self.mask_ratio
        )
        features = self.encode_spatial(masked_images)
        reconstruction = self.decoder(features)
        if reconstruction.shape[-2:] != images.shape[-2:]:
            reconstruction = F.interpolate(
                reconstruction,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction, pixel_mask, masked_images

    def transferable_encoder_state(self) -> dict[str, torch.Tensor]:
        """Return ResNet weights excluding the ImageNet classifier head."""
        return {
            key: value.detach().cpu()
            for key, value in self.encoder.state_dict().items()
            if not key.startswith("fc.")
        }

