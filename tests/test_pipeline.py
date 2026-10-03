from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch

from src.data import build_dataloaders
from src.engine import ExponentialMovingAverage, mixup_batch
from src.metrics import ClassificationAccumulator
from src.masked_autoencoder import (
    ConvMaskedAutoencoder,
    masked_reconstruction_loss,
    random_block_mask,
)
from src.models import build_model
from src.utils import load_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineSmokeTests(unittest.TestCase):
    def _config(self, filename: str) -> dict:
        config = load_json(PROJECT_ROOT / "configs" / filename)
        config["data"]["num_workers"] = 0
        config.setdefault("runtime", {})["resolved_device"] = "cpu"
        return config

    def test_data_bundle_matches_fixed_manifest(self) -> None:
        data = build_dataloaders(self._config("e2_resnet18.json"), PROJECT_ROOT)
        self.assertEqual((data.train_size, data.val_size, data.test_size), (1920, 480, 400))
        self.assertEqual(len(data.class_names), 16)
        images, labels = next(iter(data.train_loader))
        self.assertEqual(tuple(images.shape[1:]), (3, 128, 128))
        self.assertTrue(torch.isfinite(images).all())
        self.assertTrue(((0 <= labels) & (labels < 16)).all())

    def test_all_supervised_models_produce_sixteen_logits(self) -> None:
        cases = [
            ("e0_tnet.json", (2, 1, 64, 64)),
            ("e1_scene_cnn.json", (2, 3, 128, 128)),
            ("e2_resnet18.json", (2, 3, 128, 128)),
            ("e7_convnext_tiny_224.json", (1, 3, 224, 224)),
            ("e7_convnext_small_224.json", (1, 3, 224, 224)),
            ("e8_convnext_base_224.json", (1, 3, 224, 224)),
            ("e8_resnet50_224.json", (1, 3, 224, 224)),
        ]
        for filename, shape in cases:
            with self.subTest(config=filename):
                config = self._config(filename)
                config = copy.deepcopy(config)
                config["model"]["pretrained"] = False
                model = build_model(config, num_classes=16).eval()
                with torch.inference_mode():
                    logits = model(torch.randn(shape))
                self.assertEqual(tuple(logits.shape), (shape[0], 16))

    def test_metrics_are_computed_from_confusion_matrix(self) -> None:
        logits = torch.tensor([[4.0, 1.0], [1.0, 4.0], [3.0, 2.0]])
        labels = torch.tensor([0, 1, 1])
        loss = torch.tensor(0.25)
        accumulator = ClassificationAccumulator(num_classes=2)
        accumulator.update(logits, labels, loss)
        metrics = accumulator.compute(include_confusion=True)
        self.assertAlmostEqual(metrics["loss"], 0.25)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [1, 1]])

    def test_exponential_moving_average_updates_parameters(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema = ExponentialMovingAverage(model, decay=0.5)
        with torch.no_grad():
            model.weight.fill_(3.0)
        ema.update(model)
        self.assertTrue(
            torch.equal(ema.module.weight, torch.full_like(ema.module.weight, 2.0))
        )
        self.assertFalse(ema.module.weight.requires_grad)

    def test_mixup_batch_preserves_shapes_and_uses_dominant_target(self) -> None:
        images = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2, 1)
        labels = torch.tensor([0, 1, 2, 3])
        mixed, paired_labels, mixing_weight = mixup_batch(images, labels, alpha=0.2)
        self.assertEqual(mixed.shape, images.shape)
        self.assertEqual(paired_labels.shape, labels.shape)
        self.assertGreaterEqual(mixing_weight, 0.5)
        self.assertLessEqual(mixing_weight, 1.0)
        self.assertTrue(torch.isfinite(mixed).all())

    def test_convnext_masked_auxiliary_loss(self) -> None:
        config = self._config("e7_convnext_small_224.json")
        config["model"]["pretrained"] = False
        config["auxiliary"] = {
            "enabled": True,
            "type": "masked_patch_prediction",
            "mask_ratio": 0.40,
            "weight": 0.20,
        }
        model = build_model(config, num_classes=16).eval()
        images = torch.randn(1, 3, 224, 224)
        with torch.inference_mode():
            logits = model(images)
            auxiliary_loss = model.masked_patch_prediction_loss(images, mask_ratio=0.40)
        self.assertEqual(tuple(logits.shape), (1, 16))
        self.assertTrue(torch.isfinite(auxiliary_loss))
        self.assertGreater(float(auxiliary_loss), 0.0)

    def test_convolutional_masked_autoencoder_shapes_and_loss(self) -> None:
        images = torch.randn(2, 3, 128, 128)
        masked, mask = random_block_mask(images, patch_size=16, mask_ratio=0.5)
        self.assertEqual(masked.shape, images.shape)
        self.assertEqual(tuple(mask.shape), (2, 1, 128, 128))
        self.assertAlmostEqual(float(mask.mean()), 0.5)

        model = ConvMaskedAutoencoder(
            pretrained=False, patch_size=16, mask_ratio=0.5
        ).eval()
        with torch.inference_mode():
            reconstruction, model_mask, masked_images = model(images)
            loss = masked_reconstruction_loss(reconstruction, images, model_mask)
        self.assertEqual(reconstruction.shape, images.shape)
        self.assertEqual(masked_images.shape, images.shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
