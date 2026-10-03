"""Dependency-light classification metrics."""

from __future__ import annotations

from typing import Any

import torch


class ClassificationAccumulator:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.loss_sum = 0.0
        self.sample_count = 0
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(
        self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor
    ) -> None:
        batch_size = targets.numel()
        self.loss_sum += float(loss.detach().item()) * batch_size
        self.sample_count += batch_size
        predictions = logits.detach().argmax(dim=1).to("cpu")
        targets_cpu = targets.detach().to("cpu")
        encoded = targets_cpu * self.num_classes + predictions
        counts = torch.bincount(encoded, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self, include_confusion: bool = False) -> dict[str, Any]:
        if self.sample_count == 0:
            raise ValueError("Cannot compute metrics with zero samples")
        matrix = self.confusion.to(torch.float64)
        true_positive = matrix.diag()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        precision = true_positive / predicted.clamp_min(1.0)
        recall = true_positive / support.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
        result: dict[str, Any] = {
            "loss": self.loss_sum / self.sample_count,
            "accuracy": float(true_positive.sum() / matrix.sum()),
            "macro_f1": float(f1.mean()),
            "per_class_accuracy": [float(value) for value in recall],
        }
        if include_confusion:
            result["confusion_matrix"] = self.confusion.tolist()
        return result

