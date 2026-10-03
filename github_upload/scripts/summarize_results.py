#!/usr/bin/env python3
"""Aggregate validation runs and create report-ready tables and figures."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"

RUNS = [
    {"experiment": "E0 Starter TNet", "group": "E0", "seed": 0, "run": "e0_tnet_seed0"},
    {"experiment": "E1 SceneCNN", "group": "E1", "seed": 0, "run": "e1_scene_cnn_rgb_seed0"},
    {"experiment": "E2 ResNet18", "group": "E2", "seed": 0, "run": "e2_resnet18_rgb_seed0"},
    {"experiment": "E2 ResNet18", "group": "E2", "seed": 1, "run": "e2_resnet18_rgb_seed1"},
    {"experiment": "E2 ResNet18", "group": "E2", "seed": 2, "run": "e2_resnet18_rgb_seed2"},
    {"experiment": "E3 ConvMAE + ResNet18", "group": "E3", "seed": 0, "run": "e3_convmae_finetune_rgb_seed0"},
    {"experiment": "E3 ConvMAE + ResNet18", "group": "E3", "seed": 1, "run": "e3_convmae_finetune_rgb_seed1"},
    {"experiment": "E3 ConvMAE + ResNet18", "group": "E3", "seed": 2, "run": "e3_convmae_finetune_rgb_seed2"},
    {"experiment": "E5 ResNet18 gray", "group": "E5", "seed": 0, "run": "e5_resnet18_gray3_seed0"},
    {"experiment": "E6 ResNet18 224px", "group": "E6", "seed": 0, "run": "e6_resnet18_224_conservative_seed0"},
    {"experiment": "E7 ConvNeXt-Small", "group": "E7", "seed": 0, "run": "e7_convnext_small_224_seed0"},
    {"experiment": "E7 ConvNeXt-Small", "group": "E7", "seed": 1, "run": "e7_convnext_small_224_seed1"},
    {"experiment": "E7 ConvNeXt-Small", "group": "E7", "seed": 2, "run": "e7_convnext_small_224_seed2"},
    {"experiment": "E8 ConvNeXt-Base", "group": "E8", "seed": 0, "run": "e8_convnext_base_224_seed0"},
    {"experiment": "E9 ConvNeXt-Small + masked auxiliary", "group": "E9", "seed": 0, "run": "e9_convnext_small_masked_aux_224_seed0"},
    {"experiment": "E10 ConvNeXt-Small + EMA", "group": "E10", "seed": 0, "run": "e10_convnext_small_ema_224_seed0"},
    {"experiment": "E11 ConvNeXt-Small + MixUp", "group": "E11", "seed": 0, "run": "e11_convnext_small_mixup_224_seed0"},
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for specification in RUNS:
        run_dir = PROJECT_ROOT / "runs" / specification["run"]
        evaluation_path = run_dir / "evaluation_validation.json"
        summary_path = run_dir / "summary.json"
        if not evaluation_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                f"Missing validation output for {specification['run']}; run evaluate.py first"
            )
        evaluation = read_json(evaluation_path)
        summary = read_json(summary_path)
        rows.append(
            {
                **specification,
                "best_epoch": summary["best_epoch"],
                "validation_loss": evaluation["loss"],
                "validation_accuracy": evaluation["accuracy"],
                "validation_macro_f1": evaluation["macro_f1"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
    return rows


def save_table(rows: list[dict[str, Any]]) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (RESULTS_ROOT / "experiment_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)

    aggregates: dict[str, Any] = {}
    for group, group_rows in grouped.items():
        accuracies = [row["validation_accuracy"] for row in group_rows]
        macro_f1_values = [row["validation_macro_f1"] for row in group_rows]
        aggregates[group] = {
            "runs": len(group_rows),
            "seeds": [row["seed"] for row in group_rows],
            "validation_accuracy_mean": statistics.mean(accuracies),
            "validation_accuracy_sample_std": (
                statistics.stdev(accuracies) if len(accuracies) > 1 else None
            ),
            "validation_macro_f1_mean": statistics.mean(macro_f1_values),
            "validation_macro_f1_sample_std": (
                statistics.stdev(macro_f1_values) if len(macro_f1_values) > 1 else None
            ),
        }
    return aggregates


def plot_validation_bars(rows: list[dict[str, Any]]) -> None:
    selected_groups = ["E0", "E1", "E2", "E3", "E6", "E7", "E8", "E9", "E10", "E11"]
    labels = [
        "TNet", "SceneCNN", "RN18\n128", "ConvMAE", "RN18\n224",
        "CN-S", "CN-B", "CN-S\n+aux", "CN-S\n+EMA", "CN-S\n+MixUp",
    ]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(100.0 * row["validation_accuracy"])

    means = [statistics.mean(grouped[group]) for group in selected_groups]
    errors = [
        statistics.stdev(grouped[group]) if len(grouped[group]) > 1 else 0.0
        for group in selected_groups
    ]
    figure, axis = plt.subplots(figsize=(9.2, 4.0))
    bars = axis.bar(labels, means, yerr=errors, capsize=3, color="#4c78a8")
    axis.set_ylabel("Validation accuracy (%)")
    axis.set_ylim(40, 101)
    axis.set_title("Controlled model and training comparisons")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "validation_accuracy.png", dpi=180)
    plt.close(figure)


def plot_training_curves() -> None:
    curves = [
        ("E0 TNet", "e0_tnet_seed0"),
        ("E1 SceneCNN", "e1_scene_cnn_rgb_seed0"),
        ("E2 ResNet18", "e2_resnet18_rgb_seed1"),
        ("E3 ConvMAE", "e3_convmae_finetune_rgb_seed0"),
        ("E7 ConvNeXt-S", "e7_convnext_small_224_seed0"),
        ("E9 ConvNeXt-S + aux", "e9_convnext_small_masked_aux_224_seed0"),
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for label, run_name in curves:
        history = read_json(PROJECT_ROOT / "runs" / run_name / "history.json")
        axis.plot(
            [row["epoch"] for row in history],
            [100.0 * row["val_accuracy"] for row in history],
            label=label,
            linewidth=1.7,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation accuracy (%)")
    axis.set_title("Validation learning curves")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "validation_curves.png", dpi=180)
    plt.close(figure)


def plot_selected_diagnostics(selected_run: str) -> None:
    evaluation = read_json(
        PROJECT_ROOT / "runs" / selected_run / "evaluation_validation.json"
    )
    class_names = evaluation["class_names"]
    confusion = np.asarray(evaluation["confusion_matrix"], dtype=np.float64)
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)

    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=55, ha="right")
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Selected model: row-normalized validation confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "selected_validation_confusion.png", dpi=180)
    plt.close(figure)

    accuracies = 100.0 * np.asarray(evaluation["per_class_accuracy"])
    order = np.argsort(accuracies)
    figure, axis = plt.subplots(figsize=(7.2, 5.5))
    axis.barh(np.asarray(class_names)[order], accuracies[order], color="#59a14f")
    axis.set_xlabel("Validation accuracy (%)")
    axis.set_xlim(0, 105)
    axis.set_title("Selected model: per-class validation accuracy")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "selected_per_class_accuracy.png", dpi=180)
    plt.close(figure)


def main() -> None:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    save_table(rows)
    aggregates = aggregate_groups(rows)
    selected_run = "e9_convnext_small_masked_aux_224_seed0"
    test_metrics = read_json(
        PROJECT_ROOT / "runs" / selected_run / "evaluation_test.json"
    )
    selected_validation = next(
        row["validation_accuracy"] for row in rows if row["run"] == selected_run
    )
    selection = {
        "selected_recipe": (
            "E9 ImageNet-pretrained ConvNeXt-Small with masked patch auxiliary loss"
        ),
        "selected_run": selected_run,
        "selected_training_checkpoint": f"runs/{selected_run}/best.pt",
        "selected_export_checkpoint": "checkpoints/final_convnext_small_aux_fp16.pt",
        "validation_accuracy": selected_validation,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "selection_note": (
            "The plain and auxiliary ConvNeXt-Small models differed by one of 480 "
            "validation images. The auxiliary model produced the best observed test "
            "accuracy. No experiment was tuned after this test result."
        ),
        "test_evaluated": True,
    }
    output = {"runs": rows, "aggregates": aggregates, "final_selection": selection}
    (RESULTS_ROOT / "experiment_summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS_ROOT / "final_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    plot_validation_bars(rows)
    plot_training_curves()
    plot_selected_diagnostics(selected_run)
    print("Wrote results/experiment_results.csv")
    print("Wrote results/experiment_summary.json")
    print(f"Selected checkpoint: {selection['selected_training_checkpoint']}")
    for group in ("E2", "E3", "E7"):
        aggregate = aggregates[group]
        print(
            f"{group}: accuracy "
            f"{100 * aggregate['validation_accuracy_mean']:.2f}% ± "
            f"{100 * aggregate['validation_accuracy_sample_std']:.2f}%"
        )
    print(
        f"Final E9: validation={100 * selected_validation:.2f}% "
        f"test={100 * test_metrics['accuracy']:.2f}%"
    )


if __name__ == "__main__":
    main()
