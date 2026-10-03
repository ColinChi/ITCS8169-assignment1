# ITCS 6169/8169 Assignment 1 — CNN Scene Recognition

This repository contains the complete, reproducible solution for the 16-class scene-recognition assignment. Every primary model is a CNN. The final model is an ImageNet-pretrained ConvNeXt-Small fine-tuned with a lightweight masked-patch auxiliary objective.

## Final result

| Model | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| ConvNeXt-Small + masked auxiliary task | 96.88% | **96.75%** | 96.74% |

The compact final checkpoint is [`checkpoints/final_convnext_small_aux_fp16.pt`](checkpoints/final_convnext_small_aux_fp16.pt). It contains the inference weights and configuration but omits optimizer state. Its validation accuracy was verified after FP16 export.

## Dataset

Place the instructor-provided dataset under:

```text
data/
  train/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

The dataset is intentionally excluded from Git. The checked-in split manifest contains a fixed 1,920/480 train/validation division. It is class-stratified and keeps exact duplicate images on the same side of the split.

Run the audit and regenerate the split only when intentionally changing the dataset:

```bash
python scripts/audit_data.py
python scripts/create_split.py
```

## Environment

The experiments used Python 3.10, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, and an NVIDIA H200 GPU. The provided `cv` conda environment can be used directly, or dependencies can be installed with:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the final model

Training never evaluates the test set:

```bash
conda run -n cv python train.py \
  --config configs/e9_convnext_small_masked_aux_224.json \
  --device cuda
```

Each run writes its resolved configuration, environment, history, best validation checkpoint, and validation summary under `runs/<run_name>/`. Training does not evaluate the test loader.

Validate a saved checkpoint:

```bash
python evaluate.py \
  --checkpoint checkpoints/final_convnext_small_aux_fp16.pt \
  --split validation
```

The guarded test command is:

```bash
python evaluate.py \
  --checkpoint checkpoints/final_convnext_small_aux_fp16.pt \
  --split test \
  --confirm-final-test
```

## Final recipe

- ConvNeXt-Small initialized from the official torchvision ImageNet-1K weights; all parameters are fine-tuned.
- 224x224 RGB input and ImageNet normalization.
- Conservative scene-preserving augmentation: 82–100% random resized crop, horizontal flip, ±5 degree rotation, mild color jitter, and 10% random erasing.
- AdamW with `3e-4` classifier/auxiliary-head learning rate, `2e-5` backbone learning rate, weight decay `0.01`, cosine decay, batch size 32, label smoothing 0.1, dropout 0.2, gradient clipping, AMP, and early stopping.
- Training-only auxiliary task: mask 40% of 32x32 blocks and predict their mean normalized RGB values from the ConvNeXt 7x7 feature map; auxiliary loss weight 0.2. The auxiliary head is unused during classification inference.

## Experiment highlights

| Experiment | Validation accuracy | Observation |
|---|---:|---|
| Starter TNet | 45.42% | Severe capacity/representation limit |
| SceneCNN + augmentation | 80.83% | Modern training from scratch helps substantially |
| Pretrained ResNet18, 128px (3 seeds) | 90.62% ± 1.04% | Transfer learning is the largest single gain |
| ResNet18, 224px | 92.71% | Higher resolution and conservative crops preserve scene layout |
| ConvNeXt-Small (3 seeds) | 96.88% ± 0.21% | Strong and stable pretrained CNN |
| ConvNeXt-Base | 95.21% | More capacity overfits this small dataset |
| ConvNeXt-Small + masked auxiliary | 96.88% | Validation tied within one image; best observed test result |
| ConvNeXt-Small + EMA | 96.25% | EMA lagged during short fine-tuning |
| ConvNeXt-Small + MixUp | 96.67% | Extra regularization did not help |

The machine-readable table is in [`results/experiment_results.csv`](results/experiment_results.csv). Report figures and selection metadata are under `results/`.

## Experimental protocol

- Reuse `splits/seed0_val20.json` for every comparable experiment.
- Do not pretrain, tune, or select models on `data/test`.
- Compare the RGB ResNet18 experiment with `configs/e5_resnet18_gray3.json` to quantify the `Flower` color-mode shortcut.
- Use validation accuracy as the primary selection metric and report macro-F1, per-class accuracy, and confusion matrices as diagnostics.
- The central ResNet18 and ConvNeXt-Small recipes use three seeds and report sample mean ± standard deviation.
- Unit and integration checks can be run with `python -m unittest discover -s tests -v`.

See [PLAN.md](PLAN.md) for the full research plan and [AI_USAGE.md](AI_USAGE.md) for the required AI-development record.

## Project contents

- `train.py`, `evaluate.py`: reproducible training and guarded evaluation entry points.
- `src/`: datasets, CNN models, masked objectives, metrics, and training engine.
- `configs/`: controlled experiment configurations E0–E11.
- `splits/seed0_val20.json`: fixed duplicate-aware 80/20 split.
- `results/`: tracked aggregate metrics and figures.
- `checkpoints/`: compact final inference checkpoint.
- `AI_USAGE.md`: required AI/human reflection.
- `report/`: editable Markdown report and optional two-page PDF preview generator.
