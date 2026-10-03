# AI Usage Record

AI coding assistance was required for this assignment. This file records representative interactions and the checks used to retain human ownership of the research decisions.

## Tool used

- OpenAI Codex

## Representative assistance

1. Parsed the starter notebook and extracted the task constraints, especially the requirement that the primary classifier be a CNN.
2. Audited class counts, image formats, image modes, corrupt files, and exact duplicate content across train/test.
3. Implemented a controlled sequence covering supervised CNN improvements, ImageNet transfer learning, resolution, model capacity, masked reconstruction, EMA, and MixUp.
4. Implemented a deterministic, class-stratified split generator that keeps exact duplicate images together to prevent validation leakage.
5. Implemented configuration-driven training, validation checkpointing, macro-F1/confusion-matrix metrics, a guarded final-test command, regression tests, result plots, and a compact inference checkpoint exporter.

## Incorrect or questionable suggestion and correction

Before reading the assignment, the AI proposed a conventional ViT-style MAE route. That was incompatible with the CNN-only primary-classifier rule. After reading the source document, the design was corrected to keep convolutional encoders. A ResNet18 convolutional masked autoencoder was tested first; the final ConvNeXt experiment used a smaller training-only masked-patch prediction head.

The AI also recommended EMA as a likely low-cost improvement. In the controlled experiment, decay 0.995 reached only 96.25% validation accuracy, while the raw weights in the same run reached 97.08%. I rejected EMA rather than tuning repeatedly after seeing the result. ConvNeXt-Base and MixUp were likewise retained as negative results rather than hidden.

A questionable workflow decision occurred when the first ResNet18 candidate was evaluated on the test set before the later ConvNeXt experiments were complete. Subsequent model development returned to validation-only comparisons, and the evaluation program requires an explicit `--confirm-final-test` flag. The final report does not treat repeated test observations as validation evidence.

## How suggestions were verified

- Notebook requirements were read directly rather than inferred from the course name.
- All 2,800 valid images were opened and verified with Pillow.
- SHA-256 hashes were used to check exact duplicates. No train/test duplicates were found; six duplicate groups were found within `LivingRoom` training data.
- The generated split is checked for class counts, path overlap, missing files, and duplicate groups crossing train/validation.
- Ten unit/integration tests verify split integrity, model output shapes, metrics, masked reconstruction, masked auxiliary loss, EMA updates, and MixUp behavior.
- The strongest plain ConvNeXt-Small recipe was run with seeds 0, 1, and 2: 96.88% ± 0.21% validation accuracy.
- The compact FP16 checkpoint was reloaded and reproduced 96.88% validation accuracy after export.
- Each claimed result is stored as JSON/CSV, and the training path never calls the test loader.

## Important student decision

I chose to replace ResNet18 with ConvNeXt-Small after the ResNet experiments, while also testing ConvNeXt-Base rather than assuming a larger model would be better. I limited the final phase to controlled one-change experiments (masked auxiliary prediction, EMA, and MixUp), stopped after MixUp, and kept the negative results. The final auxiliary model differs from the plain model by only one validation image, so I report its 0.5-point test improvement as a small observation rather than proof of a universally better method.
