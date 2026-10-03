# GitHub upload checklist

This folder is the clean repository bundle. It intentionally excludes the dataset, source ZIP archives, `runs/`, caches, and local Git metadata.

## 1. Finalize the editable report

Edit `report/assignment1_report.md`, replace `YOUR NAME` and the GitHub URL, and make any desired content changes. Export the final submission as a maximum two-page PDF using your preferred Markdown editor or document tool.

An optional ReportLab preview generator is also included. To use it, edit `report/report_metadata.json`:

Edit `report/report_metadata.json`:

```json
{
  "student_name": "Your Name",
  "github_url": "https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY"
}
```

Then rebuild and verify the two-page PDF:

```bash
conda run -n cv python scripts/build_report.py
conda run -n cv python -c "from pypdf import PdfReader; print(len(PdfReader('report/assignment1_report.pdf').pages))"
```

The printed page count must be `2`. Open the final PDF once and verify the name and clickable repository link before submission.

## 2. Verify the repository

The dataset must be placed under `data/train/` and `data/test/` only when reproducing locally; `data/` is ignored by Git.

```bash
conda run -n cv python -m unittest discover -s tests -v
conda run -n cv python evaluate.py \
  --checkpoint checkpoints/final_convnext_small_aux_fp16.pt \
  --split validation --device cuda
```

Expected validation accuracy: `96.88%`. The checkpoint SHA-256 is recorded in `results/final_checkpoint_sha256.txt`.

## 3. Create a meaningful commit history

Run these commands from this folder, replacing the identity and repository URL:

```bash
git init -b main
git config user.name "Your Name"
git config user.email "your.email@example.com"

git add .gitignore requirements.txt train.py evaluate.py evaluate_ensemble.py pretrain_mae.py src tests splits scripts/audit_data.py scripts/create_split.py
git commit -m "Build reproducible CNN training and validation pipeline"

git add configs results checkpoints assignment1_solution.ipynb scripts/summarize_results.py scripts/export_checkpoint.py
git commit -m "Add controlled experiments and final ConvNeXt checkpoint"

git add README.md AI_USAGE.md report scripts/build_report.py scripts/build_notebook.py GITHUB_UPLOAD_INSTRUCTIONS.md
git commit -m "Document results and add two-page assignment report"

git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

The final checkpoint is approximately 99.06 MB. It is below GitHub's 100 MiB Git limit but above the browser uploader's 25 MB limit, so push this repository with Git rather than dragging files into the GitHub website.

## 4. Submission check

- Confirm the GitHub repository link works in a logged-out/private browser as appropriate, or grant the instructor access if the repository is private.
- Confirm `checkpoints/final_convnext_small_aux_fp16.pt` is visible in the repository.
- Submit `report/assignment1_report.pdf` as the single assignment PDF.
- Do not upload the instructor-provided dataset.
