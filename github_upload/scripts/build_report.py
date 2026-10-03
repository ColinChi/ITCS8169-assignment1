#!/usr/bin/env python3
"""Build the assignment's maximum-two-page PDF report."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "report"
METADATA_PATH = REPORT_ROOT / "report_metadata.json"
OUTPUT_PATH = REPORT_ROOT / "assignment1_report.pdf"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#c9c9c9"))
    canvas.line(0.45 * inch, 0.35 * inch, 8.05 * inch, 0.35 * inch)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(0.45 * inch, 0.22 * inch, "ITCS 6169/8169 - Assignment 1: The CNN Challenge")
    canvas.drawRightString(8.05 * inch, 0.22 * inch, f"Page {document.page}")
    canvas.restoreState()


def main() -> None:
    metadata = load_json(METADATA_PATH)
    github_url = metadata["github_url"]
    student_name = metadata["student_name"]

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=17,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365d"),
        spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=8.2,
        leading=10,
        alignment=TA_CENTER,
        spaceAfter=7,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=12,
        textColor=colors.HexColor("#17365d"),
        spaceBefore=5,
        spaceAfter=2,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.25,
        leading=10.25,
        spaceAfter=3,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=7.2,
        leading=8.5,
        spaceAfter=0,
    )
    result_big = ParagraphStyle(
        "ResultBig",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=17,
        textColor=colors.HexColor("#1b5e20"),
        alignment=TA_CENTER,
    )

    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.38 * inch,
        bottomMargin=0.45 * inch,
        title="ITCS 6169/8169 Assignment 1 Report",
        author=student_name,
    )

    story = [
        Paragraph("Small-Data Scene Recognition with ConvNeXt", title),
        Paragraph(
            f"{student_name} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f'<link href="{github_url}" color="#1155cc">{github_url}</link>',
            subtitle,
        ),
    ]

    result_table = Table(
        [
            [Paragraph("BEST TEST ACCURACY", small), Paragraph("VALIDATION ACCURACY", small), Paragraph("TEST MACRO-F1", small)],
            [Paragraph("96.75%", result_big), Paragraph("96.88%", result_big), Paragraph("96.74%", result_big)],
            [Paragraph("387 / 400 images", small), Paragraph("465 / 480 images", small), Paragraph("16 balanced classes", small)],
        ],
        colWidths=[2.52 * inch] * 3,
    )
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef5eb")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#70ad47")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b7cfaa")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend([result_table, Spacer(1, 3)])

    story.extend(
        [
            Paragraph("1. Final Result and Validation Strategy", heading),
            Paragraph(
                "I used a fixed, class-stratified 80/20 split of the 2,400 provided training images "
                "(1,920 train; 480 validation; exactly 30 validation images per class). SHA-256 duplicate "
                "groups were kept on one side of the split; no train/test duplicates were found. Model "
                "selection used validation accuracy with macro-F1 as a diagnostic. The central plain "
                "ConvNeXt-Small recipe was repeated over three seeds and achieved 96.88% +/- 0.21%. "
                "The final auxiliary model scored 96.88% validation accuracy and 96.75% on the 400-image "
                "test set. Its plain counterpart was one validation image better (97.08%) but two test "
                "images worse (96.25%); I therefore treat the observed auxiliary gain as small, not conclusive.",
                body,
            ),
            Paragraph("2. Secret Recipe", heading),
            Paragraph(
                "<b>Architecture and initialization.</b> The classifier is torchvision ConvNeXt-Small "
                "(49.47M parameters), initialized from supervised ImageNet-1K weights. All backbone "
                "parameters and the new 16-class head were fine-tuned. Inputs are 224x224 RGB images "
                "normalized with ImageNet statistics. ConvNeXt-Base was tested and rejected, so capacity "
                "was chosen empirically rather than by size alone.",
                body,
            ),
            Paragraph(
                "<b>Optimization.</b> AdamW used learning rates 2e-5 for the pretrained backbone and "
                "3e-4 for the classification/auxiliary heads, weight decay 0.01, cosine decay to 5e-7, "
                "batch size 32, label smoothing 0.1, dropout 0.2, gradient-norm clipping at 5, FP16 AMP, "
                "and at most 30 epochs. The best validation checkpoint occurred at epoch 10; training "
                "stopped at epoch 18 after eight epochs without improvement.",
                body,
            ),
            Paragraph(
                "<b>Augmentation.</b> Because scene identity depends on global layout, I used conservative "
                "rather than aggressive crops: random resized crop retaining 82-100% of the image, "
                "horizontal flip, +/-5-degree rotation, mild brightness/contrast/saturation jitter, and "
                "10% random erasing. Deterministic resize and center crop were used for validation/test.",
                body,
            ),
            Paragraph(
                "<b>Masked auxiliary objective.</b> During training only, a second pass masks 40% of "
                "32x32 blocks. A 1x1 convolution on the 7x7 ConvNeXt feature map predicts each masked "
                "block's mean normalized RGB value. The total objective is classification cross-entropy "
                "+ 0.2 times masked MSE. The auxiliary head is not called at inference, so test cost is "
                "the same as plain ConvNeXt-Small.",
                body,
            ),
            Paragraph("Reproduction", heading),
            Paragraph(
                "Train: <font name=\"Courier\">python train.py --config "
                "configs/e9_convnext_small_masked_aux_224.json</font><br/>"
                "Evaluate: <font name=\"Courier\">python evaluate.py --checkpoint "
                "checkpoints/final_convnext_small_aux_fp16.pt --split test --confirm-final-test</font>",
                small,
            ),
            PageBreak(),
            Paragraph("3. Experimental Journey", heading),
        ]
    )

    table_rows = [
        ["Experiment", "Validation", "Observation"],
        ["Starter TNet", "45.42%", "Severe under-capacity; starting reference"],
        ["SceneCNN + augmentation", "80.83%", "Residual CNN and augmentation gave +35.4 points"],
        ["Pretrained ResNet18, 128px", "90.62 +/- 1.04%", "ImageNet transfer was the largest clean gain"],
        ["ConvMAE + ResNet18", "90.49 +/- 0.32%", "Self-supervised reconstruction reduced variance, not mean"],
        ["ResNet18, 224px", "92.71%", "Resolution and layout-preserving crops helped"],
        ["ConvNeXt-Small", "96.88 +/- 0.21%", "Strongest stable backbone; best single run 97.08%"],
        ["ConvNeXt-Base", "95.21%", "Extra capacity hurt on 1,920 training images"],
        ["ConvNeXt-S + masked aux", "96.88%", "Validation tied within one image; test 96.75%"],
        ["ConvNeXt-S + EMA", "96.25%", "EMA lagged behind during short fine-tuning"],
        ["ConvNeXt-S + MixUp", "96.67%", "Additional regularization did not improve validation"],
    ]
    experiment_table = Table(
        [[Paragraph(str(cell), small) for cell in row] for row in table_rows],
        colWidths=[2.15 * inch, 1.18 * inch, 4.23 * inch],
        repeatRows=1,
    )
    experiment_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b7b7b7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f9")]),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    story.extend(
        [
            experiment_table,
            Paragraph("4. Failure Analysis", heading),
            Paragraph(
                "I expected masked reconstruction to adapt a CNN to the scene domain. The first attempt "
                "used a convolutional ResNet18 autoencoder for 30 epochs, then fine-tuned its encoder. "
                "It reached 90.49% +/- 0.32%, essentially the same as direct ImageNet initialization "
                "(90.62% +/- 1.04%). Reconstructions captured coarse layout but were blurry, and only "
                "1,920 unlabeled images were available. The lesson was that a label-free objective is not "
                "automatically useful when the external supervised pretraining set is much larger. This "
                "motivated the cheaper joint auxiliary head rather than more self-supervised pretraining. "
                "Similarly, ConvNeXt-Base (95.21%), EMA (96.25%), and MixUp (96.67%) showed that extra "
                "capacity or regularization can be counterproductive in this small-data regime.",
                body,
            ),
            Paragraph("5. Error Analysis", heading),
            Paragraph(
                "The final model's weakest test class was <b>InsideCity</b> (84%), followed by "
                "<b>Mountain</b> (92%); 14 of 16 classes were at least 96%, and seven reached 100%. "
                "The remaining errors concentrate in visually heterogeneous outdoor/urban categories, "
                "where local texture is insufficient and scene boundaries are ambiguous. All Flower "
                "images are RGB while the other classes are grayscale; a grayscale ResNet control "
                "changed validation accuracy only from 89.58% to 89.79%, so this artifact was disclosed "
                "but did not explain the overall gains.",
                body,
            ),
            Paragraph("6. AI + Human", heading),
            Paragraph(
                "OpenAI Codex helped audit the data, implement the configuration-driven PyTorch pipeline, "
                "add tests, and generate plotting/report utilities. Its initial suggestion of a standard "
                "ViT-style MAE was incompatible with the CNN-only rule; after I checked the assignment, "
                "I rejected that route and required convolutional encoders. I also chose to compare "
                "ConvNeXt-Small with Base and to stop after the pre-declared EMA/MixUp ablations instead "
                "of continuing to search after negative results. Full details, including a questionable "
                "early test evaluation and how it was corrected, are documented in AI_USAGE.md.",
                body,
            ),
            Paragraph("Reproducibility note", heading),
            Paragraph(
                "The repository contains the fixed split manifest, all E0-E11 configurations, source and "
                "evaluation code, ten automated tests, environment requirements, aggregate JSON/CSV "
                "results, and the 99.06 MB FP16 inference checkpoint. The exported checkpoint was reloaded "
                "and reproduced 96.88% validation accuracy.",
                body,
            ),
        ]
    )

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
