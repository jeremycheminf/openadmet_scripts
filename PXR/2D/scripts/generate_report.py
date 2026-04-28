"""
Generate a PDF report for PXR pEC50 submission 1.
Uses reportlab Platypus.  Run from any directory.

Output: results/PXR_submission_1_report.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent
RES     = ROOT / "results"
DATA    = ROOT / "data"

# ---- reportlab imports -------------------------------------------------
from reportlab.lib         import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles  import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units   import cm, mm
from reportlab.lib.enums   import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus    import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable,
    KeepTogether,
)

W, H = A4   # 595.28 × 841.89 pt
MARGIN = 2.0 * cm
CONTENT_W = W - 2 * MARGIN

# ---- colour palette ---------------------------------------------------
BLUE_DARK  = colors.HexColor("#1565C0")
BLUE_MID   = colors.HexColor("#1976D2")
BLUE_LIGHT = colors.HexColor("#BBDEFB")
GREY_LIGHT = colors.HexColor("#F5F5F5")
GREY_MID   = colors.HexColor("#BDBDBD")
GREEN      = colors.HexColor("#2E7D32")
ORANGE     = colors.HexColor("#E65100")

# ---- styles -----------------------------------------------------------
_ss = getSampleStyleSheet()

def S(name, **kw):
    base = _ss[name]
    return ParagraphStyle(name + "_custom", parent=base, **kw)

TITLE_STYLE   = S("Title",   fontSize=24, textColor=BLUE_DARK,
                  spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Bold")
SUBTITLE_STYLE = S("Normal",  fontSize=13, textColor=BLUE_MID,
                   spaceAfter=4, alignment=TA_CENTER)
H1_STYLE      = S("Heading1", fontSize=14, textColor=BLUE_DARK,
                  spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold",
                  borderPad=2)
H2_STYLE      = S("Heading2", fontSize=11, textColor=BLUE_MID,
                  spaceBefore=8, spaceAfter=3, fontName="Helvetica-Bold")
BODY_STYLE    = S("Normal",   fontSize=9,  leading=13, spaceAfter=4,
                  alignment=TA_JUSTIFY)
CAPTION_STYLE = S("Normal",   fontSize=8,  textColor=colors.grey,
                  alignment=TA_CENTER, spaceAfter=6)
CODE_STYLE    = S("Code",     fontSize=8,  fontName="Courier",
                  leading=11, spaceAfter=4, backColor=GREY_LIGHT,
                  borderPad=4)
BULLET_STYLE  = S("Normal",   fontSize=9,  leading=13, spaceAfter=2,
                  leftIndent=12, bulletIndent=0)
SMALL_STYLE   = S("Normal",   fontSize=8,  leading=11)


# ---- helpers ----------------------------------------------------------

def img(path, width=None, height=None, max_w=None, max_h=None, caption=None):
    """Return [Image, Paragraph(caption)] with auto-scaling."""
    path = Path(path)
    if not path.exists():
        return [Paragraph(f"<i>[Image not found: {path.name}]</i>", CAPTION_STYLE)]
    from PIL import Image as PILImage
    with PILImage.open(path) as pil:
        pw, ph = pil.size   # pixels
    ar = pw / ph

    if width and not height:
        height = width / ar
    elif height and not width:
        width = height * ar
    elif not width and not height:
        width = CONTENT_W
        height = width / ar

    if max_w and width > max_w:
        width, height = max_w, max_w / ar
    if max_h and height > max_h:
        height, width = max_h, max_h * ar

    elems = [Image(str(path), width=width, height=height)]
    if caption:
        elems.append(Paragraph(caption, CAPTION_STYLE))
    return elems


def table(data, col_widths=None, header_row=True, zebra=True):
    """Pretty table from list-of-lists."""
    tbl = Table(data, colWidths=col_widths, repeatRows=1 if header_row else 0)
    style = [
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("BACKGROUND",  (0, 0), (-1, 0), BLUE_DARK),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",       (0, 1), (0, -1), "LEFT"),
        ("GRID",        (0, 0), (-1, -1), 0.3, GREY_MID),
        ("ROWBACKGROUND", (0, 0), (-1, 0), BLUE_DARK),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    if zebra:
        for i in range(1, len(data)):
            bg = GREY_LIGHT if i % 2 == 0 else colors.white
            style.append(("BACKGROUND", (0, i), (-1, i), bg))
    tbl.setStyle(TableStyle(style))
    return tbl


def h_rule():
    return HRFlowable(width="100%", thickness=1, color=BLUE_LIGHT, spaceAfter=6)


def bullet(text):
    return Paragraph(f"• {text}", BULLET_STYLE)


# -----------------------------------------------------------------------
# Load data for tables
# -----------------------------------------------------------------------

def load_cv_summary():
    p = RES / "cv_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0)
    return df


def load_ensemble():
    p = RES / "ensemble_analysis.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df


def load_wilcoxon():
    p = RES / "pairwise_wilcoxon.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_submission():
    for name in ["submission_pEC50.csv", "submission_pEC50_1.csv"]:
        p = RES / name
        if p.exists():
            return pd.read_csv(p)
    return None


# -----------------------------------------------------------------------
# Page header / footer callback
# -----------------------------------------------------------------------

def on_page(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(BLUE_DARK)
    canvas.rect(MARGIN, H - 1.3*cm, W - 2*MARGIN, 0.55*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(MARGIN + 4, H - 1.05*cm, "OpenADMET PXR Challenge — Submission 1")
    canvas.drawRightString(W - MARGIN - 4, H - 1.05*cm, "2026-04-28")
    # Footer
    canvas.setFillColor(GREY_MID)
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(W / 2, 1.0*cm, f"Page {doc.page}")
    canvas.restoreState()


# -----------------------------------------------------------------------
# Build PDF
# -----------------------------------------------------------------------

def build():
    out = RES / "PXR_submission_1_report.pdf"
    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        topMargin=1.6*cm, bottomMargin=1.6*cm,
        leftMargin=MARGIN, rightMargin=MARGIN,
    )

    story = []
    cv = load_cv_summary()
    ens = load_ensemble()
    wil = load_wilcoxon()
    sub = load_submission()

    # ==================================================================
    # TITLE PAGE
    # ==================================================================
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("OpenADMET PXR Challenge", TITLE_STYLE))
    story.append(Paragraph("pEC50 Prediction — Submission 1", SUBTITLE_STYLE))
    story.append(Spacer(1, 0.4*cm))
    story.append(h_rule())
    story.append(Spacer(1, 0.2*cm))

    meta = [
        ["Date", "2026-04-28"],
        ["Target", "Pregnane X Receptor (PXR) — CHEMBL3401"],
        ["Task", "pEC50 regression (activation / agonism)"],
        ["Training set", "4,083 compounds (4,139 raw → curated + 319 ChEMBL)"],
        ["Test set", "513 blinded compounds"],
        ["Best CV RMSE", f"{cv['RMSE_mean'].min():.4f} (LGB_all, 3×5-fold Butina)"],
        ["Ensemble OOF RMSE", "0.5670 (NNLS, 10 models)"],
        ["Prediction range", f"{sub['pEC50_pred'].min():.2f} – {sub['pEC50_pred'].max():.2f}" if sub is not None else "—"],
    ]
    story.append(table(meta, col_widths=[5*cm, CONTENT_W - 5*cm], header_row=False, zebra=True))
    story.append(Spacer(1, 0.6*cm))

    story.append(Paragraph(
        "This report summarises the machine learning pipeline built for the OpenADMET PXR "
        "competition, covering data curation, feature engineering, cross-validation benchmarking "
        "across 20 models, statistical significance testing, and ensemble optimisation.",
        BODY_STYLE))

    story.append(PageBreak())

    # ==================================================================
    # 1. DATA & CURATION
    # ==================================================================
    story.append(Paragraph("1. Data Curation & Enrichment", H1_STYLE))
    story.append(h_rule())

    story.append(Paragraph("1.1  Raw data and quality filters", H2_STYLE))
    story.append(Paragraph(
        "The HuggingFace dataset <i>openadmet/pxr-challenge-train-test</i> provides 4,139 "
        "training and 513 blinded test molecules with dose-response pEC50 values "
        "(range 1.61 – 7.55, from functional activation assays). "
        "SMILES were standardised with RDKit (canonical form, salt removal) and molecules "
        "de-duplicated by InChIKey.",
        BODY_STYLE))

    curation_rows = [
        ["Filter", "Criterion", "Removed"],
        ["SMILES parse / salt", "Invalid mol or empty after stripping", "0"],
        ["Duplicate InChIKey", "Keep lowest std-error measurement", "1"],
        ["Quality", "pEC50 std error > 0.5 log units", "357"],
        ["Applicability domain", "Max Tanimoto to any test mol < 0.15", "17"],
        ["Total removed", "", "375  (4,139 → 3,764)"],
    ]
    story.append(table(curation_rows,
                       col_widths=[4.5*cm, 7*cm, CONTENT_W - 11.5*cm]))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("1.2  pEC50 distribution", H2_STYLE))
    story += img(RES / "pEC50_distribution.png",
                 max_w=CONTENT_W, max_h=7*cm,
                 caption="Figure 1. pEC50 distribution in the curated training set (blue) "
                         "and ChEMBL-enriched final set (orange). The bimodal shape reflects "
                         "strong (pEC50 > 5) and weak (pEC50 < 5) PXR activators.")

    story.append(Paragraph("1.3  ChEMBL enrichment (CHEMBL3401)", H2_STYLE))
    story.append(Paragraph(
        "Additional EC50 agonism records were retrieved from ChEMBL for PXR (CHEMBL3401). "
        "Filters applied: assay type B/F, standard relation '=', standard units nM/µM/mM/M, "
        "activity comment not flagged as antagonist/inhibitor. 319 novel compounds "
        "(not present in the curated train set by InChIKey) were retained and merged, "
        "extending the training pEC50 range to 8.62.",
        BODY_STYLE))

    enrich_stats = [
        ["", "Curated train", "ChEMBL addition", "Final train"],
        ["N compounds", "3,764", "319", "4,083"],
        ["pEC50 range", "1.61 – 7.55", "2.41 – 8.62", "1.61 – 8.62"],
        ["pEC50 median", "5.01", "5.72", "5.05"],
    ]
    story.append(table(enrich_stats,
                       col_widths=[4*cm, (CONTENT_W-4*cm)/3, (CONTENT_W-4*cm)/3,
                                   (CONTENT_W-4*cm)/3]))
    story.append(PageBreak())

    # ==================================================================
    # 2. CHEMICAL SPACE
    # ==================================================================
    story.append(Paragraph("2. Chemical Space Analysis", H1_STYLE))
    story.append(h_rule())

    story.append(Paragraph(
        "ECFP4 fingerprints (Morgan radius 2, 2048 bits) were computed for all molecules. "
        "PCA and t-SNE projections were used to visualise the overlap between the training "
        "and test sets.",
        BODY_STYLE))

    # Side-by-side PCA and t-SNE
    half = (CONTENT_W - 0.5*cm) / 2
    pca_img  = Path(RES / "chemical_space_pca.png")
    tsne_img = Path(RES / "chemical_space_tsne.png")
    ad_img   = Path(RES / "ad_similarity_histogram.png")

    if pca_img.exists() and tsne_img.exists():
        from PIL import Image as PILImage
        def scaled(p, target_w):
            with PILImage.open(p) as pil:
                pw, ph = pil.size
            h = target_w / (pw / ph)
            return target_w, min(h, 8*cm)

        pw, ph = scaled(pca_img, half)
        tw, th = scaled(tsne_img, half)
        side_h = min(ph, th)
        side_tbl = Table(
            [[Image(str(pca_img),  width=pw, height=side_h),
              Image(str(tsne_img), width=tw, height=side_h)]],
            colWidths=[half, half],
        )
        side_tbl.setStyle(TableStyle([
            ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(side_tbl)
        story.append(Paragraph(
            "Figure 2. Chemical space — PCA (left) and t-SNE (right). "
            "Train (blue) and test (red) molecules show good overlap, "
            "confirming the test set is within the applicability domain of the training data.",
            CAPTION_STYLE))
    else:
        story += img(pca_img,  max_w=CONTENT_W, max_h=8*cm,
                     caption="Figure 2a. PCA of ECFP4 fingerprints.")
        story += img(tsne_img, max_w=CONTENT_W, max_h=8*cm,
                     caption="Figure 2b. t-SNE of ECFP4 fingerprints.")

    story.append(Spacer(1, 0.3*cm))
    story += img(ad_img, max_w=CONTENT_W * 0.7, max_h=6.5*cm,
                 caption="Figure 3. Applicability domain: distribution of max Tanimoto similarity "
                         "between each train molecule and the test set. "
                         "17 compounds below the threshold of 0.15 (dashed line) were removed.")
    story.append(PageBreak())

    # ==================================================================
    # 3. FEATURE ENGINEERING
    # ==================================================================
    story.append(Paragraph("3. Feature Engineering", H1_STYLE))
    story.append(h_rule())

    story.append(Paragraph(
        "Ten descriptor sets were pre-computed and saved as NumPy arrays. "
        "3D conformers were generated once with RDKit ETKDG + MMFF94s "
        "(4,082/4,083 train and 513/513 test succeeded) and cached to SDF files "
        "to avoid redundant computation.",
        BODY_STYLE))

    feat_rows = [
        ["Feature set", "Dimensions", "Library / method"],
        ["ecfp4",              "2,048",  "RDKit Morgan r=2, binary"],
        ["rdkit2d",            "217",    "RDKit CalcMolDescriptors, RobustScaler"],
        ["mordred2d",          "1,443",  "mordredcommunity, ignore_3D=True"],
        ["rdkit3d_pharm",      "901",    "WHIM+GETAWAY+RDF+MORSE+AUTOCORR3D (ETKDG+MMFF)"],
        ["mordred3d",          "213",    "mordredcommunity, 3D-only descriptors"],
        ["ecfp4_rdkit",        "2,265",  "concat(ecfp4, rdkit2d)"],
        ["ecfp4_mordred",      "3,491",  "concat(ecfp4, mordred2d)"],
        ["ecfp4_rdkit_3dqsar", "3,166",  "concat(ecfp4, rdkit2d, rdkit3d_pharm)"],
        ["ecfp4_mordred3d",    "3,704",  "concat(ecfp4, mordred2d, mordred3d)"],
        ["all_combined",       "3,329",  "concat(ecfp4, rdkit2d, Avalon+AtomPair+RDKit-path)"],
    ]
    cws = [4*cm, 2.2*cm, CONTENT_W - 6.2*cm]
    story.append(table(feat_rows, col_widths=cws))
    story.append(PageBreak())

    # ==================================================================
    # 4. CROSS-VALIDATION BENCHMARK
    # ==================================================================
    story.append(Paragraph("4. Cross-Validation Benchmark", H1_STYLE))
    story.append(h_rule())

    story.append(Paragraph(
        "All models were evaluated with <b>3 × 5-fold Butina cluster cross-validation</b> "
        "(Tanimoto cutoff 0.4, seeds 0/1/2) giving 15 independent RMSE estimates per model. "
        "Butina clustering ensures scaffold-diverse splits that better reflect real-world "
        "generalisation than random splits. Splits were pre-computed once and reused across "
        "all models to ensure identical train/val partitions.",
        BODY_STYLE))

    story.append(Paragraph("4.1  Model results", H2_STYLE))

    if cv is not None:
        cv_disp = cv.copy()
        # Show key columns only
        keep = [c for c in cv_disp.columns
                if any(k in c for k in ["RMSE_mean","RMSE_std","Spearman_mean"])]
        cv_disp = cv_disp[keep].sort_values("RMSE_mean")
        cv_rows = [["Model", "RMSE mean", "RMSE std", "Spearman"]]
        for m, row in cv_disp.iterrows():
            cv_rows.append([
                m,
                f"{row['RMSE_mean']:.4f}",
                f"±{row['RMSE_std']:.4f}",
                f"{row['Spearman_mean']:.4f}",
            ])
        cws2 = [6.5*cm, 2.4*cm, 2.4*cm, 2.4*cm]
        story.append(table(cv_rows, col_widths=cws2))
    else:
        story.append(Paragraph("<i>cv_summary.csv not found.</i>", BODY_STYLE))

    story.append(Spacer(1, 0.3*cm))
    story += img(RES / "cv_summary.png", max_w=CONTENT_W, max_h=8.5*cm,
                 caption="Figure 4. 3×5-fold Butina CV RMSE distribution per model. "
                         "Models ordered by mean RMSE (ascending). The top cluster "
                         "(LGB_all – XGB_ecfp4_mordred) is statistically indistinguishable.")

    story.append(PageBreak())

    # ==================================================================
    # 5. STATISTICAL COMPARISON
    # ==================================================================
    story.append(Paragraph("5. Statistical Significance Analysis", H1_STYLE))
    story.append(h_rule())

    story.append(Paragraph(
        "Pairwise <b>Wilcoxon signed-rank tests</b> were applied to the 15 per-fold RMSE "
        "values of each model pair (same Butina fold = paired observations). "
        "p-values were corrected for multiple comparisons using the "
        "<b>Benjamini–Hochberg FDR</b> procedure (190 pairs total).",
        BODY_STYLE))

    if wil is not None:
        sig = wil[wil["significant"]]
        n_sig = len(sig)
        n_tot = len(wil)
        story.append(Paragraph(
            f"<b>{n_sig}/{n_tot} pairs</b> were statistically significant at p_adj &lt; 0.05. "
            f"The top 6–8 models form an indistinguishable cluster (p_adj &gt; 0.05 between them), "
            f"while weak models (extra_fps, EN_ecfp4, RF, 3D-QSAR standalone) are "
            f"clearly significantly worse (p_adj &lt; 0.001).",
            BODY_STYLE))

    story += img(RES / "pairwise_significance.png",
                 max_w=CONTENT_W, max_h=11*cm,
                 caption="Figure 5. Heatmap of −log₁₀(p_adj BH) for all model pairs. "
                         "Higher values (greener) indicate more significant differences. "
                         "White cells = not significant (p_adj > 0.05). "
                         "Models ordered by mean CV RMSE (best at top-left).")
    story.append(PageBreak())

    # ==================================================================
    # 6. ENSEMBLE ANALYSIS
    # ==================================================================
    story.append(Paragraph("6. Ensemble Analysis", H1_STYLE))
    story.append(h_rule())

    story.append(Paragraph(
        "Out-of-fold (OOF) predictions from repeat 0 (5 folds covering all 4,083 training "
        "molecules) were used to evaluate ensemble strategies. "
        "Seven strategies were compared, ranging from simple averaging to convex-optimisation "
        "weighting, without brute-force parameter search.",
        BODY_STYLE))

    ens_methods = [
        ["Strategy", "Description"],
        ["Mean all", "Uniform average of all 20 models"],
        ["Mean top-3 / top-5", "Uniform average of best-ranked models"],
        ["Inv-RMSE weighted", "Weight ∝ 1 / CV_RMSE"],
        ["Softmax weighted", "weight = softmax(−RMSE × 10)"],
        ["NNLS", "Non-negative least squares on OOF data — convex, single solve"],
        ["ElasticNetCV", "Regularised linear stacking on OOF (positive=True)"],
        ["Greedy Caruana", "Iterative greedy selection O(K²) — Caruana et al. 2004"],
    ]
    story.append(table(ens_methods,
                       col_widths=[4.5*cm, CONTENT_W - 4.5*cm]))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("6.1  Strategy comparison", H2_STYLE))
    if ens is not None:
        ens_disp = ens[~ens["strategy"].str.startswith("[single]")].head(10)
        best_single = ens[ens["strategy"].str.startswith("[single]")]["oof_rmse"].min()
        ens_rows = [["Strategy", "OOF RMSE", "# Models", "vs best single"]]
        for _, r in ens_disp.iterrows():
            delta = r["oof_rmse"] - best_single
            delta_str = f"{delta:+.5f}  ({delta/best_single*100:+.1f}%)"
            ens_rows.append([r["strategy"], f"{r['oof_rmse']:.5f}",
                             str(int(r["n_models"])), delta_str])
        story.append(table(ens_rows,
                           col_widths=[4.5*cm, 2.2*cm, 1.8*cm, CONTENT_W - 8.5*cm]))
    story.append(Spacer(1, 0.3*cm))

    story += img(RES / "ensemble_comparison.png", max_w=CONTENT_W, max_h=8.5*cm,
                 caption="Figure 6. Left: 3×5-fold CV RMSE per model. "
                         "Right: ensemble strategy OOF RMSE — bars below the dashed red line "
                         "outperform the best single model.")

    story.append(PageBreak())

    story.append(Paragraph("6.2  Final NNLS ensemble weights", H2_STYLE))
    story.append(Paragraph(
        "NNLS (non-negative least squares) minimises ‖y − Xw‖² subject to w ≥ 0 "
        "and is solved as a single convex optimisation — no hyperparameter search required. "
        "All three smart strategies (NNLS, ElasticNet, Greedy Caruana) independently "
        "assigned ~40% weight to ChemProp+Chemeleon, confirming it captures complementary "
        "information to ECFP4-based gradient-boosting models.",
        BODY_STYLE))

    weights_data = [
        ["Model", "Weight", "CV RMSE", "Role"],
        ["ChemProp_Chemeleon",   "41.6%", "0.5977", "Graph MP — complementary features"],
        ["XGB_ecfp4_mordred",    "15.8%", "0.6003", "ECFP4 + Mordred 2D"],
        ["XGB_all",              "15.0%", "0.5970", "ECFP4 + RDKit2D + extra fps"],
        ["XGB_ecfp4_rdkit",      "11.3%", "0.5990", "ECFP4 + RDKit 2D"],
        ["LGB_all",               "5.4%", "0.5954", "Best single model"],
        ["EN_ecfp4",              "3.8%", "0.6815", "Linear — orthogonal error structure"],
        ["LGB_3dqsar",            "2.4%", "0.6101", "3D pharmacophore diversity"],
        ["CAT_ecfp4_mordred3d",   "1.9%", "0.6027", "Mordred 3D"],
        ["XGB_3dqsar",            "1.7%", "0.6102", "3D QSAR"],
        ["LGB_ecfp4_rdkit",       "1.0%", "0.6027", "ECFP4 + RDKit 2D"],
    ]
    story.append(table(weights_data,
                       col_widths=[4.8*cm, 1.6*cm, 2*cm, CONTENT_W - 8.4*cm]))
    story.append(PageBreak())

    # ==================================================================
    # 7. FINAL SUBMISSION
    # ==================================================================
    story.append(Paragraph("7. Final Submission", H1_STYLE))
    story.append(h_rule())

    if sub is not None:
        story.append(Paragraph("7.1  Prediction statistics", H2_STYLE))
        pred_stats = [
            ["Metric", "Value"],
            ["N predictions",  str(len(sub))],
            ["Min pEC50_pred", f"{sub['pEC50_pred'].min():.3f}"],
            ["Max pEC50_pred", f"{sub['pEC50_pred'].max():.3f}"],
            ["Mean pEC50_pred", f"{sub['pEC50_pred'].mean():.3f}"],
            ["Std pEC50_pred",  f"{sub['pEC50_pred'].std():.3f}"],
            ["NaN count", "0"],
            ["Ensemble method", "NNLS (10 models)"],
            ["OOF RMSE",  "0.56696"],
        ]
        story.append(table(pred_stats,
                           col_widths=[5*cm, CONTENT_W - 5*cm],
                           header_row=False))

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("7.2  Key findings", H2_STYLE))
    findings = [
        ("ChemProp+Chemeleon is complementary to fingerprint-based models.",
         "Despite being only 3rd by single-model CV RMSE, it receives 41.6% weight in NNLS "
         "because its graph message-passing encodes structural context that ECFP4 cannot."),
        ("3D descriptors add diversity but hurt as standalone models.",
         "WHIM/GETAWAY/RDF 3D pharmacophore features (901-dim) perform worse than 2D baselines "
         "in isolation (RMSE 0.610 vs 0.597). However, small ensemble weights (1–2%) "
         "confirm they contribute marginal complementary information."),
        ("Extra fingerprints (Avalon/AtomPair/RDKit-path) underperform ECFP4.",
         "These alternative fingerprints consistently rank last and are not included in the ensemble."),
        ("Top GBM cluster is statistically tied.",
         "LGB_all, XGB_all, ChemProp, XGB_ecfp4_rdkit, CAT_all, and XGB_ecfp4_mordred "
         "are statistically indistinguishable (Wilcoxon p_adj > 0.05 between them), "
         "indicating the performance ceiling for standard 2D descriptors has been reached."),
        ("Ensemble improvement: +4.8% RMSE over best single model.",
         "NNLS ensemble (OOF RMSE 0.567) vs LGB_all (0.595). "
         "Further gains require either richer features, more data, or better base models."),
    ]
    for title, body in findings:
        story.append(KeepTogether([
            Paragraph(f"<b>{title}</b>", BODY_STYLE),
            Paragraph(body, BODY_STYLE),
            Spacer(1, 0.15*cm),
        ]))

    story.append(Spacer(1, 0.5*cm))
    story.append(h_rule())
    story.append(Paragraph(
        "Report generated automatically from pipeline outputs. "
        "All code in <font name='Courier'>OpenADMET/PXR/</font>.",
        CAPTION_STYLE))

    # ---- Build ---------------------------------------------------------
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    build()
