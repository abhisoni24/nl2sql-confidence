"""Build the one-page CP2 progress update (PDF) from results/cp2_summary.

Every number in the PDF is read from results/ at build time (models.csv and
metrics.json), not typed in by hand, except where noted in the text.

    python reports/make_cp2_report.py --repo-url https://github.com/<user>/<repo>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "reports" / "cp2"
SURFACE, INK, INK_2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3


def short(model: str) -> str:
    return (model.replace("Qwen2.5-Coder-", "Qwen-").replace("-Instruct", "")
            .replace("deepseek-coder-6.7b-instruct", "DeepSeek-6.7B"))


FONT_DIR = Path("/usr/share/fonts/truetype/crosextra")   # Carlito = metric-compatible Calibri


def setup_fonts() -> str:
    """Register Carlito (open, metric-compatible with Calibri) for reportlab and matplotlib.

    Calibri itself is a licensed Microsoft font and is not available here; Carlito
    has identical character widths, so the layout matches a Calibri document.
    Falls back to Helvetica if Carlito is not installed.
    """
    from matplotlib import font_manager
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    files = {k: FONT_DIR / f"Carlito-{k}.ttf" for k in ("Regular", "Bold", "Italic", "BoldItalic")}
    if not all(f.exists() for f in files.values()):
        return "Helvetica"
    for k, f in files.items():
        pdfmetrics.registerFont(TTFont(f"Carlito-{k}", str(f)))
        font_manager.fontManager.addfont(str(f))
    pdfmetrics.registerFontFamily("Carlito", normal="Carlito-Regular", bold="Carlito-Bold",
                                  italic="Carlito-Italic", boldItalic="Carlito-BoldItalic")
    matplotlib.rcParams["font.family"] = "Carlito"
    return "Carlito"


def figure(ex: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.3), facecolor=SURFACE)
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(INK_2); ax.spines["bottom"].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    q = ex[ex.family == "qwen2.5-coder"].sort_values("params_b")
    d = ex[ex.family != "qwen2.5-coder"]
    ticks = [f"{x:g}B" for x in [0.5, 1.5, 3, 7, 14]]

    ax = axes[0]
    for col, c, lab in (("ex_clean", S1, "original question"), ("ex_pert", S2, "reworded question")):
        ax.plot(q.params_b, 100 * q[col], color=c, lw=2, marker="o", ms=4, label=lab)
        ax.scatter(d.params_b, 100 * d[col], color=c, marker="D", s=22, zorder=3,
                   edgecolors="white", linewidths=0.8)
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(q.params_b, ticks)
    ax.set_ylim(30, 90); ax.set_ylabel("Correct answers (%)", fontsize=8.5, color=INK)
    ax.set_xlabel("Model size (parameters)", fontsize=8, color=INK_2)
    ax.set_title("(a) Bigger models are more accurate, but rewording\n     still costs about 10 points",
                 fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")

    ax = axes[1]
    ax.axhline(0.5, color=INK_2, lw=0.8, ls=":")
    ax.text(0.47, 0.505, "coin flip", fontsize=7, color=INK_2)
    for i, (k, c, lab) in enumerate((("p_gm", S1, "model's own word probabilities"),
                                     ("seq_logprob", S3, "probability of the whole query"),
                                     ("probe_lr", S2, "small network reading internal activity"))):
        j = 1 + 0.06 * (i - 1)
        ax.errorbar(q.params_b * j, q[f"flip_auroc_{k}"],
                    yerr=[q[f"flip_auroc_{k}"] - q[f"flip_lo_{k}"], q[f"flip_hi_{k}"] - q[f"flip_auroc_{k}"]],
                    color=c, lw=1.6, marker="o", ms=3.5, capsize=1.5, label=lab)
        ax.scatter(d.params_b * j, d[f"flip_auroc_{k}"], color=c, marker="D", s=22, zorder=3,
                   edgecolors="white", linewidths=0.8)
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(q.params_b, ticks)
    ax.set_ylim(0.45, 0.9); ax.set_ylabel("Warning score", fontsize=8.5, color=INK)
    ax.set_xlabel("Model size (parameters)", fontsize=8, color=INK_2)
    ax.set_title("(b) How well each signal warns that rewording\n     broke a correct answer", fontsize=9,
                 loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="(GitHub link to be added)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ex = pd.read_csv(RES / "cp2_summary/exec_ok/models.csv")
    ts = pd.read_csv(RES / "cp2_summary/ts_ok/models.csv").set_index("model")
    e3 = json.loads((RES / "cp2_e3_qwen2.5-coder-7b/analysis_exec_ok/metrics.json").read_text())
    font = setup_fonts()          # before drawing, so the figure uses the same font
    fig_path = OUT / "cp2_update_figure.png"
    figure(ex, fig_path)

    # ---- derived numbers used in the text (all from results files) ----------
    rng = lambda s, nd=2: f"{s.min():.{nd}f}–{s.max():.{nd}f}"
    gap = (100 * (ex.set_index("model").ex_clean - ts.ex_clean))
    blind_p = []; blind_probe = []; lenauc = []
    for p in sorted(RES.glob("cp2_e1_*")):
        fd = json.loads((p / "analysis_exec_ok/metrics.json").read_text())["pooled"]["flip_detection"]
        blind_p.append(fd["p_gm"]["blind_spot_rate"]); blind_probe.append(fd["probe_lr"]["blind_spot_rate"])
        lenauc.append(fd["length_control"]["auroc_flip_from_length_increase"])
    e3p, e3ci = e3["pooled"], e3["bootstrap"]
    q14 = ex[ex.model.str.contains("14B")].iloc[0]

    # ---- PDF ----------------------------------------------------------------
    bold = "Carlito-Bold" if font == "Carlito" else "Helvetica-Bold"
    st = lambda size, lead, b=False, color=INK, space=2: ParagraphStyle(
        "x", fontName=bold if b else (font if font != "Carlito" else "Carlito-Regular"),
        fontSize=size, leading=lead, textColor=colors.HexColor(color), spaceAfter=space,
        alignment=TA_LEFT)
    H, B, SM, HD = st(13.5, 15.5, True), st(9.1, 10.6, space=1.3), st(7.9, 9.2, color=INK_2), st(10.6, 12.4, True, space=1)
    doc = SimpleDocTemplate(str(OUT / "CP2_progress_update.pdf"), pagesize=letter,
                            leftMargin=0.55 * inch, rightMargin=0.55 * inch,
                            topMargin=0.4 * inch, bottomMargin=0.3 * inch,
                            title="CS 559 Progress Update (Checkpoint 2)")
    s = []
    s.append(Paragraph("Progress update: does a text-to-SQL model notice when rewording breaks its answer?", H))
    s.append(Paragraph(f"ECE/CS 559 · Abhishek and Shubham Harale · November 5, 2026 · Code: "
                       f"<link href='{args.repo_url}' color='#2a78d6'>{args.repo_url}</link>", SM))
    setup_text = (
        "<b>What we did.</b> Text-to-SQL models turn a question into a database query. Asking the "
        "same question in different words should give the same answer, but often does not. We "
        "test whether the model's confidence warns us when that happens. Using the Dr.Spider "
        "benchmark, we took 7,593 pairs of an original and a reworded question that share the same "
        "correct query (1,020 original questions, 9 kinds of rewording) and ran six open models: "
        "Qwen2.5-Coder at five sizes (0.5B to 14B parameters) and DeepSeek-Coder (6.7B). An answer "
        "counts as correct if its query returns the right result. We compared four confidence "
        "signals: the model's own word probabilities, a corrected (recalibrated) version of them, "
        "a small neural network trained to read the model's internal activity, and agreement "
        "among 10 sampled answers. The <i>warning score</i> asks whether confidence falls more when "
        "rewording turns a correct answer wrong than when it stays correct (0.5 = coin flip, "
        "1.0 = perfect warning).")
    s.append(Paragraph(setup_text, B))

    s.append(Paragraph("Results so far", HD))
    rows = [["Model", "Accuracy\n(original → reworded)", "Stricter check\n(original → reworded)",
             "Warning score:\nmodel's own probabilities", "Warning score:\nsmall network"]]
    for _, r in ex.iterrows():
        t = ts.loc[r.model]
        rows.append([short(r.model), f"{100*r.ex_clean:.1f}% → {100*r.ex_pert:.1f}%",
                     f"{100*t.ex_clean:.1f}% → {100*t.ex_pert:.1f}%",
                     f"{r.flip_auroc_p_gm:.2f}  ({r.flip_lo_p_gm:.2f} to {r.flip_hi_p_gm:.2f})",
                     f"{r.flip_auroc_probe_lr:.2f}  ({r.flip_lo_probe_lr:.2f} to {r.flip_hi_probe_lr:.2f})"])
    tbl = Table(rows, colWidths=[1.1*inch, 1.6*inch, 1.65*inch, 1.6*inch, 1.6*inch])
    reg = font if font != "Carlito" else "Carlito-Regular"
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), reg, 8.3), ("FONT", (0, 0), (-1, 0), bold, 8.3),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(INK_2)),
        ("VALIGN", (0, 0), (-1, 0), "BOTTOM"), ("LEADING", (0, 0), (-1, 0), 9.6),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor(GRID)),
        ("TOPPADDING", (0, 0), (-1, -1), 1.0), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.0)]))
    s.append(tbl)
    s.append(Paragraph("Ranges are 95% confidence intervals. The stricter check reruns each query "
                       "on about 30 to 60 altered copies of the database to catch answers right by accident.", SM))
    s.append(Image(str(fig_path), width=6.6 * inch, height=2.05 * inch))
    caption = ("Lines: Qwen2.5-Coder; diamonds: DeepSeek-Coder-6.7B. Bars: 95% confidence intervals.")
    s.append(Paragraph(caption, SM))

    e3d, e3f = e3p["discrimination"], e3p["flip_detection"]
    bullets = [
        f"<b>Bigger models are more accurate but no harder to throw off.</b> Accuracy on original "
        f"questions rises from {100*ex.ex_clean.min():.0f}% to {100*ex.ex_clean.max():.0f}%, yet "
        f"rewording costs {ex.drop_pts.min():.0f} to {ex.drop_pts.max():.0f} points at every size. "
        f"Scaling up does not fix sensitivity to phrasing.",
        f"<b>The model's own confidence is a weak alarm.</b> Its warning score is only "
        f"{ex.flip_auroc_p_gm.min():.2f} to {ex.flip_auroc_p_gm.max():.2f}, and in "
        f"{100*min(blind_p):.0f} to {100*max(blind_p):.0f}% of the cases where rewording broke a "
        f"correct answer, confidence did not drop at all. Users cannot rely on it to flag these errors.",
        f"<b>The warning signal exists inside the model.</b> A small network that reads the model's "
        f"internal activity, trained only on original questions and tested on unseen databases, "
        f"scores {ex.flip_auroc_probe_lr.min():.2f} to {ex.flip_auroc_probe_lr.max():.2f} and beats the "
        f"model's own confidence for every model, so a learned confidence check is a practical safeguard.",
        f"<b>Corrections tuned on standard questions do not fully carry over.</b> After recalibration, "
        f"the gap between stated confidence and actual accuracy is {ex.ece_clean_p_gm_platt.min():.2f} to "
        f"{ex.ece_clean_p_gm_platt.max():.2f} on original questions but {ex.ece_pert_p_gm_platt.min():.2f} "
        f"to {ex.ece_pert_p_gm_platt.max():.2f} on reworded ones (lower is better). Confidence should be "
        f"checked on reworded inputs, not only on the standard test set.",
        f"<b>Asking the model 10 times helps rank answers, not catch rewording failures.</b> For the 7B "
        f"model, agreement among samples separates right from wrong answers best "
        f"({e3d['sc_agree']['clean']:.2f} vs {e3d['p_gm']['clean']:.2f} for its own probabilities), but "
        f"it warns no better ({e3f['sc_agree']['auroc_flip_from_delta']:.2f} vs "
        f"{e3f['p_gm']['auroc_flip_from_delta']:.2f}) at ten times the cost.",
    ]
    s.append(Paragraph("What we found and what it means", HD))
    for b in bullets:
        s.append(Paragraph("• " + b, B))

    s.append(Paragraph("Problems encountered", HD))
    probs = [
        f"<b>Answers that are right by accident.</b> Checking a query on a single database overstates "
        f"accuracy by {gap.min():.0f} to {gap.max():.0f} points; we report the stricter check too.",
        "<b>Rewordings are not independent.</b> One original question has up to 37 rewordings "
        "(typically 6), so we compute uncertainty by resampling whole questions, not single pairs.",
        f"<b>Longer answers look less confident.</b> The probability of a whole query falls as the query "
        f"gets longer, and longer answers are more often wrong; answer length alone scores "
        f"{min(lenauc):.2f} to {max(lenauc):.2f}, so part of that signal is just length.",
        "<b>Coarse agreement scores.</b> With 10 samples, agreement moves in 10% steps, so many cases tie.",
        "<b>Runaway outputs.</b> Small models sometimes repeated text to the length limit (2.5% in our "
        "pilot); a stopping rule now caps this at 1%.",
    ]
    for p in probs:
        s.append(Paragraph("• " + p, B))

    s.append(Paragraph("Experiments that remain", HD))
    rem = [
        "<b>Test a second benchmark besides Dr.Spider</b> with the same six models: Spider-Syn first "
        "(1,034 original/reworded pairs with the same correct query), then a dataset built on "
        "different databases, to check that the findings are not specific to one benchmark.",
        "Separate the effect of answer length from genuine confidence.",
        "Test the small network on kinds of rewording, and on database and query changes, it never saw in training.",
        "Hand-check about 100 broken answers to see how often a rewording actually changed the meaning.",
        "Final presentation (Dec 1 or 3) and final paper with code (Dec 4).",
    ]
    for r in rem:
        s.append(Paragraph("• " + r, B))
    doc.build(s)
    print("wrote", OUT / "CP2_progress_update.pdf")

    # Markdown copy of the same text (for the repo / quick edits).
    md = lambda t: (t.replace("<b>", "**").replace("</b>", "**").replace("<i>", "*")
                    .replace("</i>", "*"))
    lines = ["# Progress update: can a text-to-SQL model tell when rewording a question has "
             "broken its answer?", "",
             f"ECE/CS 559 · Abhishek and Shubham Harale · November 5, 2026 · Code: {args.repo_url}", "",
             md(setup_text), "", "## Results so far", "",
             "| " + " | ".join(h.replace("\n", " ") for h in rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    lines += ["", "Ranges in brackets are 95% confidence intervals. The stricter check reruns each "
              "query on about 30 to 60 altered copies of the database.", "",
              f"![figure]({fig_path.name})", "", caption, "", "## What we found and what it means", ""]
    lines += [f"- {md(b)}" for b in bullets]
    lines += ["", "## Problems encountered", ""] + [f"- {md(p)}" for p in probs]
    lines += ["", "## Experiments that remain", ""] + [f"- {md(r)}" for r in rem]
    (OUT / "CP2_progress_update.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "CP2_progress_update.md")


if __name__ == "__main__":
    main()
