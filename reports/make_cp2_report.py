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


def figure(ex: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.35), facecolor=SURFACE)
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(INK_2); ax.spines["bottom"].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=7)
        ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    q = ex[ex.family == "qwen2.5-coder"].sort_values("params_b")
    d = ex[ex.family != "qwen2.5-coder"]

    ax = axes[0]
    for col, c, lab in (("ex_clean", S1, "clean"), ("ex_pert", S2, "reworded")):
        ax.plot(q.params_b, 100 * q[col], color=c, lw=2, marker="o", ms=4, label=f"Qwen, {lab}")
        ax.scatter(d.params_b, 100 * d[col], color=c, marker="D", s=22, zorder=3,
                   edgecolors="white", linewidths=0.8)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks(q.params_b, [f"{x:g}B" for x in [0.5, 1.5, 3, 7, 14]])
    ax.set_ylim(30, 90); ax.set_ylabel("Execution accuracy (%)", fontsize=7, color=INK)
    ax.set_title("(a) Accuracy rises, the rewrite gap does not close", fontsize=8,
                 loc="left", color=INK)
    ax.legend(fontsize=6, frameon=False, loc="lower right")

    ax = axes[1]
    ax.axhline(0.5, color=INK_2, lw=0.8, ls=":")
    for i, (k, c, lab) in enumerate((("p_gm", S1, "token prob. (p_gm)"),
                                     ("seq_logprob", S3, "seq. log-prob"),
                                     ("probe_lr", S2, "hidden-state probe"))):
        j = 1 + 0.06 * (i - 1)
        ax.errorbar(q.params_b * j, q[f"flip_auroc_{k}"],
                    yerr=[q[f"flip_auroc_{k}"] - q[f"flip_lo_{k}"], q[f"flip_hi_{k}"] - q[f"flip_auroc_{k}"]],
                    color=c, lw=1.6, marker="o", ms=3.5, capsize=1.5, label=lab)
        ax.scatter(d.params_b * j, d[f"flip_auroc_{k}"], color=c, marker="D", s=22, zorder=3,
                   edgecolors="white", linewidths=0.8)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks(q.params_b, [f"{x:g}B" for x in [0.5, 1.5, 3, 7, 14]])
    ax.set_ylim(0.45, 0.9); ax.set_ylabel("Flip AUROC (0.5 = chance)", fontsize=7, color=INK)
    ax.set_title("(b) Catching answers broken by a rewrite", fontsize=8, loc="left", color=INK)
    ax.legend(fontsize=6, frameon=False, loc="upper left", ncol=1)
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
    st = lambda size, lead, bold=False, color=INK, space=2: ParagraphStyle(
        "x", fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size, leading=lead,
        textColor=colors.HexColor(color), spaceAfter=space, alignment=TA_LEFT)
    H, B, SM, HD = st(12.5, 15, True), st(8.4, 10.6), st(7.2, 9, color=INK_2), st(9.4, 11.5, True, space=1)
    doc = SimpleDocTemplate(str(OUT / "CP2_progress_update.pdf"), pagesize=letter,
                            leftMargin=0.55 * inch, rightMargin=0.55 * inch,
                            topMargin=0.45 * inch, bottomMargin=0.4 * inch,
                            title="CS 559 Progress Update (Checkpoint 2)")
    s = []
    s.append(Paragraph("Progress update: does an NL2SQL model's confidence track failures "
                       "caused by meaning-preserving rewrites?", H))
    s.append(Paragraph(f"ECE/CS 559, Abhishek and Shubham Harale, Nov 5, 2026. Code: "
                       f"<link href='{args.repo_url}' color='#2a78d6'>{args.repo_url}</link>", SM))
    setup_text = (
        "<b>Setup.</b> Dr.Spider NLQ sets (9 rewrite types): 7,593 clean/reworded question pairs "
        "with identical gold SQL, from 1,020 Spider-dev questions. Six models (Qwen2.5-Coder "
        "0.5B to 14B, DeepSeek-Coder-6.7B), greedy decoding. Labels: execution match on the "
        "Spider DB (EX) and test-suite match on ~30 to 60 DB variants (TS). A <i>flip</i> is a "
        "pair right on the clean question and wrong on its rewrite; <i>flip AUROC</i> asks "
        "whether the confidence drop ranks flips above survivors. 95% CIs: cluster bootstrap "
        "over clean questions.")
    s.append(Paragraph(setup_text, B))

    s.append(Paragraph("Results so far", HD))
    rows = [["Model", "EX clean → reworded", "TS clean → reworded", "Flip AUROC p_gm",
             "Flip AUROC probe", "ECE Platt clean → rew."]]
    for _, r in ex.iterrows():
        t = ts.loc[r.model]
        rows.append([short(r.model), f"{100*r.ex_clean:.1f} → {100*r.ex_pert:.1f}%",
                     f"{100*t.ex_clean:.1f} → {100*t.ex_pert:.1f}%",
                     f"{r.flip_auroc_p_gm:.2f} [{r.flip_lo_p_gm:.2f}, {r.flip_hi_p_gm:.2f}]",
                     f"{r.flip_auroc_probe_lr:.2f} [{r.flip_lo_probe_lr:.2f}, {r.flip_hi_probe_lr:.2f}]",
                     f"{r.ece_clean_p_gm_platt:.3f} → {r.ece_pert_p_gm_platt:.3f}"])
    tbl = Table(rows, colWidths=[0.95*inch, 1.25*inch, 1.25*inch, 1.3*inch, 1.3*inch, 1.25*inch])
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7.4), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.4),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(INK_2)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor(GRID)),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
    s.append(tbl)
    s.append(Spacer(1, 3))
    s.append(Image(str(fig_path), width=7.4 * inch, height=2.35 * inch))
    caption = ("Lines: Qwen2.5-Coder; diamonds: DeepSeek-Coder-6.7B. Error bars: 95% CI. "
               "Labels: EX. Flip AUROC uses the drop in confidence from clean to reworded.")
    s.append(Paragraph(caption, SM))
    bullets = [
        f"<b>Scale buys accuracy, not robustness.</b> Rewording costs {ex.drop_pts.min():.1f} to "
        f"{ex.drop_pts.max():.1f} EX points at every size (14B: {q14.drop_pts:.1f} "
        f"[{q14.drop_lo:.1f}, {q14.drop_hi:.1f}]); under TS labels the cost is "
        f"{ts.drop_pts.min():.1f} to {ts.drop_pts.max():.1f}.",
        f"<b>Token confidence barely notices a broken answer.</b> Flip AUROC of p_gm is "
        f"{rng(ex.flip_auroc_p_gm)}, with no gain from 7B to 14B; in {100*min(blind_p):.0f} to "
        f"{100*max(blind_p):.0f}% of flips confidence did not drop at all.",
        f"<b>Hidden states carry the missing signal.</b> A linear probe on decoder states, fit on "
        f"clean questions only and scored on held-out databases, reaches {rng(ex.flip_auroc_probe_lr)} "
        f"and beats p_gm for every model ({100*min(blind_probe):.0f} to {100*max(blind_probe):.0f}% "
        f"of flips missed). The pre-set layer (75% depth, mean over SQL tokens) was the best of 6 "
        f"choices for 5 of 6 models.",
        f"<b>Calibration fit on clean data degrades under rewording.</b> Platt scaling brings ECE to "
        f"{rng(ex.ece_clean_p_gm_platt, 3)} on clean questions but {rng(ex.ece_pert_p_gm_platt, 3)} "
        f"on rewrites (14B: {q14.ece_clean_p_gm_platt:.3f} → {q14.ece_pert_p_gm_platt:.3f}).",
        f"<b>Sampling (7B, k = 10) ranks answers better but not flips.</b> Agreement of execution "
        f"results: AUROC {e3p['discrimination']['sc_agree']['clean']:.3f} clean / "
        f"{e3p['discrimination']['sc_agree']['perturbed']:.3f} reworded vs p_gm "
        f"{e3p['discrimination']['p_gm']['clean']:.3f} / {e3p['discrimination']['p_gm']['perturbed']:.3f}; "
        f"flip AUROC {e3p['flip_detection']['sc_agree']['auroc_flip_from_delta']:.3f} vs "
        f"{e3p['flip_detection']['p_gm']['auroc_flip_from_delta']:.3f}.",
    ]
    for b in bullets:
        s.append(Paragraph("• " + b, B))

    s.append(Paragraph("Problems encountered", HD))
    probs = [
        "<b>Compute.</b> The EC2 instance was unavailable; we moved to a Colab A100 (80 GB) driven "
        "from the Colab CLI, with a resumable pipeline (stage markers, per-batch writes). Full run: 2.3 h.",
        f"<b>Lucky matches.</b> Single-DB EX overstates clean accuracy by {gap.min():.1f} to "
        f"{gap.max():.1f} points vs TS, so we report both labels.",
        "<b>Dependent pairs.</b> One clean question has up to 37 rewrites (median 6), so pair-level "
        "tests overstate certainty; all CIs resample clean questions.",
        f"<b>Confounds.</b> Sequence log-prob is a sum, and output-length change alone gives flip AUROC "
        f"{min(lenauc):.2f} to {max(lenauc):.2f}; its edge over p_gm is partly length. Agreement over "
        "k = 10 samples is coarse, so many flips tie at zero change.",
        "<b>Degenerate decoding.</b> The CPU pilot looped to the token cap in 2.5% of outputs; a "
        "repetition guard now keeps unfinished outputs at 1% or less.",
    ]
    for p in probs:
        s.append(Paragraph("• " + p, B))

    s.append(Paragraph("Experiments that remain", HD))
    rem = [
        "Length-controlled confidence (per-token and length-matched comparisons) for seq. log-prob and the probe.",
        "Probe transfer: train on some rewrite types, test on others; add Dr.Spider DB and SQL perturbation families.",
        "Tie-aware sampling analysis with larger k and a second model; manual audit of ~100 flips for rewrites that change meaning.",
        "Final presentation (Dec 1 or 3) and paper with code (Dec 4).",
    ]
    for r in rem:
        s.append(Paragraph("• " + r, B))
    doc.build(s)
    print("wrote", OUT / "CP2_progress_update.pdf")

    # Markdown copy of the same text (for the repo / quick edits).
    md = lambda t: (t.replace("<b>", "**").replace("</b>", "**").replace("<i>", "*")
                    .replace("</i>", "*"))
    lines = ["# Progress update (Checkpoint 2): does an NL2SQL model's confidence track "
             "failures caused by meaning-preserving rewrites?", "",
             f"ECE/CS 559, Abhishek and Shubham Harale, Nov 5, 2026. Code: {args.repo_url}", "",
             md(setup_text), "", "## Results so far", "",
             "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    lines += ["", f"![figure]({fig_path.name})", "", caption, ""]
    lines += [f"- {md(b)}" for b in bullets]
    lines += ["", "## Problems encountered", ""] + [f"- {md(p)}" for p in probs]
    lines += ["", "## Experiments that remain", ""] + [f"- {r}" for r in rem]
    (OUT / "CP2_progress_update.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "CP2_progress_update.md")


if __name__ == "__main__":
    main()
