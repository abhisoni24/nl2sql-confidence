"""Cross-model summary for Checkpoint 2 (E1 size sweep, E2 labels, E3-E6 scores).

Reads every finished run listed in configs/cp2_plan.yaml and writes, per label,
results/cp2_summary/<label>/:
    models.csv            one row per model, headline numbers with 95% CIs
    per_set.csv           E4: drop and flip AUROC per perturbation set x model
    CP2_SUMMARY.md        tables for the write-up
    figures/*.png

    python scripts/compare_models.py                  # all labels found
    python scripts/compare_models.py --results-root /content/drive/MyDrive/...
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
from omegaconf import OmegaConf

from nl2sql_conf.config import load_config
from nl2sql_conf.utils import REPO_ROOT, get_logger, repo_path

log = get_logger("compare_models")

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Categorical slots in fixed order (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
CLEAN_C, PERT_C = SERIES[0], SERIES[1]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def load_runs(plan, results_root: Path, label: str) -> list[dict]:
    runs = []
    for exp in plan.e1_runs:
        cfg = load_config(exp)
        mfile = results_root / cfg.run_name / f"analysis_{label}" / "metrics.json"
        if not mfile.exists():
            log.warning("No %s results yet for %s", label, cfg.run_name)
            continue
        abl = results_root / cfg.run_name / "probe_ablation.json"
        runs.append({"cfg": cfg, "m": json.loads(mfile.read_text()),
                     "ablation": json.loads(abl.read_text()) if abl.exists() else {}})
    return runs


def model_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        cfg, m = r["cfg"], r["m"]
        p, ci = m["pooled"], m["bootstrap"]
        rb, fd, disc, cal = (p["robustness"], p["flip_detection"], p["discrimination"],
                             p["calibration"])
        row = {"model": cfg.model.hf_id.split("/")[-1], "family": cfg.model.family,
               "params_b": cfg.model.params_b, "pairs": rb["n_pairs"],
               "ex_clean": rb["ex_clean"], "ex_pert": rb["ex_pert"],
               "drop_pts": rb["ex_drop_pts"], "drop_lo": ci["ex_drop_pts"][0],
               "drop_hi": ci["ex_drop_pts"][1], "n_flips": fd["n_flips"],
               "no_fence_rate_pert": rb["truncated_rate_pert"]}
        for k, v in disc.items():
            row[f"auroc_clean_{k}"], row[f"auroc_pert_{k}"] = v["clean"], v["perturbed"]
            row[f"flip_auroc_{k}"] = fd[k]["auroc_flip_from_delta"]
            row[f"flip_lo_{k}"], row[f"flip_hi_{k}"] = ci[f"auroc_flip_from_delta_{k}"]
            row[f"blind_spot_{k}"] = fd[k]["blind_spot_rate"]
        for k, v in cal.items():
            row[f"ece_clean_{k}"], row[f"ece_pert_{k}"] = v["clean"]["ece"], v["perturbed"]["ece"]
        rows.append(row)
    return pd.DataFrame(rows)


def per_set_table(runs: list[dict], key: str) -> pd.DataFrame:
    rows = []
    for r in runs:
        name = r["cfg"].model.hf_id.split("/")[-1]
        for s, b in r["m"]["per_set"].items():
            rows.append({"model": name, "set": s, "pairs": b["robustness"]["n_pairs"],
                         "drop_pts": b["robustness"]["ex_drop_pts"],
                         "n_flips": b["flip_detection"]["n_flips"],
                         "flip_auroc": b["flip_detection"][key]["auroc_flip_from_delta"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def fig_accuracy_vs_size(df, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    _style(ax)
    q = df[df.family == "qwen2.5-coder"].sort_values("params_b")
    for col, color, lab in (("ex_clean", CLEAN_C, "clean"), ("ex_pert", PERT_C, "perturbed")):
        ax.plot(q.params_b, 100 * q[col], color=color, linewidth=2, marker="o", markersize=6,
                label=f"Qwen2.5-Coder, {lab}")
        other = df[df.family != "qwen2.5-coder"]
        ax.scatter(other.params_b, 100 * other[col], color=color, marker="D", s=50,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3,
                   label=f"DeepSeek-Coder, {lab}" if len(other) else None)
    ax.set_xscale("log")
    ax.set_xticks(df.params_b, [f"{x:g}B" for x in df.params_b])
    ax.minorticks_off()
    ax.set_ylabel("Execution accuracy (%)", color=INK)
    ax.set_xlabel("Parameters (log scale)", color=INK)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Accuracy vs model size", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_flip_vs_size(df, metrics, path):
    fig, ax = plt.subplots(figsize=(8.4, 4.0), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0.5, color=INK_2, linewidth=1, linestyle=":")
    ax.text(df.params_b.min(), 0.505, "chance", color=INK_2, fontsize=8)
    q = df[df.family == "qwen2.5-coder"].sort_values("params_b")
    for i, k in enumerate(m for m in metrics if f"flip_auroc_{m}" in df):
        color = SERIES[i % len(SERIES)]
        jitter = 1 + 0.03 * (i - len(metrics) / 2)       # keep error bars apart
        ax.errorbar(q.params_b * jitter, q[f"flip_auroc_{k}"],
                    yerr=[q[f"flip_auroc_{k}"] - q[f"flip_lo_{k}"],
                          q[f"flip_hi_{k}"] - q[f"flip_auroc_{k}"]],
                    color=color, linewidth=2, marker="o", markersize=5, capsize=2, label=k)
    ax.set_xscale("log")
    ax.set_xticks(q.params_b, [f"{x:g}B" for x in q.params_b])
    ax.minorticks_off()
    ax.set_ylabel("Flip AUROC from −Δconf (95% CI)", color=INK)
    ax.set_xlabel("Qwen2.5-Coder parameters (log scale)", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Does confidence catch rewrite-induced failures as models grow?",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_per_set(ps: pd.DataFrame, value: str, title: str, path, center: float | None):
    piv = ps.pivot(index="set", columns="model", values=value)
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 4.5, 0.45 * piv.shape[0] + 1.6),
                           facecolor=SURFACE)
    vals = piv.to_numpy(float)
    if center is None:      # sequential: one hue, light -> dark
        cmap, vmin, vmax = "Blues", np.nanmin(vals), np.nanmax(vals)
    else:                   # diverging around a neutral midpoint
        span = np.nanmax(np.abs(vals - center))
        cmap, vmin, vmax = "RdBu", center - span, center + span
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(vals):
        ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}" if center else f"{v:.1f}",
                ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def e3_section(plan, results_root: Path, label: str) -> list[str]:
    cfg = load_config(plan.e3_run)
    mfile = results_root / cfg.run_name / f"analysis_{label}" / "metrics.json"
    if not mfile.exists():
        return ["", "## E3: sampling-based confidence", "", "_Not run yet._"]
    m = json.loads(mfile.read_text())
    p, ci = m["pooled"], m["bootstrap"]
    lines = ["", f"## E3: sampling-based confidence ({cfg.model.hf_id.split('/')[-1]}, "
             f"k = {cfg.sampling.k}, "
             f"{p['robustness']['n_pairs']} pairs, {p['flip_detection']['n_flips']} flips)", "",
             "| confidence | AUROC clean | AUROC perturbed | flip AUROC from −Δ [95% CI] | blind-spot rate |",
             "|---|---|---|---|---|"]
    for k, v in p["discrimination"].items():
        fd = p["flip_detection"][k]
        lo, hi = ci[f"auroc_flip_from_delta_{k}"]
        lines.append(f"| {k} | {_f(v['clean'])} | {_f(v['perturbed'])} | "
                     f"{_f(fd['auroc_flip_from_delta'])} [{_f(lo)}, {_f(hi)}] | "
                     f"{_f(fd['blind_spot_rate'])} |")
    return lines


def write_summary(df, ps, runs, plan, results_root, label, metrics, path):
    lines = [f"# Checkpoint 2 summary (label `{label}`)", "",
             "Auto-generated by `scripts/compare_models.py` from each run's "
             "`analysis_<label>/metrics.json`. 95% CIs: cluster bootstrap over clean questions.",
             "", "## E1: robustness vs model size", "",
             "| model | pairs | EX clean | EX perturbed | drop pts [95% CI] | flips | no-fence rate (pert) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.model} | {r.pairs} | {100*r.ex_clean:.1f}% | {100*r.ex_pert:.1f}% | "
                     f"{r.drop_pts:.1f} [{r.drop_lo:.1f}, {r.drop_hi:.1f}] | {r.n_flips} | "
                     f"{100*r.no_fence_rate_pert:.1f}% |")
    lines += ["", "## E1/E5/E6: does confidence catch rewrite-induced failures?", "",
              "Flip AUROC from −Δconf among clean-correct pairs (0.5 = blind), 95% CI:", "",
              "| model | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for _, r in df.iterrows():
        cells = [f"{_f(r.get(f'flip_auroc_{k}'))} [{_f(r.get(f'flip_lo_{k}'), 2)}, "
                 f"{_f(r.get(f'flip_hi_{k}'), 2)}]" for k in metrics]
        lines.append(f"| {r.model} | " + " | ".join(cells) + " |")
    lines += ["", "AUROC for correct vs wrong, clean → perturbed:", "",
              "| model | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for _, r in df.iterrows():
        cells = [f"{_f(r.get(f'auroc_clean_{k}'))} → {_f(r.get(f'auroc_pert_{k}'))}"
                 for k in metrics]
        lines.append(f"| {r.model} | " + " | ".join(cells) + " |")
    cal = [c.removeprefix("ece_pert_") for c in df.columns if c.startswith("ece_pert_")]
    lines += ["", "ECE clean → perturbed (E6: Platt fit on clean; E5: probes fit on clean):", "",
              "| model | " + " | ".join(cal) + " |", "|---|" + "---|" * len(cal)]
    for _, r in df.iterrows():
        lines.append(f"| {r.model} | " + " | ".join(
            f"{_f(r[f'ece_clean_{k}'])} → {_f(r[f'ece_pert_{k}'])}" for k in cal) + " |")
    abl_rows = [(r["cfg"].model.hf_id.split("/")[-1], k, v) for r in runs
                for k, v in r["ablation"].items()]
    if abl_rows:
        lines += ["", "E5 linear-probe ablation (label exec_ok; AUROC clean / perturbed):", "",
                  "| model | layer, position | clean | perturbed |", "|---|---|---|---|"]
        lines += [f"| {m} | {k} | {_f(v['auroc_clean'])} | {_f(v['auroc_perturbed'])} |"
                  for m, k, v in abl_rows]
    lines += e3_section(plan, results_root, label)
    lines += ["", "## E4: by perturbation set", "", "See `per_set.csv` and "
              "`figures/figD_per_set_flip_auroc.png`, `figures/figE_per_set_drop.png`."]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="configs/cp2_plan.yaml")
    ap.add_argument("--results-root", default=None,
                    help="override paths.results_root (e.g. a Google Drive folder)")
    args = ap.parse_args()
    plan = OmegaConf.load(repo_path(args.plan))
    base = load_config(plan.e1_runs[0])
    results_root = Path(args.results_root) if args.results_root else repo_path(
        base.paths.results_root)
    metrics = list(base.analysis.confidence_metrics)
    key = base.analysis.primary_confidence

    for label in base.analysis.labels:
        runs = load_runs(plan, results_root, label)
        if not runs:
            continue
        out = results_root / plan.summary_dir / label
        (out / "figures").mkdir(parents=True, exist_ok=True)
        df, ps = model_table(runs), per_set_table(runs, key)
        df.to_csv(out / "models.csv", index=False)
        ps.to_csv(out / "per_set.csv", index=False)
        present = [k for k in metrics if f"flip_auroc_{k}" in df]
        fig_accuracy_vs_size(df, out / "figures" / "figA_accuracy_vs_size.png")
        fig_flip_vs_size(df, present, out / "figures" / "figB_flip_auroc_vs_size.png")
        fig_per_set(ps, "flip_auroc", f"E4: flip AUROC ({key}) by set and model",
                    out / "figures" / "figD_per_set_flip_auroc.png", center=0.5)
        fig_per_set(ps, "drop_pts", "E4: accuracy drop (pts) by set and model",
                    out / "figures" / "figE_per_set_drop.png", center=None)
        write_summary(df, ps, runs, plan, results_root, label, present, out / "CP2_SUMMARY.md")
        log.info("[%s] summary for %d models -> %s", label, len(runs), out)


if __name__ == "__main__":
    main()
