"""Step 5: paired confidence-vs-robustness analysis, tables, figures, summary.

Unit of analysis is the **pair**: each (clean, perturbed) twin contributes one
clean and one perturbed observation, so every comparison is paired.

Inputs   results/<run>/labeled.jsonl (+ any results/<run>/scores_*.jsonl, merged by uid)
         data/processed/<run>/pairs.jsonl
Outputs  per correctness label (exec_ok, ts_ok; ``analysis.labels``):
         results/<run>/analysis_<label>/{metrics.json, pairs_scored.csv, figures/, SUMMARY.md}
         CP1 configs (no ``analysis.labels``) keep the CP1 layout: files directly in results/<run>/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _cli import parse
from nl2sql_conf import metrics as M
from nl2sql_conf.config import processed_dir, run_dir, save_resolved_config
from nl2sql_conf.utils import get_logger, read_jsonl

log = get_logger("analyze")

# Transition labels: clean outcome -> perturbed outcome (R = right, W = wrong).
TRANSITIONS = ["R→R", "R→W", "W→R", "W→W"]

# Figure styling (validated reference palette, light mode; thin marks, quiet axes).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CLEAN_C, PERT_C = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
TRANS_C = {"R→R": "#2a78d6", "R→W": "#eb6834", "W→R": "#1baf7a", "W→W": "#4a3aa7"}


class Spec:
    """What to analyze in one pass: label, confidence scores, output folder."""

    def __init__(self, cfg, label: str, metrics: list[str], calibrated: list[str], out: Path):
        self.cfg, self.label, self.metrics, self.calibrated, self.out = (
            cfg, label, metrics, calibrated, out)
        a = cfg.analysis
        self.primary = a.primary_confidence
        self.bins = a.ece_bins
        self.n_boot = a.bootstrap_resamples
        self.cluster_key = a.get("cluster_key")


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def load_rows(cfg) -> dict[str, dict]:
    """labeled.jsonl merged with every scores_*.jsonl (extra confidence scores)."""
    out = run_dir(cfg)
    by_uid = {r["uid"]: r for r in read_jsonl(out / "labeled.jsonl")}
    for f in sorted(out.glob("scores_*.jsonl")):
        for r in read_jsonl(f):
            if r["uid"] in by_uid:
                by_uid[r["uid"]].update({k: v for k, v in r.items() if k != "uid"})
        log.info("Merged scores from %s", f.name)
    return by_uid


def build_pair_table(cfg, by_uid: dict, label: str, metrics: list[str]) -> pd.DataFrame:
    """One row per pair with clean (_c) and perturbed (_p) outcomes and confidences."""
    keys = list(metrics) + ["n_tokens"]
    rows, skipped = [], 0
    for p in read_jsonl(processed_dir(cfg) / "pairs.jsonl"):
        c, q = by_uid.get(p["clean_uid"]), by_uid.get(p["perturbed_uid"])
        if c is None or q is None:          # inference not finished for this pair
            skipped += 1
            continue
        row = {"pair_id": p["pair_id"], "set": p["set_name"].removeprefix("NLQ_"),
               "spider_qid": p["spider_qid"], "db_id": p["db_id"],
               "ok_c": bool(c[label]), "ok_p": bool(q[label]),
               "err_c": c["exec_error"] is not None, "err_p": q["exec_error"] is not None,
               "same_pred": c["pred_sql"].strip() == q["pred_sql"].strip(),
               # Decode ended without a closing fence (token cap or loop guard).
               "trunc_c": not c["hit_stop"], "trunc_p": not q["hit_stop"],
               "q_c": c["question"], "q_p": q["question"],
               "pred_c": c["pred_sql"], "pred_p": q["pred_sql"], "gold": c["gold_sql"]}
        for k in keys:
            row[f"{k}_c"], row[f"{k}_p"] = c.get(k), q.get(k)
        rows.append(row)
    if skipped:
        log.warning("%d pairs skipped (missing predictions)", skipped)
    df = pd.DataFrame(rows)
    for k in keys:  # missing scores (None) -> NaN
        df[f"{k}_c"] = pd.to_numeric(df[f"{k}_c"], errors="coerce")
        df[f"{k}_p"] = pd.to_numeric(df[f"{k}_p"], errors="coerce")
    df["transition"] = [f"{'R' if a else 'W'}→{'R' if b else 'W'}"
                        for a, b in zip(df.ok_c, df.ok_p)]
    for k in metrics:
        df[f"d_{k}"] = df[f"{k}_p"] - df[f"{k}_c"]   # Δconf = perturbed - clean
    return df


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def robustness(df: pd.DataFrame) -> dict:
    return {"n_pairs": len(df), "n_clean_questions": int(df.spider_qid.nunique()),
            "ex_clean": df.ok_c.mean(), "ex_pert": df.ok_p.mean(),
            "ex_drop_pts": 100 * (df.ok_c.mean() - df.ok_p.mean()),
            "exec_error_rate_clean": df.err_c.mean(), "exec_error_rate_pert": df.err_p.mean(),
            "truncated_rate_clean": df.trunc_c.mean(), "truncated_rate_pert": df.trunc_p.mean(),
            "identical_prediction_rate": df.same_pred.mean(),
            # Exact McNemar treats pairs as independent; the cluster-bootstrap CI on
            # the drop (bootstrap block) is the dependence-robust check.
            "mcnemar": M.mcnemar_exact(df.ok_c, df.ok_p),
            "transitions": {t: int((df.transition == t).sum()) for t in TRANSITIONS}}


def discrimination(df: pd.DataFrame, spec: Spec) -> dict:
    """AUROC of each confidence score for separating correct from wrong outputs."""
    return {k: {"clean": M.auroc(df[f"{k}_c"], df.ok_c),
                "perturbed": M.auroc(df[f"{k}_p"], df.ok_p)} for k in spec.metrics}


def calibration(df: pd.DataFrame, spec: Spec) -> dict:
    out = {}
    for k in spec.calibrated:
        out[k] = {}
        for side, ok in (("clean", "ok_c"), ("perturbed", "ok_p")):
            conf = df[f"{k}_{side[0]}"].to_numpy()
            out[k][side] = {"mean_conf": float(np.nanmean(conf)), "accuracy": df[ok].mean(),
                            "ece": M.ece(conf, df[ok], spec.bins),
                            "brier": M.brier(conf, df[ok]),
                            "bins": M.reliability_bins(conf, df[ok], spec.bins)}
    return out


def flip_detection(df: pd.DataFrame, spec: Spec) -> dict:
    """Core test: among clean-correct pairs, does confidence drop when the rewrite fails?

    Positive class = R→W (flip). Score = -Δconf (a larger drop should mean a flip).
    AUROC 0.5 means confidence change carries no information about the flip.
    """
    base = df[df.ok_c]
    flip = (~base.ok_p).to_numpy().astype(int)
    out = {"n_clean_correct": len(base), "n_flips": int(flip.sum())}
    for k in spec.metrics:
        d = base[f"d_{k}"].to_numpy()
        conf_p = base[f"{k}_p"].to_numpy()
        rr, rw = d[flip == 0], d[flip == 1]
        out[k] = {
            "mean_delta_RR": float(np.nanmean(rr)) if len(rr) else None,
            "mean_delta_RW": float(np.nanmean(rw)) if len(rw) else None,
            "median_delta_RR": float(np.nanmedian(rr)) if len(rr) else None,
            "median_delta_RW": float(np.nanmedian(rw)) if len(rw) else None,
            "auroc_flip_from_delta": M.auroc(-d, flip),
            # Same question using only perturbed-side confidence (no clean twin needed).
            "auroc_flip_from_pert_conf": M.auroc(-conf_p, flip),
            "mannwhitney_p": M.mann_whitney(rw, rr),
            # Blind spot: flips where confidence did not go down at all.
            "blind_spot_rate": float(np.mean(rw[~np.isnan(rw)] >= 0)) if len(rw) else None,
        }
    # Length control: seq_logprob is a sum over tokens, so it falls whenever the
    # output gets longer. If output-length change alone predicts flips as well as
    # Δseq_logprob does, the seq_logprob signal is mostly a length effect.
    dlen = (base.n_tokens_p - base.n_tokens_c).to_numpy()
    out["length_control"] = {
        "mean_dlen_RR": float(np.nanmean(dlen[flip == 0])) if (flip == 0).any() else None,
        "mean_dlen_RW": float(np.nanmean(dlen[flip == 1])) if flip.any() else None,
        "auroc_flip_from_length_increase": M.auroc(dlen, flip),
    }
    return out


def bootstrap(df: pd.DataFrame, spec: Spec) -> dict:
    """95% CIs for the headline numbers (cluster bootstrap if cluster_key is set)."""
    n, seed = spec.n_boot, spec.cfg.seed
    g_all = df[spec.cluster_key].to_numpy() if spec.cluster_key else None
    ok_c, ok_p = df.ok_c.to_numpy(float), df.ok_p.to_numpy(float)
    base = df[df.ok_c]
    g_base = base[spec.cluster_key].to_numpy() if spec.cluster_key else None
    flip = (~base.ok_p).to_numpy().astype(int)
    out = {"cluster_key": spec.cluster_key,
           "ex_drop_pts": M.bootstrap_ci(lambda x, y: 100 * (x.mean() - y.mean()),
                                         [ok_c, ok_p], n, seed, groups=g_all)}
    for m in spec.metrics:
        out[f"auroc_{m}_clean"] = M.bootstrap_ci(
            M.auroc, [df[f"{m}_c"].to_numpy(), ok_c], n, seed, groups=g_all)
        out[f"auroc_{m}_perturbed"] = M.bootstrap_ci(
            M.auroc, [df[f"{m}_p"].to_numpy(), ok_p], n, seed, groups=g_all)
        out[f"auroc_flip_from_delta_{m}"] = M.bootstrap_ci(
            lambda s, y: M.auroc(-s, y), [base[f"d_{m}"].to_numpy(), flip], n, seed,
            groups=g_base)
    dlen = (base.n_tokens_p - base.n_tokens_c).to_numpy()
    out["auroc_flip_from_length_increase"] = M.bootstrap_ci(
        M.auroc, [dlen, flip], n, seed, groups=g_base)
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(w: float, h: float, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax in axes:
        _style(ax)
    return fig, axes


def fig_accuracy(per_set: dict, path) -> None:
    names = list(per_set)
    clean = [100 * per_set[s]["robustness"]["ex_clean"] for s in names]
    pert = [100 * per_set[s]["robustness"]["ex_pert"] for s in names]
    x, w = np.arange(len(names)), 0.36
    fig, (ax,) = _figure(max(7.2, 1.1 * len(names) + 2), 3.8)
    for off, vals, color, label in ((-w / 2, clean, CLEAN_C, "clean"),
                                    (w / 2, pert, PERT_C, "perturbed")):
        bars = ax.bar(x + off, vals, w - 0.03, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=8, color=INK_2)
    ax.axvline(len(names) - 1.5, color=GRID, linewidth=1)   # separates "pooled"
    ax.set_xticks(x, [f"{n.replace('_', chr(10))}\nn={per_set[n]['robustness']['n_pairs']}"
                      for n in names], color=INK, fontsize=8)
    ax.set_ylabel("Execution accuracy (%)", color=INK)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Accuracy on clean vs perturbed twins", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_reliability(cal: dict, path) -> None:
    keys = list(cal)
    fig, axes = _figure(4.2 * len(keys), 4.2, ncols=len(keys))
    for ax, key in zip(axes, keys):
        ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=":", label="perfect")
        for side, color in (("clean", CLEAN_C), ("perturbed", PERT_C)):
            pts = [(b["mean_conf"], b["accuracy"]) for b in cal[key][side]["bins"] if b["n"]]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                    label=f"{side} (ECE {cal[key][side]['ece']:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Confidence ({key}), binned", color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_title(key, loc="left", color=INK, fontsize=10)
    axes[0].set_ylabel("Accuracy in bin", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_delta(df: pd.DataFrame, key: str, path) -> None:
    fig, (ax,) = _figure(6.4, 3.8)
    rng = np.random.default_rng(0)  # jitter only; does not touch results
    data = [df.loc[df.transition == t, f"d_{key}"].dropna().to_numpy() for t in TRANSITIONS]
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.boxplot(data, positions=range(4), widths=0.5, showfliers=False,
               medianprops={"color": INK, "linewidth": 1.5},
               boxprops={"color": INK_2}, whiskerprops={"color": INK_2},
               capprops={"color": INK_2})
    for i, (t, d) in enumerate(zip(TRANSITIONS, data)):
        if len(d) > 400:  # thin dense strips so the boxes stay readable
            d = rng.choice(d, 400, replace=False)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=10, color=TRANS_C[t],
                   alpha=0.6, edgecolors=SURFACE, linewidths=0.4, zorder=3)
    counts = [int((df.transition == t).sum()) for t in TRANSITIONS]
    ax.set_xticks(range(4), [f"{t}\nn={c}" for t, c in zip(TRANSITIONS, counts)], color=INK)
    ax.set_ylabel(f"Δ{key} = perturbed − clean", color=INK)
    ax.set_xlabel("Outcome transition (clean → perturbed; R = right, W = wrong)",
                  color=INK_2, fontsize=9)
    ax.set_title("Confidence change by outcome transition", loc="left", color=INK,
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_conf_hist(df: pd.DataFrame, key: str, path) -> None:
    fig, axes = _figure(8.4, 3.4, ncols=2)
    vals = df[[f"{key}_c", f"{key}_p"]].to_numpy()
    bins = np.linspace(np.nanmin(vals), np.nanmax(vals), 21)
    for ax, side, ok in ((axes[0], "c", "ok_c"), (axes[1], "p", "ok_p")):
        for flag, color, label in ((True, CLEAN_C, "correct"), (False, PERT_C, "wrong")):
            # Outline histograms so the two distributions stay readable where they overlap.
            ax.hist(df.loc[df[ok] == flag, f"{key}_{side}"].dropna(), bins=bins, color=color,
                    histtype="step", linewidth=2, label=label)
        ax.set_title("clean" if side == "c" else "perturbed", loc="left", color=INK,
                     fontsize=10)
        ax.set_xlabel(key, color=INK)
    axes[0].set_ylabel("pairs", color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _ci(ci, nd=3):
    return f"[{_f(ci[0], nd)}, {_f(ci[1], nd)}]"


def write_summary(res: dict, spec: Spec, path) -> None:
    cfg, k = spec.cfg, spec.primary
    pooled, ci = res["pooled"], res["bootstrap"]
    r, fd = pooled["robustness"], pooled["flip_detection"]
    boot = (f"cluster bootstrap over `{spec.cluster_key}`" if spec.cluster_key
            else "pair bootstrap")
    lines = [
        f"# {cfg.run_name}: auto-generated results summary (label `{spec.label}`)",
        "",
        f"Model `{cfg.model.hf_id}` (rev `{cfg.model.revision[:8]}`), greedy decoding. "
        f"Sets: {', '.join(cfg.data.perturbation_sets)}. Seed {cfg.seed}. "
        f"95% CIs: {boot}, {spec.n_boot} resamples. "
        "Every number below is computed by `scripts/analyze.py`.",
        "",
        "## 1. Robustness (pair level)",
        "",
        "| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, block in list(res["per_set"].items()) + [("**pooled**", pooled)]:
        rb = block["robustness"]
        lines.append(f"| {name} | {rb['n_pairs']} | {100*rb['ex_clean']:.1f}% | "
                     f"{100*rb['ex_pert']:.1f}% | {rb['ex_drop_pts']:.1f} | "
                     f"{rb['mcnemar']['b_clean_only']} | {rb['mcnemar']['c_pert_only']} | "
                     f"{rb['mcnemar']['p_value']:.3g} |")
    lines += [
        "",
        f"Pooled drop 95% CI: {_ci(ci['ex_drop_pts'], 1)} pts "
        f"({r['n_clean_questions']} distinct clean questions). "
        f"Identical SQL for both twins in {100*r['identical_prediction_rate']:.1f}% of pairs. "
        f"No closing fence (cap or loop guard): clean {100*r['truncated_rate_clean']:.1f}%, "
        f"perturbed {100*r['truncated_rate_pert']:.1f}%. Execution errors: clean "
        f"{100*r['exec_error_rate_clean']:.1f}%, perturbed {100*r['exec_error_rate_pert']:.1f}%.",
        "",
        "## 2. Discrimination: AUROC (correct vs wrong), pooled",
        "",
        "| confidence | clean [95% CI] | perturbed [95% CI] |",
        "|---|---|---|",
    ]
    for m, v in pooled["discrimination"].items():
        lines.append(f"| {m} | {_f(v['clean'])} {_ci(ci[f'auroc_{m}_clean'])} | "
                     f"{_f(v['perturbed'])} {_ci(ci[f'auroc_{m}_perturbed'])} |")
    lines += ["", "## 3. Calibration, pooled", "",
              "| confidence | side | mean conf | accuracy | ECE | Brier |",
              "|---|---|---|---|---|---|"]
    for m, sides in pooled["calibration"].items():
        for side in ("clean", "perturbed"):
            c = sides[side]
            lines.append(f"| {m} | {side} | {_f(c['mean_conf'])} | {_f(c['accuracy'])} | "
                         f"{_f(c['ece'])} | {_f(c['brier'])} |")
    lines += [
        "",
        "## 4. Core test: does confidence drop when a rewrite breaks a correct answer?",
        "",
        f"Among {fd['n_clean_correct']} clean-correct pairs, {fd['n_flips']} flip to wrong (R→W).",
        "",
        "| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in spec.metrics:
        v = fd[m]
        lines.append(f"| {m} | {_f(v['mean_delta_RR'], 4)} | {_f(v['mean_delta_RW'], 4)} | "
                     f"{_f(v['auroc_flip_from_delta'])} {_ci(ci[f'auroc_flip_from_delta_{m}'])} | "
                     f"{_f(v['auroc_flip_from_pert_conf'])} | {_f(v['mannwhitney_p'], 4)} | "
                     f"{_f(v['blind_spot_rate'])} |")
    lc = fd["length_control"]
    lines += ["", "Blind-spot rate = share of R→W flips whose confidence did not decrease. "
              f"Length control: mean Δ output tokens R→R {_f(lc['mean_dlen_RR'], 2)}, "
              f"R→W {_f(lc['mean_dlen_RW'], 2)}; flip AUROC from length increase alone "
              f"{_f(lc['auroc_flip_from_length_increase'])} "
              f"{_ci(ci['auroc_flip_from_length_increase'])}.",
              "", "## 5. Flip detection by perturbation set (E4), primary score", "",
              f"| set | clean-correct | flips | flip AUROC ({k}) | blind-spot rate |",
              "|---|---|---|---|---|"]
    for name, block in res["per_set"].items():
        b = block["flip_detection"]
        lines.append(f"| {name} | {b['n_clean_correct']} | {b['n_flips']} | "
                     f"{_f(b[k]['auroc_flip_from_delta'])} | {_f(b[k]['blind_spot_rate'])} |")
    lines += ["", "Figures: `figures/`."]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
def analyze_block(df: pd.DataFrame, spec: Spec) -> dict:
    return {"robustness": robustness(df), "discrimination": discrimination(df, spec),
            "calibration": calibration(df, spec), "flip_detection": flip_detection(df, spec)}


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def run_spec(by_uid: dict, spec: Spec) -> None:
    spec.out.mkdir(parents=True, exist_ok=True)
    fig_dir = spec.out / "figures"
    fig_dir.mkdir(exist_ok=True)

    df = build_pair_table(spec.cfg, by_uid, spec.label, spec.metrics)
    log.info("[%s] analyzing %d complete pairs", spec.label, len(df))
    res = {"label": spec.label,
           "per_set": {s: analyze_block(g, spec) for s, g in df.groupby("set", sort=False)},
           "pooled": analyze_block(df, spec),
           "bootstrap": bootstrap(df, spec)}

    (spec.out / "metrics.json").write_text(json.dumps(res, indent=2, default=_jsonable))
    df.to_csv(spec.out / "pairs_scored.csv", index=False)
    fig_accuracy({**res["per_set"], "pooled": res["pooled"]}, fig_dir / "fig1_accuracy.png")
    fig_reliability(res["pooled"]["calibration"], fig_dir / "fig2_reliability.png")
    fig_delta(df, spec.primary, fig_dir / "fig3_delta_confidence.png")
    fig_conf_hist(df, spec.primary, fig_dir / "fig4_confidence_hist.png")
    write_summary(res, spec, spec.out / "SUMMARY.md")
    log.info("[%s] wrote metrics.json, SUMMARY.md, figures to %s", spec.label, spec.out)


def main() -> None:
    _, cfg = parse(__doc__)
    a = cfg.analysis
    by_uid = load_rows(cfg)
    sample = next(iter(by_uid.values()))

    # Keep only scores that exist (e.g. probes are absent if features were off).
    present = lambda keys: [k for k in keys
                            if any(r.get(k) is not None for r in by_uid.values())]
    metrics = present(a.confidence_metrics)
    calibrated = present(a.get("calibrated_metrics", [a.primary_confidence]))
    dropped = sorted(set(a.confidence_metrics) - set(metrics))
    if dropped:
        log.warning("Confidence scores not found, skipped: %s", dropped)

    labels = a.get("labels")
    if labels is None:                      # CP1 layout
        specs = [Spec(cfg, "exec_ok", metrics, calibrated, run_dir(cfg))]
    else:
        specs = [Spec(cfg, lab, metrics, calibrated, run_dir(cfg) / f"analysis_{lab}")
                 for lab in labels if lab in sample]
        skipped = [lab for lab in labels if lab not in sample]
        if skipped:
            log.warning("Labels not present in labeled.jsonl, skipped: %s", skipped)
    save_resolved_config(cfg, "config_analysis.yaml")
    for spec in specs:
        run_spec(by_uid, spec)


if __name__ == "__main__":
    main()
