"""Evaluation statistics: discrimination, calibration, paired tests, bootstrap CIs."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative). NaN if one class."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    keep = ~np.isnan(scores)
    scores, labels = scores[keep], labels[keep]
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan")  # undefined without both classes
    return float(roc_auc_score(labels, scores))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins (Guo et al., 2017).

    ECE = sum_b (n_b / N) * |accuracy_b - mean_confidence_b|
    """
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    keep = ~np.isnan(probs)
    probs, labels = probs[keep], labels[keep]
    if probs.size == 0:
        return float("nan")
    # Bin index in [0, n_bins-1]; a probability of exactly 1.0 goes to the top bin.
    idx = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(total)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Per-bin counts, mean confidence and accuracy (for reliability diagrams)."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    keep = ~np.isnan(probs)
    probs, labels = probs[keep], labels[keep]
    idx = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append({
            "bin_lo": b / n_bins, "bin_hi": (b + 1) / n_bins, "n": int(m.sum()),
            "mean_conf": float(probs[m].mean()) if m.any() else None,
            "accuracy": float(labels[m].mean()) if m.any() else None,
        })
    return out


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    keep = ~np.isnan(probs)
    return float(np.mean((probs[keep] - labels[keep]) ** 2))


def mcnemar_exact(clean_ok: np.ndarray, pert_ok: np.ndarray) -> dict:
    """Exact McNemar test on paired binary outcomes.

    b = right on clean, wrong on perturbed; c = wrong on clean, right on perturbed.
    Under H0 (no perturbation effect) b ~ Binomial(b + c, 0.5).
    """
    clean_ok, pert_ok = np.asarray(clean_ok, bool), np.asarray(pert_ok, bool)
    b = int((clean_ok & ~pert_ok).sum())
    c = int((~clean_ok & pert_ok).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"b_clean_only": b, "c_pert_only": c, "p_value": float(p)}


def mann_whitney(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided Mann-Whitney U p-value (NaN if either group is empty)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def bootstrap_ci(
    fn: Callable[..., float], arrays: list[np.ndarray], n_resamples: int,
    seed: int, alpha: float = 0.05, groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``fn(*arrays)``; rows are resampled jointly.

    With ``groups`` (cluster bootstrap), whole clusters are resampled: e.g. all
    pairs sharing one clean question go in or out together, which keeps CIs honest
    when one clean question has several rewrites.
    """
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    if groups is not None:
        _, inverse = np.unique(np.asarray(groups), return_inverse=True)
        members = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    vals = []
    for _ in range(n_resamples):
        if groups is None:
            i = rng.integers(0, n, n)
        else:
            i = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        v = fn(*(a[i] for a in arrays))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
