"""Metric sanity checks on cases with known answers."""

import numpy as np

from nl2sql_conf import metrics as M


def test_auroc_perfect_and_chance():
    y = np.array([0, 0, 1, 1])
    assert M.auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert M.auroc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5
    assert np.isnan(M.auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_ece_perfect_and_overconfident():
    # 10 predictions at conf 0.8 of which 8 are right -> perfectly calibrated bin.
    assert M.ece(np.full(10, 0.8), np.array([1] * 8 + [0] * 2)) == 0.0
    # conf 0.9 but accuracy 0.5 -> ECE 0.4
    assert np.isclose(M.ece(np.full(10, 0.9), np.array([1] * 5 + [0] * 5)), 0.4)


def test_mcnemar_counts():
    clean = np.array([1, 1, 1, 0, 0], bool)
    pert = np.array([1, 0, 0, 1, 0], bool)
    r = M.mcnemar_exact(clean, pert)
    assert (r["b_clean_only"], r["c_pert_only"]) == (2, 1)


def test_auroc_empty_is_nan():
    assert np.isnan(M.auroc(np.array([]), np.array([])))
