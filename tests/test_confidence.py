"""Confidence formulas match the hand-computed worked example."""

import math

from nl2sql_conf.confidence import confidence_scores


def test_worked_example():
    probs = [0.9, 0.5, 0.99]
    s = confidence_scores([math.log(p) for p in probs], [0.1, 0.7, 0.05])
    assert math.isclose(s["p_gm"], (0.9 * 0.5 * 0.99) ** (1 / 3))
    assert math.isclose(s["p_min"], 0.5)
    assert math.isclose(s["seq_logprob"], math.log(0.9 * 0.5 * 0.99))
    assert math.isclose(s["neg_entropy"], -(0.1 + 0.7 + 0.05) / 3)
    assert s["n_tokens"] == 3


def test_empty_is_nan():
    s = confidence_scores([], [])
    assert math.isnan(s["p_gm"]) and s["n_tokens"] == 0
