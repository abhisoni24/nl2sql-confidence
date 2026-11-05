"""Unit tests for CP2 additions: loop guard, consistency scores, cluster bootstrap,
test-suite matching."""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

from nl2sql_conf import metrics as M
from nl2sql_conf.execution import suite_match
from nl2sql_conf.model import is_looping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_consistency import consistency_scores  # noqa: E402


def test_loop_detection():
    t = torch.tensor([[5, 1, 2, 1, 2, 1, 2], [5, 1, 2, 3, 4, 6, 7]])
    assert is_looping(t, ngram=2, repeats=3).tolist() == [True, False]
    assert is_looping(t[:, :3], ngram=2, repeats=3).tolist() == [False, False]  # too short


def test_consistency_scores():
    # greedy result "a"; samples: 3x a, 1x b, 1x error
    s = consistency_scores("a", ["a", "a", "a", "b", None])
    assert s["sc_agree"] == 0.6 and s["sc_majority"] == 0.6
    expected = 0.6 * np.log(0.6) + 2 * 0.2 * np.log(0.2)
    assert np.isclose(s["sc_neg_entropy"], expected)
    assert consistency_scores(None, ["a", "a"])["sc_agree"] == 0.0   # greedy failed
    assert consistency_scores("a", ["a", "a"])["sc_neg_entropy"] == 0.0


def test_cluster_bootstrap_is_wider_for_clustered_data():
    # Two clusters of identical values (0s and 1s). Resampling clusters picks
    # "both 0" or "both 1" a quarter of the time each, so the 95% CI spans [0, 1];
    # resampling rows treats the 6 rows as independent and gives a narrower CI.
    x = np.array([0, 0, 0, 1, 1, 1], float)
    g = np.array([0, 0, 0, 1, 1, 1])
    lo_c, hi_c = M.bootstrap_ci(lambda a: a.mean(), [x], 2000, seed=0, groups=g)
    lo_r, hi_r = M.bootstrap_ci(lambda a: a.mean(), [x], 2000, seed=0)
    assert (lo_c, hi_c) == (0.0, 1.0)
    assert lo_r > 0.0 and hi_r < 1.0


def _db(path, rows):
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE t(a INTEGER);")
    con.executemany("INSERT INTO t VALUES (?)", [(r,) for r in rows])
    con.commit()
    con.close()
    return path


def test_suite_match_catches_coincidental_match(tmp_path):
    # On the original DB, "a > 1" and "a = 2" give the same rows; a variant separates them.
    original = _db(tmp_path / "orig.sqlite", [1, 2])
    variant = _db(tmp_path / "var.sqlite", [1, 2, 3])
    gold, pred = "SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE a = 2"
    assert suite_match([original], gold, pred, 5)[0]
    ok, checked, skipped = suite_match([original, variant], gold, pred, 5)
    assert not ok and checked == 2 and skipped == 0
