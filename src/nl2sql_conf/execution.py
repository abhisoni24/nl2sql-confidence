"""Execution-match (EX) labels: run gold and predicted SQL, compare results.

Comparison rules (a simplified version of the Spider test-suite comparator):

1. Floats are rounded to ``float_decimals`` so 3.3333333 == 3.3333.
2. Row order matters only if the gold query has ORDER BY.
3. Column order does not matter: ``SELECT a, b`` matches ``SELECT b, a``.

Example::

    gold rows [(1, 'x'), (2, 'y')], pred rows [('y', 2), ('x', 1)], no ORDER BY
    -> match (same multiset of rows after permuting columns)
"""

from __future__ import annotations

import itertools
import sqlite3
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

Row = tuple[Any, ...]


class QueryError(Exception):
    pass


def execute(db_file: Path, sql: str, timeout_s: float) -> list[Row]:
    """Run ``sql`` read-only; raise QueryError on any failure or timeout."""
    con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    # Some Spider DBs contain invalid UTF-8; decode leniently instead of crashing.
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout_s
    # Returning non-zero from the progress handler aborts the running statement.
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        return con.execute(sql).fetchall()
    except Exception as exc:  # sqlite3.Error, OverflowError, Warning (multi-stmt), ...
        raise QueryError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        con.close()


def _normalize(rows: list[Row], decimals: int) -> list[Row]:
    def norm(v):
        return round(v, decimals) if isinstance(v, float) else v
    return [tuple(norm(v) for v in row) for row in rows]


def _same(gold: list[Row], pred: list[Row], ordered: bool) -> bool:
    return gold == pred if ordered else Counter(gold) == Counter(pred)


def results_match(gold: list[Row], pred: list[Row], ordered: bool, decimals: int = 4) -> bool:
    gold, pred = _normalize(gold, decimals), _normalize(pred, decimals)
    if len(gold) != len(pred):
        return False
    if not gold:
        return True  # both empty
    n_cols = len(gold[0])
    if any(len(r) != n_cols for r in pred):
        return False
    if _same(gold, pred, ordered):
        return True

    # Try column permutations of pred, but only those where each column's
    # multiset of values matches the target gold column (cheap pruning).
    gold_cols = [Counter(r[i] for r in gold) for i in range(n_cols)]
    pred_cols = [Counter(r[i] for r in pred) for i in range(n_cols)]
    candidates = [[j for j in range(n_cols) if pred_cols[j] == gold_cols[i]]
                  for i in range(n_cols)]
    if any(not c for c in candidates):
        return False
    for perm in itertools.product(*candidates):
        if len(set(perm)) != n_cols:
            continue
        permuted = [tuple(r[j] for j in perm) for r in pred]
        if _same(gold, permuted, ordered):
            return True
    return False


def gold_is_ordered(gold_sql: str) -> bool:
    return "order by" in " ".join(gold_sql.lower().split())


# --------------------------------------------------------------------------- #
# Test-suite execution match (E2)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8192)
def _gold_rows(db_file: str, gold_sql: str, timeout_s: float) -> list[Row] | None:
    """Gold results per (variant, query), cached: ~1k gold queries x ~40 variants."""
    try:
        return execute(Path(db_file), gold_sql, timeout_s)
    except QueryError:
        return None


def suite_match(db_files: list[Path], gold_sql: str, pred_sql: str, timeout_s: float,
                decimals: int = 4) -> tuple[bool, int, int]:
    """Correct only if pred matches gold on every DB variant (Zhong et al., 2020).

    A simplified re-implementation of the test-suite idea with this module's
    comparator. Stops at the first mismatching variant. Variants on which the
    *gold* query itself fails are skipped (counted in the return value).

    Returns (match, variants_checked, variants_skipped).
    """
    ordered = gold_is_ordered(gold_sql)
    checked = skipped = 0
    for db in db_files:
        gold = _gold_rows(str(db), gold_sql, timeout_s)
        if gold is None:
            skipped += 1
            continue
        checked += 1
        try:
            pred = execute(db, pred_sql, timeout_s)
        except QueryError:
            return False, checked, skipped
        if not results_match(gold, pred, ordered, decimals):
            return False, checked, skipped
    return checked > 0, checked, skipped
