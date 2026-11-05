"""Step 3: attach correctness labels to greedy predictions.

    exec_ok  execution match on the original Spider DB (as in CP1)
    ts_ok    test-suite match (E2): exec_ok AND identical results on every DB
             variant of the test suite (Zhong et al., EMNLP 2020). Only computed
             when ``evaluation.test_suite.enabled``; since the original DB is one of
             the variants, ts_ok can only turn a CP1-style "correct" into "wrong".

Reads  results/<run_name>/predictions.jsonl
Writes results/<run_name>/labeled.jsonl
"""

from __future__ import annotations

import os
import time
from collections import Counter
from multiprocessing import Pool

from _cli import parse
from nl2sql_conf.config import run_dir
from nl2sql_conf.data import db_path, ensure_test_suite, suite_db_files
from nl2sql_conf.execution import (QueryError, execute, gold_is_ordered, results_match,
                                   suite_match)
from nl2sql_conf.utils import get_logger, read_jsonl, write_jsonl

log = get_logger("evaluate")


def _suite_task(args):
    """Worker: (index, db files, gold, pred, timeout, decimals) -> (index, result)."""
    i, files, gold, pred, timeout, decimals = args
    return i, suite_match(files, gold, pred, timeout, decimals)


def main() -> None:
    _, cfg = parse(__doc__)
    preds = read_jsonl(run_dir(cfg) / "predictions.jsonl")
    ev = cfg.evaluation
    gold_cache: dict[tuple[str, str], list] = {}
    labeled, outcomes = [], Counter()

    # ---- single-DB execution match (exec_ok) --------------------------------
    for p in preds:
        db = db_path(cfg, p["db_id"])
        key = (p["db_id"], p["gold_sql"])
        if key not in gold_cache:
            # Gold must execute; a failure here is a data problem, not a model error.
            gold_cache[key] = execute(db, p["gold_sql"], ev.timeout_s)

        error = None
        if not p["pred_sql"]:
            ok, error = False, "empty prediction"
        else:
            try:
                pred_rows = execute(db, p["pred_sql"], ev.timeout_s)
                ok = results_match(gold_cache[key], pred_rows,
                                   gold_is_ordered(p["gold_sql"]), ev.float_decimals)
            except QueryError as exc:
                ok, error = False, str(exc)

        outcomes["correct" if ok else ("exec_error" if error else "wrong_result")] += 1
        labeled.append({**p, "exec_ok": bool(ok), "exec_error": error})
    log.info("exec_ok labels for %d predictions: %s", len(labeled), dict(outcomes))

    # ---- test-suite match (ts_ok), only for exec_ok predictions --------------
    ts_cfg = ev.get("test_suite")
    if ts_cfg and ts_cfg.enabled:
        root = ensure_test_suite(cfg, {p["db_id"] for p in labeled})
        files = {d: [str(f) for f in suite_db_files(root, d)]
                 for d in {p["db_id"] for p in labeled}}
        tasks = [(i, files[r["db_id"]], r["gold_sql"], r["pred_sql"], ts_cfg.timeout_s,
                  ev.float_decimals) for i, r in enumerate(labeled) if r["exec_ok"]]
        tasks.sort(key=lambda t: t[2])  # group by gold query -> better gold-result caching
        workers = os.cpu_count() if ts_cfg.workers == "auto" else int(ts_cfg.workers)
        t0 = time.monotonic()
        for r in labeled:
            r["ts_ok"], r["ts_variants"], r["ts_skipped"] = False, 0, 0
        with Pool(workers) as pool:
            for n, (i, (ok, checked, skipped)) in enumerate(
                    pool.imap_unordered(_suite_task, tasks, chunksize=16), 1):
                labeled[i].update(ts_ok=bool(ok), ts_variants=checked, ts_skipped=skipped)
                if n % 500 == 0:
                    log.info("test-suite %d/%d (%.0fs)", n, len(tasks), time.monotonic() - t0)
        n_ex = sum(r["exec_ok"] for r in labeled)
        n_ts = sum(r["ts_ok"] for r in labeled)
        log.info("ts_ok: %d of %d exec_ok predictions survive the test suite (%.0fs, %d workers)",
                 n_ts, n_ex, time.monotonic() - t0, workers)

    write_jsonl(run_dir(cfg) / "labeled.jsonl", labeled)


if __name__ == "__main__":
    main()
