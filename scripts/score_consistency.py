"""E3: sampling-based confidence from agreement of execution results.

For each question, the k sampled SQL queries are executed and grouped by their
result (same rows, ignoring row order and column order). From the groups:

  sc_agree        share of samples whose result equals the greedy prediction's result
                  (0 if the greedy prediction fails to execute)
  sc_majority     share of samples in the largest result group
  sc_neg_entropy  minus the entropy of the group distribution (failed samples form one
                  "error" group), a result-level analogue of semantic entropy
                  (Kuhn et al., ICLR 2023)

Greedy predictions and their labels are reused from the E1 run named in
``consistency.greedy_run``; this script writes, for the E3 run:
  results/<run>/labeled.jsonl            greedy rows for the E3 questions (copied)
  results/<run>/scores_consistency.jsonl sc_* scores
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from multiprocessing import Pool

from _cli import parse
from nl2sql_conf.config import processed_dir, run_dir
from nl2sql_conf.data import db_path
from nl2sql_conf.execution import QueryError, execute
from nl2sql_conf.utils import get_logger, read_jsonl, repo_path, write_jsonl

log = get_logger("score_consistency")


def result_key(rows, decimals: int = 4) -> str:
    """Order-insensitive fingerprint of a result set (rows and columns)."""
    norm = sorted(tuple(sorted(repr(round(v, decimals) if isinstance(v, float) else v)
                               for v in r)) for r in rows)
    return hashlib.sha1(repr(norm).encode()).hexdigest()


def _keys_for(task):
    """Worker: execute greedy + samples for one question -> result keys (None = error)."""
    uid, db, sqls, timeout = task
    keys = []
    for sql in sqls:
        try:
            keys.append(result_key(execute(db, sql, timeout)) if sql else None)
        except QueryError:
            keys.append(None)
    return uid, keys


def consistency_scores(greedy_key: str | None, sample_keys: list[str | None]) -> dict:
    k = len(sample_keys)
    groups = Counter(x if x is not None else "<error>" for x in sample_keys)
    probs = [c / k for c in groups.values()]
    valid = Counter(x for x in sample_keys if x is not None)
    return {
        "sc_agree": (sum(x == greedy_key for x in sample_keys) / k) if greedy_key else 0.0,
        "sc_majority": (max(valid.values()) / k) if valid else 0.0,
        "sc_neg_entropy": sum(p * math.log(p) for p in probs),
    }


def main() -> None:
    _, cfg = parse(__doc__)
    c = cfg.consistency
    out = run_dir(cfg)
    examples = {e["uid"] for e in read_jsonl(processed_dir(cfg) / "examples.jsonl")}
    samples = {r["uid"]: r for r in read_jsonl(out / "samples.jsonl")}
    greedy_dir = repo_path(cfg.paths.results_root) / c.greedy_run
    greedy = {r["uid"]: r for r in read_jsonl(greedy_dir / "labeled.jsonl")}

    missing = [u for u in examples if u not in samples or u not in greedy]
    if missing:
        raise SystemExit(f"{len(missing)} E3 questions lack samples or greedy labels; "
                         "finish inference/evaluation first.")

    tasks = [(u, str(db_path(cfg, greedy[u]["db_id"])),
              [greedy[u]["pred_sql"]] + [s["sql"] for s in samples[u]["samples"]],
              c.timeout_s) for u in sorted(examples)]
    workers = os.cpu_count() if c.workers == "auto" else int(c.workers)
    rows = []
    with Pool(workers) as pool:
        for n, (uid, keys) in enumerate(pool.imap_unordered(_keys_for, tasks, chunksize=8), 1):
            rows.append({"uid": uid, **consistency_scores(keys[0], keys[1:])})
            if n % 500 == 0:
                log.info("%d/%d questions scored", n, len(tasks))

    write_jsonl(out / "labeled.jsonl", [greedy[u] for u in sorted(examples)])
    write_jsonl(out / "scores_consistency.jsonl", rows)
    log.info("Wrote consistency scores for %d questions (k = %d)", len(rows),
             cfg.sampling.k)


if __name__ == "__main__":
    main()
