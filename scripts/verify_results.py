"""Independent re-check of the headline numbers in metrics.json (CP1/CP2 gate).

Deliberately does NOT import nl2sql_conf.metrics or pandas: AUROC is recomputed
from its pairwise definition, ECE and McNemar counts by hand, and a random
sample of execution labels is re-derived with a fresh SQLite connection.
Exits non-zero if anything disagrees.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import time
from collections import Counter

from _cli import parse
from nl2sql_conf.config import processed_dir, run_dir
from nl2sql_conf.data import db_path
from nl2sql_conf.utils import read_jsonl


def auroc_pairwise(scores, labels):
    """AUROC = P(s_pos > s_neg) + 0.5 * P(tie), over all positive/negative pairs."""
    pos = [s for s, y in zip(scores, labels) if y and not math.isnan(s)]
    neg = [s for s, y in zip(scores, labels) if not y and not math.isnan(s)]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def ece_manual(conf, ok, n_bins):
    bins = [[] for _ in range(n_bins)]
    pairs = [(c, y) for c, y in zip(conf, ok) if not math.isnan(c)]
    conf = [c for c, _ in pairs]
    for c, y in pairs:
        bins[min(int(c * n_bins), n_bins - 1)].append((c, y))
    n = len(conf)
    return sum(len(b) / n * abs(sum(y for _, y in b) / len(b) - sum(c for c, _ in b) / len(b))
               for b in bins if b)


def close(a, b, tol=1e-9):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol


def main() -> None:
    args, cfg = parse(__doc__, lambda p: p.add_argument(
        "--label", default="exec_ok", help="which analysis_<label>/ to check (CP2)"))
    label = args.label
    # CP1 wrote metrics.json at the run root; CP2 writes analysis_<label>/metrics.json.
    adir = run_dir(cfg) / f"analysis_{label}"
    if not adir.exists():
        adir = run_dir(cfg)
    k = cfg.analysis.primary_confidence
    labeled = {r["uid"]: r for r in read_jsonl(run_dir(cfg) / "labeled.jsonl")}
    pairs = [p for p in read_jsonl(processed_dir(cfg) / "pairs.jsonl")
             if p["clean_uid"] in labeled and p["perturbed_uid"] in labeled]
    m = json.loads((adir / "metrics.json").read_text())["pooled"]
    checks = []

    c = [labeled[p["clean_uid"]] for p in pairs]
    q = [labeled[p["perturbed_uid"]] for p in pairs]
    ok_c, ok_p = [bool(r[label]) for r in c], [bool(r[label]) for r in q]

    # 1. Accuracy and McNemar discordant counts.
    checks.append(("EX clean", sum(ok_c) / len(ok_c), m["robustness"]["ex_clean"]))
    checks.append(("EX perturbed", sum(ok_p) / len(ok_p), m["robustness"]["ex_pert"]))
    b = sum(a and not z for a, z in zip(ok_c, ok_p))
    cc = sum(z and not a for a, z in zip(ok_c, ok_p))
    checks.append(("McNemar b", b, m["robustness"]["mcnemar"]["b_clean_only"]))
    checks.append(("McNemar c", cc, m["robustness"]["mcnemar"]["c_pert_only"]))

    # 2. AUROC and ECE of the primary confidence.
    conf_c, conf_p = [r[k] for r in c], [r[k] for r in q]
    checks.append(("AUROC clean", auroc_pairwise(conf_c, ok_c), m["discrimination"][k]["clean"]))
    checks.append(("AUROC pert", auroc_pairwise(conf_p, ok_p), m["discrimination"][k]["perturbed"]))
    checks.append(("ECE clean", ece_manual(conf_c, ok_c, cfg.analysis.ece_bins),
                   m["calibration"][k]["clean"]["ece"]))
    checks.append(("ECE pert", ece_manual(conf_p, ok_p, cfg.analysis.ece_bins),
                   m["calibration"][k]["perturbed"]["ece"]))

    # 3. Flip detection among clean-correct pairs.
    base = [(rc[k], rq[k], bool(rq[label])) for rc, rq in zip(c, q) if rc[label]]
    deltas = [p_ - c_ for c_, p_, _ in base]
    flips = [not okp for _, _, okp in base]
    checks.append(("n flips", sum(flips), m["flip_detection"]["n_flips"]))
    checks.append(("flip AUROC", auroc_pairwise([-d for d in deltas], flips),
                   m["flip_detection"][k]["auroc_flip_from_delta"]))

    # 4. Re-derive a random sample of execution labels with a fresh, simple comparator
    #    (multiset of rows with sorted cell values, so column order is ignored).
    rng = random.Random(cfg.seed)
    sample = rng.sample(list(labeled.values()), min(60, len(labeled)))
    agree = 0
    for r in sample:
        def run(sql):
            con = sqlite3.connect(f"file:{db_path(cfg, r['db_id'])}?mode=ro", uri=True)
            con.text_factory = lambda x: x.decode("utf-8", errors="replace")
            deadline = time.monotonic() + cfg.evaluation.timeout_s   # same budget as evaluate.py
            con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
            try:
                rows = con.execute(sql).fetchall()
            finally:
                con.close()
            return Counter(tuple(sorted(map(lambda v: repr(round(v, 4) if isinstance(v, float) else v), row)))
                           for row in rows)
        try:
            ok = bool(r["pred_sql"]) and run(r["pred_sql"]) == run(r["gold_sql"])
        except Exception:
            ok = False
        agree += ok == r["exec_ok"]
    checks.append(("label agreement on sample (ORDER BY ignored here)", agree / len(sample), None))

    failed = False
    for name, mine, reported in checks:
        if reported is None:
            print(f"[info] {name}: {mine:.3f}")
            continue
        ok = close(float(mine), float(reported))
        failed |= not ok
        print(f"[{'ok' if ok else 'MISMATCH'}] {name}: recomputed {mine} vs reported {reported}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
