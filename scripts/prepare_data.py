"""Step 1: download/verify Dr.Spider and sample the clean/perturbed pairs.

Outputs (data/processed/<run_name>/):
    pairs.jsonl     one row per sampled pair (links clean_uid <-> perturbed_uid)
    examples.jsonl  one row per unique question to run through the model

Usage:
    python scripts/prepare_data.py --config configs/experiment/cp1_pilot.yaml
"""

from __future__ import annotations

from collections import Counter

from _cli import parse
from nl2sql_conf.config import processed_dir, save_resolved_config
from nl2sql_conf.data import db_path, ensure_raw_data, sample_pairs, to_rows
from nl2sql_conf.utils import get_logger, write_jsonl

log = get_logger("prepare_data")


def main() -> None:
    _, cfg = parse(__doc__)
    root = ensure_raw_data(cfg)
    pairs, examples = sample_pairs(cfg, root)

    # Every referenced database must exist before we spend compute on inference.
    missing = {e.db_id for e in examples.values() if not db_path(cfg, e.db_id).exists()}
    if missing:
        raise FileNotFoundError(f"Missing SQLite files for: {sorted(missing)}")

    out = processed_dir(cfg)
    write_jsonl(out / "pairs.jsonl", to_rows(pairs))
    write_jsonl(out / "examples.jsonl", to_rows(examples.values()))
    save_resolved_config(cfg, "config_prepare.yaml")

    uses = Counter(p.clean_uid for p in pairs)
    log.info("Pairs per set: %s", dict(Counter(p.set_name for p in pairs)))
    log.info("%d pairs -> %d unique examples (%d clean questions reused across sets)",
             len(pairs), len(examples), sum(n > 1 for n in uses.values()))


if __name__ == "__main__":
    main()
