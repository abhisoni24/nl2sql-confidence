"""Step 2: generate SQL for every example (resumable).

Modes
  greedy (default)  one greedy decode per example + token confidence statistics
                    -> results/<run>/predictions.jsonl
                    + hidden-state features if ``features.enabled``
                    -> results/<run>/features/shard_*.npz          (E1, E5)
  sample            ``sampling.k`` sampled decodes per example
                    -> results/<run>/samples.jsonl                  (E3)

Re-running skips uids already written, so a disconnect (e.g. Colab) loses at most
one batch. Batches are formed from length-sorted prompts, longest first, so an
out-of-memory error shows up immediately; on OOM the batch is split in half and
retried.

    python scripts/run_inference.py --config configs/experiment/cp2_e1_qwen7b.yaml
    python scripts/run_inference.py --config configs/experiment/cp2_e3_qwen7b.yaml --mode sample
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from _cli import parse
from nl2sql_conf.config import processed_dir, run_dir, save_resolved_config
from nl2sql_conf.confidence import confidence_scores
from nl2sql_conf.data import db_path
from nl2sql_conf.model import SQLGenerator
from nl2sql_conf.prompting import build_prompt
from nl2sql_conf.schema import schema_ddl
from nl2sql_conf.utils import append_jsonl, get_logger, read_jsonl, set_seed

log = get_logger("run_inference")


def extra(parser) -> None:
    parser.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="stop cleanly after this wall-clock budget")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most this many new examples (smoke tests)")


def run_with_backoff(fn, batch: list, min_size: int = 1) -> list:
    """Call fn(batch); on CUDA OOM split the batch in half and recurse."""
    try:
        return fn(batch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) <= min_size:
            raise
        half = len(batch) // 2
        log.warning("OOM at batch size %d; retrying as %d + %d", len(batch), half,
                    len(batch) - half)
        return run_with_backoff(fn, batch[:half]) + run_with_backoff(fn, batch[half:])


def batch_seed(base: int, uids: list[str]) -> int:
    """Seed derived from the batch contents, so a resumed run samples identically."""
    return (base + int(hashlib.sha1("".join(uids).encode()).hexdigest()[:8], 16)) % 2**31


def main() -> None:
    args, cfg = parse(__doc__, extra)
    set_seed(cfg.seed)
    t_start = time.monotonic()
    out_dir = run_dir(cfg)

    examples = read_jsonl(processed_dir(cfg) / "examples.jsonl")
    if not examples:
        raise SystemExit("No examples found; run scripts/prepare_data.py first.")
    out_file = out_dir / ("predictions.jsonl" if args.mode == "greedy" else "samples.jsonl")
    done = {r["uid"] for r in read_jsonl(out_file)}
    todo = [e for e in examples if e["uid"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    log.info("[%s] %d examples total, %d done, %d to run", args.mode, len(examples),
             len(done), len(todo))
    if not todo:
        return
    save_resolved_config(cfg, f"config_inference_{args.mode}.yaml")

    features = cfg.get("features") if args.mode == "greedy" else None
    gen = SQLGenerator(cfg.model, cfg.generation, cfg.prompt.stop_string, features)
    feat_dir = out_dir / "features"
    if gen.feature_cfg:
        feat_dir.mkdir(exist_ok=True)

    # Build prompts, then sort longest-first (OOM surfaces early; padding stays small).
    for e in todo:
        schema = schema_ddl(str(db_path(cfg, e["db_id"])))
        e["prompt"] = build_prompt(cfg.prompt, gen.tokenizer, schema, e["question"])
        e["prompt_tokens"] = gen.prompt_length(e["prompt"])
    todo.sort(key=lambda e: -e["prompt_tokens"])

    if args.mode == "greedy":
        bs = cfg.generation.batch_size
        step = lambda b: gen.generate([e["prompt"] for e in b])
    else:
        s = cfg.sampling
        bs = s.prompts_per_batch
        step = lambda b: gen.sample([e["prompt"] for e in b], s.k, s.temperature, s.top_p,
                                    batch_seed(cfg.seed, [e["uid"] for e in b]))

    n_written = 0
    for i in range(0, len(todo), bs):
        if args.max_seconds and time.monotonic() - t_start > args.max_seconds:
            log.info("Time budget reached; stopping after %d examples", n_written)
            break
        batch = todo[i:i + bs]
        t0 = time.monotonic()
        outs = run_with_backoff(step, batch)
        dt = (time.monotonic() - t0) / len(batch)

        rows = []
        for e, g in zip(batch, outs):
            base = {"uid": e["uid"], "db_id": e["db_id"], "question": e["question"],
                    "gold_sql": e["gold_sql"], "prompt_tokens": e["prompt_tokens"],
                    "seconds_per_example": dt}
            if args.mode == "greedy":
                rows.append({**base, "pred_sql": g.sql, "raw_output": g.raw_text,
                             "hit_stop": g.hit_stop, "stop_reason": g.stop_reason,
                             "n_generated": g.n_generated,
                             "token_logprobs": g.token_logprobs,
                             "token_entropies": g.token_entropies,
                             **confidence_scores(g.token_logprobs, g.token_entropies)})
            else:
                rows.append({**base, "samples": [
                    {"sql": x.sql, "stop_reason": x.stop_reason} for x in g]})

        # Features first, predictions second: a uid is "done" only once both exist.
        if gen.feature_cfg:
            np.savez(feat_dir / f"shard_{int(time.time() * 1000)}_{i:06d}.npz",
                     uids=np.array([e["uid"] for e in batch]),
                     feats=np.stack([g.features for g in outs]))
        append_jsonl(out_file, rows)
        n_written += len(rows)
        log.info("%d/%d  (%.2fs/example, batch %d)", len(done) + n_written, len(examples),
                 dt, len(batch))


if __name__ == "__main__":
    main()
