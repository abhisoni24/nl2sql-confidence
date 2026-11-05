"""Checkpoint 2 orchestrator: runs every experiment in configs/cp2_plan.yaml, resumably.

Per E1 model (in plan order):
    GPU, in this process   prepare -> infer (greedy + features)
    CPU, background queue  evaluate (exec_ok + E2 ts_ok) -> augment (E5, E6)
                           -> analyze (E1, E4) -> verify
E3 (after its base model):
    GPU   prepare -> infer --mode sample
    CPU   score_consistency -> analyze -> verify   (needs the base model's labels)
Finally: compare_models (cross-model tables and figures).

The GPU never waits for CPU post-processing: while model i is being evaluated on
CPU, model i+1 is generating. Every stage leaves a marker file
results/<run>/.done_<stage>; re-running skips finished stages, and inference
itself resumes mid-run. Stage logs: results/<run>/logs/<stage>.log.

    python scripts/run_cp2.py                          # everything
    python scripts/run_cp2.py --results-root /content/drive/MyDrive/nl2sql_cp2/results
    python scripts/run_cp2.py --only 0.5b,1.5b --skip-e3
    python scripts/run_cp2.py --status                 # progress table, runs nothing
    python scripts/run_cp2.py --dry-run                # print the plan
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from omegaconf import OmegaConf

from nl2sql_conf.config import load_config
from nl2sql_conf.utils import REPO_ROOT, get_logger, read_jsonl, repo_path

log = get_logger("run_cp2")
PY = sys.executable


class Run:
    """One experiment config plus where its outputs live."""

    def __init__(self, config: str, overrides: list[str], batch_scale: float = 1.0):
        self.config = config
        self.overrides = list(overrides)
        if batch_scale != 1.0:  # e.g. 2x on an 80 GB GPU; OOM backoff still halves if needed
            base = load_config(config, overrides).generation.batch_size
            self.overrides.append(f"generation.batch_size={max(1, int(base * batch_scale))}")
        self.cfg = load_config(config, self.overrides)
        self.name = self.cfg.run_name
        self.dir = repo_path(self.cfg.paths.results_root) / self.name
        self.processed = repo_path(self.cfg.paths.processed_dir) / self.name

    def done(self, stage: str) -> bool:
        return (self.dir / f".done_{stage}").exists()

    def mark(self, stage: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f".done_{stage}").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))

    def count(self, filename: str) -> int:
        f = self.dir / filename
        return len({r["uid"] for r in read_jsonl(f)}) if f.exists() else 0

    def n_examples(self) -> int:
        return len(read_jsonl(self.processed / "examples.jsonl"))


def stage(run: Run, name: str, script: str, *extra: str, dry: bool = False) -> bool:
    """Run one pipeline script as a subprocess, logging to results/<run>/logs/."""
    if run.done(name):
        log.info("[%s] %s: already done", run.name, name)
        return True
    cmd = [PY, str(REPO_ROOT / "scripts" / script), "--config", run.config, *extra,
           *run.overrides]
    if dry:
        log.info("[%s] would run: %s", run.name, " ".join(cmd[1:]))
        return True
    logdir = run.dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    log.info("[%s] %s: started", run.name, name)
    with (logdir / f"{name}.log").open("a") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    mins = (time.monotonic() - t0) / 60
    if rc != 0:
        tail = (logdir / f"{name}.log").read_text().splitlines()[-15:]
        log.error("[%s] %s FAILED (exit %d, %.1f min). Log tail:\n  %s", run.name, name, rc,
                  mins, "\n  ".join(tail))
        return False
    run.mark(name)
    log.info("[%s] %s: done in %.1f min", run.name, name, mins)
    return True


def infer(run: Run, mode: str, out_file: str, dry: bool) -> bool:
    """Inference stage; only marked done when every example has an output row."""
    name = "infer" if mode == "greedy" else "sample"
    if run.done(name):
        log.info("[%s] %s: already done", run.name, name)
        return True
    if not stage(run, f"{name}_attempt", "run_inference.py", "--mode", mode, dry=dry):
        return False
    if dry:
        return True
    (run.dir / f".done_{name}_attempt").unlink(missing_ok=True)   # attempts are not stages
    n, total = run.count(out_file), run.n_examples()
    if n < total:
        log.error("[%s] %s incomplete: %d/%d", run.name, name, n, total)
        return False
    run.mark(name)
    return True


def postprocess_e1(run: Run, dry: bool) -> bool:
    return (stage(run, "evaluate", "evaluate.py", dry=dry)
            and stage(run, "augment", "augment.py", dry=dry)
            and stage(run, "analyze", "analyze.py", dry=dry)
            and all(stage(run, f"verify_{lab}", "verify_results.py", "--label", lab, dry=dry)
                    for lab in run.cfg.analysis.labels))


def postprocess_e3(run: Run, base_job: Future | None, dry: bool) -> bool:
    if base_job is not None and not base_job.result():
        log.error("[%s] skipped: base model post-processing failed", run.name)
        return False
    return (stage(run, "score_consistency", "score_consistency.py", dry=dry)
            and stage(run, "analyze", "analyze.py", dry=dry)
            and all(stage(run, f"verify_{lab}", "verify_results.py", "--label", lab, dry=dry)
                    for lab in run.cfg.analysis.labels))


def print_status(runs: list[Run], e3: Run | None) -> None:
    print(f"{'run':40s} {'examples':>8s} {'greedy':>7s} {'samples':>8s}  stages done")
    for r in runs + ([e3] if e3 else []):
        n = r.n_examples() if (r.processed / "examples.jsonl").exists() else 0
        stages = sorted(p.name.removeprefix(".done_") for p in r.dir.glob(".done_*")) \
            if r.dir.exists() else []
        print(f"{r.name:40s} {n:8d} {r.count('predictions.jsonl'):7d} "
              f"{r.count('samples.jsonl'):8d}  {', '.join(stages)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="configs/cp2_plan.yaml")
    ap.add_argument("--results-root", default=None, help="absolute dir for results/")
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings of run names to include (E1)")
    ap.add_argument("--skip-e3", action="store_true")
    ap.add_argument("--delete-models-after", action="store_true",
                    help="free disk: delete each model's HF cache after its GPU stages")
    ap.add_argument("--batch-scale", type=float, default=1.0,
                    help="multiply every model's configured batch size (configs assume 40 GB)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("overrides", nargs="*", help="extra dotlist overrides for every run")
    args = ap.parse_args()

    plan = OmegaConf.load(repo_path(args.plan))
    overrides = list(args.overrides)
    if args.results_root:
        overrides.append(f"paths.results_root={args.results_root}")
    runs = [Run(c, overrides, args.batch_scale) for c in plan.e1_runs]
    if args.only:
        keys = args.only.split(",")
        runs = [r for r in runs if any(k in r.name for k in keys)]
    e3 = None if args.skip_e3 else Run(plan.e3_run, overrides, args.batch_scale)
    e3_base = Run(plan.e3_after, overrides).name

    if args.status:
        print_status(runs, e3)
        return

    dry, failures, t_all = args.dry_run, [], time.monotonic()
    cpu_queue = ThreadPoolExecutor(max_workers=1)   # CPU stages run one model at a time
    jobs: dict[str, Future] = {}

    for run in runs:
        ok = (stage(run, "prepare", "prepare_data.py", dry=dry)
              and infer(run, "greedy", "predictions.jsonl", dry))
        if not ok:
            failures.append(f"{run.name}: GPU stage")
            continue
        jobs[run.name] = cpu_queue.submit(postprocess_e1, run, dry)
        if args.delete_models_after and not dry:
            _delete_model_cache(run.cfg.model.hf_id)

        if e3 is not None and run.name == e3_base:
            if (stage(e3, "prepare", "prepare_data.py", dry=dry)
                    and infer(e3, "sample", "samples.jsonl", dry)):
                jobs[e3.name] = cpu_queue.submit(postprocess_e3, e3, jobs[run.name], dry)
            else:
                failures.append(f"{e3.name}: sampling")

    for name, job in jobs.items():
        if not job.result():
            failures.append(f"{name}: CPU post-processing")
    cpu_queue.shutdown()

    if not dry:
        cmd = [PY, str(REPO_ROOT / "scripts" / "compare_models.py"), "--plan", args.plan]
        if args.results_root:
            cmd += ["--results-root", args.results_root]
        if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
            failures.append("compare_models")

    log.info("Finished in %.1f h. %s", (time.monotonic() - t_all) / 3600,
             "All stages OK." if not failures else f"FAILURES: {failures}")
    if failures:
        sys.exit(1)


def _delete_model_cache(hf_id: str) -> None:
    from huggingface_hub.constants import HF_HUB_CACHE
    path = Path(HF_HUB_CACHE) / f"models--{hf_id.replace('/', '--')}"
    if path.exists():
        shutil.rmtree(path)
        log.info("Deleted model cache %s", path)


if __name__ == "__main__":
    main()
