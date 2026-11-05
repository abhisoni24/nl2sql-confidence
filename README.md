# nl2sql_confidence

ECE/CS 559 project: **does a text-to-SQL model's confidence track failures caused by
meaning-preserving question rewrites?** Paired study on Dr.Spider NLQ twins.
See `PLAN.md` for design and checkpoints. Latest results: `reports/cp2/CP2_progress_update.pdf`
(one page) and `results/cp2_summary/<label>/CP2_SUMMARY.md` (all tables).
Large per-example outputs (`labeled.jsonl`, `pairs_scored.csv`) are kept locally, not in git.

## Checkpoint 2 on a GPU (Colab)

1. `python colab/make_bundle.py` builds `colab/nl2sql_confidence_code.zip`.
2. Open `colab/cp2_colab.ipynb` in Colab, choose an A100 (High-RAM) runtime, *Run all*.
   Cell 3 asks for the zip if it is not already in `MyDrive/nl2sql_cp2/`.
3. Results go to `MyDrive/nl2sql_cp2/results/`; the cross-model summary is
   `results/cp2_summary/<label>/CP2_SUMMARY.md`. After a disconnect, *Run all* again: it resumes.

On any other Linux GPU box: `make cp2-smoke`, then `make cp2` (or `scripts/run_cp2.py --help`).

## Quickstart

```bash
conda env create -f environment.yml      # creates `nl2sql-conf`, installs src/ in editable mode
conda activate nl2sql-conf
make cp1                                 # test -> data -> infer -> eval -> analyze -> verify
```

Each step can be run alone (`make data`, `make infer`, ...). Inference is resumable:
re-running skips finished examples, and `--max-seconds N` time-boxes a chunk.

## Configs

```
configs/base.yaml                       seed, paths
configs/data/drspider_nlq.yaml          sets, pairs per set, checksum
configs/model/qwen2.5-coder-0.5b.yaml   model id + pinned revision, decoding, prompt text
configs/experiment/cp1_pilot.yaml       composes the three above + eval/analysis settings
```

Override anything from the command line, without editing files:

```bash
python scripts/run_inference.py --config configs/experiment/cp1_pilot.yaml \
    generation.batch_size=2 model.device=mps
```

A new experiment = a new file in `configs/experiment/` with its own `run_name`.
Every run saves its fully resolved config next to its outputs.

## Pipeline and outputs

| step | script | output |
|---|---|---|
| 1 | `prepare_data.py` | `data/processed/<run>/pairs.jsonl`, `examples.jsonl` |
| 2 | `run_inference.py` | `results/<run>/predictions.jsonl` (SQL + per-token log-probs/entropies) |
| 3 | `evaluate.py` | `results/<run>/labeled.jsonl` (execution-match label) |
| 4 | `analyze.py` | `metrics.json`, `pairs_scored.csv`, `figures/`, `SUMMARY.md` |
| 5 | `verify_results.py` | independent recomputation of headline numbers (exit 1 on mismatch) |
| CP2 | `augment.py` | E5 probes + E6 Platt scaling, out-of-fold by DB -> `scores_augment.jsonl` |
| CP2 | `score_consistency.py` | E3 sample-agreement scores -> `scores_consistency.jsonl` |
| CP2 | `compare_models.py` | cross-model tables and figures -> `results/cp2_summary/` |
| CP2 | `run_cp2.py` | runs all of the above per `configs/cp2_plan.yaml`, resumably |

## Code map

| module | responsibility |
|---|---|
| `config.py` | layered YAML + CLI overrides (OmegaConf) |
| `data.py` | download, checksum, pair integrity checks, seeded sampling |
| `schema.py` / `prompting.py` | schema text and chat prompt (wording lives in config) |
| `model.py` | greedy/sampled decoding, token log-probs and entropies, loop guard, hidden-state features |
| `confidence.py` | sequence-level scores: `p_gm`, `p_min`, `seq_logprob`, `neg_entropy` |
| `execution.py` | read-only SQLite execution with timeout, result comparison, test-suite match |
| `metrics.py` | AUROC, ECE, Brier, McNemar, Mann-Whitney, (cluster) bootstrap CIs |

## Data

Dr.Spider (Chang et al., ICLR 2023) from `awslabs/diagnostic-robustness-text-to-sql`
(annotations under CC BY 4.0 per the repo's `LICENSE-ANNOTATIONS`; built on Spider-dev). Downloaded on first run to `data/raw/`
(git-ignored) and checked against the SHA-256 in the data config.
