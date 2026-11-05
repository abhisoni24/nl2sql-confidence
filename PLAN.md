# PLAN: Does model confidence track perturbation-induced NL2SQL failures?

ECE/CS 559 group project (Boss + Shubham Harale). Plan preview, written before implementation.
Status legend: `[x]` done, `[ ]` open. CP0 to CP2 are done; CP3 is next. Working notes for the
next steps (status, hypotheses, priorities): `ROADMAP.md`.

## 1. Question in one line

Given a clean question `q` and a meaning-preserving rewrite `q'` with the **same gold SQL**, when the
model is right on `q` but wrong on `q'`, does its confidence drop?

```
q  : "What is the name and capacity for the stadium with the highest average attendance?"
q' : "What is the name and size of the stadium that recorded the highest average attendance?"
gold (both): SELECT name, capacity FROM stadium ORDER BY average DESC LIMIT 1
```

If confidence stays high on `q'` failures, confidence is a poor guard against paraphrase-induced errors.

### Project goals

| id | goal | status |
|---|---|---|
| G1 | Measure how often meaning-preserving rewrites break correct answers, across model sizes | done on Dr.Spider (CP2) |
| G2 | Test whether confidence signals (token probabilities, calibration, sampling, hidden-state probe) warn of those breaks | done on Dr.Spider (CP2) |
| G3 | **Show the findings hold on more than one dataset**: repeat G1 and G2 with the same models on a second benchmark besides Dr.Spider (instructor request) | open (CP3) |
| G4 | Rule out confounds (answer length, label noise) and test probe transfer to unseen rewrite types and datasets | open (CP3) |

## 2. Data (verified on disk, 2026-10-05)

Dr.Spider (Chang et al., ICLR 2023, arXiv:2301.08881), `data.tar.gz` from
`awslabs/diagnostic-robustness-text-to-sql` (git-LFS, 168 MB, sha256 `d0f47e4d...a3ba9`).
Contains Spider-dev (1,034 questions, 20 SQLite DBs) plus 17 perturbation sets.

We use the 9 **NLQ** sets: pre/post question files are index-aligned, and in all 9 sets the gold SQL
is identical pre vs post (checked: 7,593 / 7,593 pairs). So correctness labels are directly comparable.

| NLQ set | pairs | NLQ set | pairs | NLQ set | pairs |
|---|---|---|---|---|---|
| keyword_synonym | 953 | column_synonym | 563 | value_synonym | 506 |
| keyword_carrier | 399 | column_carrier | 579 | column_value | 304 |
| column_attribute | 119 | multitype | 1,351 | others | 2,819 |

DB_* and SQL_* sets change the schema or gold SQL; deferred to CP3.

## 3. Measurement design

**Correctness label.** Execution match (EX): run predicted and gold SQL on the SQLite DB, compare
result multisets (order-sensitive only if gold has `ORDER BY`). CP2 upgrades to test-suite EX
(Zhong et al., EMNLP 2020) to reduce false positives.

**Confidence (white-box, from one greedy decode).** Over SQL tokens only (we prefill
` ```sql\n ` and stop at the closing fence, so formatting tokens never enter the score):

| name | formula | role |
|---|---|---|
| `p_gm` | exp(mean log p(t_i)) | primary; in [0,1], used for calibration |
| `seq_logprob` | Σ log p(t_i) | length-biased reference |
| `p_min` | min p(t_i) | weakest-link signal |
| `neg_entropy` | −mean H(p(· given prefix)) | spread of the full next-token distribution |

**Analyses** (pooled over sets, plus per set):

1. Robustness: EX(q) vs EX(q'), exact McNemar test on discordant pairs.
2. Discrimination: AUROC of confidence for correct vs wrong, clean and perturbed separately.
3. Calibration: ECE (10 equal-width bins; Guo et al., ICML 2017) and Brier score of `p_gm`.
4. **Core paired test**: among pairs correct on `q`, split into ✓→✓ (stays right) and ✓→✗ (flips).
   Compare Δconf = conf(q') − conf(q). Report AUROC of −Δconf for detecting flips (0.5 = blind)
   and a Mann-Whitney U p-value.
5. Blind-spot rate: share of ✓→✗ flips where conf(q') ≥ conf(q).

All randomness (pair sampling, bootstrap) seeded from config. 95% CIs by bootstrap (1,000 resamples).

## 4. Repository layout and conventions

```
configs/          base.yaml + data/, model/, experiment/ (composed by OmegaConf; CLI key=value overrides)
src/nl2sql_conf/  library code: config, data, schema, prompting, model, confidence, execution, metrics
scripts/          thin entry points: prepare_data, run_inference, evaluate, analyze
tests/            unit tests (execution match, confidence math, ECE/AUROC)
data/             raw + processed (git-ignored)
results/<run>/    resolved config, predictions.jsonl, labeled.jsonl, metrics.json, figures/, SUMMARY.md
```

Rules: no hard-coded paths or hyperparameters in code; every run writes its resolved config next to
its outputs; inference is resumable (append-only JSONL keyed by example id); code never edits raw data.

## 5. Checkpoints

### CP0: Infrastructure (2026-10-05)
- [x] Download Dr.Spider, verify checksum, confirm pre/post alignment and gold equality
- [x] Conda env `nl2sql-conf` from `environment.yml`
- [x] Config system, repo skeleton, unit tests passing (16 tests)
- **Gate:** `pytest` green; one example runs end-to-end (prompt → SQL → EX label → confidence).

### CP1: Pilot result for the Oct 6 presentation (done 2026-10-05)
- [x] Model: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (Hui et al., 2024, arXiv:2409.12186), greedy, CPU.
      Chosen because it fits the CPU-only sandbox; it is a pipeline test, not the final model.
- [x] Sets: keyword_synonym, column_synonym, value_synonym (three different perturbation targets:
      SQL keywords, column mentions, cell values). 100 pairs per set (seeded), at most one rewrite
      per clean question per set, since one clean question can have several rewrites
      (e.g. keyword_synonym: 953 pairs from 413 clean questions).
- [x] Run analyses 1 to 5; figures: accuracy per set, reliability diagram, Δconf by transition.
- [x] Gate passed: `scripts/verify_results.py` recomputes headline numbers independently (all match).
      Results: `results/cp1_pilot/SUMMARY.md` (auto) and `FINDINGS.md` (interpretation).
- **Gate:** every number in `results/cp1_pilot/SUMMARY.md` is reproducible from
  `predictions.jsonl` with `scripts/analyze.py`, and independently re-checked.
- **Known limits to state in class:** small model, small `n`, single-DB EX, single decode.

### CP2: Full study on GPU (done 2026-11-05)

Ran on a Colab A100 80 GB (driven via the Colab CLI) with `scripts/run_cp2.py --batch-scale 2`:
2.3 h, all stages OK, every run re-verified independently (140/140 checks).
One-page update: `reports/cp2/CP2_progress_update.pdf`; tables: `results/cp2_summary/<label>/CP2_SUMMARY.md`.
The probe and recalibration (E5, E6) moved here from CP3.

| id | experiment | what we hope to learn |
|---|---|---|
| E1 | Qwen2.5-Coder 0.5/1.5/3/7/14B + DeepSeek-Coder-6.7B, all 9 NLQ sets (7,593 pairs, 7,214 unique questions) | Does the rewrite-induced drop shrink with scale, and does confidence start to catch flips (CP1: flip AUROC 0.540 at 0.5B)? Is it Qwen-specific? |
| E2 | Test-suite labels `ts_ok` next to single-DB `exec_ok` | Do conclusions survive when coincidental matches stop counting as correct? |
| E3 | 7B, 3 pilot sets, k = 10 samples, agreement of execution results | Does sample agreement catch flips that token probabilities miss (CP1 blind spot: 32%)? |
| E4 | Per-set breakdown of E1 | Which rewrite types fool confidence most? |
| E5 | Linear and MLP probes on hidden states, fit on clean questions, tested on perturbed (out-of-fold by DB) | Do internal states carry failure signal that output probabilities lose under rewording? |
| E6 | Platt scaling of p_gm, fit on clean, tested on perturbed | Does calibration fit on clean data hold under rewording (CP1 ECE 0.468 → 0.584)? |

Harness changes (all tested; `pytest` 21 fast + 2 slow model tests):
- [x] Loop guard (stop on a 10-gram repeated 3x); CUDA OOM backoff (batch halves)
- [x] Confidence recorded by a pass-through logits processor (checked equal to `output_logits`)
- [x] Hidden-state capture (layers at 50/75/100% depth; last prompt token, mean SQL token),
      checked invariant to left padding
- [x] Test-suite labels (simplified re-implementation; test-suite zip verified downloadable)
- [x] Cluster bootstrap over clean questions (pairs sharing a clean question are dependent)
- [x] Orchestrator: GPU inference of model i+1 overlaps CPU evaluation of model i;
      stage markers make every step resumable; results can live on Google Drive
- [x] Smoke plan (`configs/cp2_plan_smoke.yaml`) runs every stage on 0.5B in minutes
- [x] Full run on GPU, then write-up (`results/cp2_summary/<label>/CP2_SUMMARY.md`)

### CP3: Remaining experiments (target: before Dec 1 final talk)
- [ ] **Second dataset (G3, instructor request):** same six models on Spider-Syn (1,034 paired
      original/synonym questions, same gold SQL, MIT license; verified 2026-11-05), then a benchmark on
      different databases (candidates and trade-offs in `ROADMAP.md`)
- [ ] Length-controlled confidence comparisons (seq_logprob edge is partly output length)
- [ ] Probe transfer across rewrite types; add DB_* and SQL_* perturbation families (E7)
- [ ] Tie-aware sampling analysis, larger k, a second model
- [ ] Manual audit of ~100 flips for rewrites that change meaning (label noise)

### CP4: Final paper and code (Dec 4)
- [ ] Frozen configs per figure/table, `make reproduce`, final write-up

## 6. Risks and fallbacks

| risk | fallback |
|---|---|
| 0.5B accuracy too low, few ✓→✗ flips for test 4 | raise `n`, or move to 1.5B for CP2 |
| CPU throughput | resumable runs; report exactly what `n` was achieved |
| EX false positives (coincidental matches) | test-suite EX in CP2; note in CP1 |
| Label noise in Spider gold | manual audit of a sample of flips in CP2 |
