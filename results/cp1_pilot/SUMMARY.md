# cp1_pilot: auto-generated results summary

Model `Qwen/Qwen2.5-Coder-0.5B-Instruct` (rev `ea3f2471`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_column_synonym, NLQ_value_synonym. Seed 42. Every number below is computed by `scripts/analyze.py` from `labeled.jsonl`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 100 | 43.0% | 34.0% | 9.0 | 17 | 8 | 0.108 |
| column_synonym | 100 | 41.0% | 24.0% | 17.0 | 19 | 2 | 0.000221 |
| value_synonym | 100 | 34.0% | 22.0% | 12.0 | 14 | 2 | 0.00418 |
| **pooled** | 300 | 39.3% | 26.7% | 12.7 | 50 | 12 | 1.21e-06 |

Pooled drop 95% CI: [8.0, 17.3] pts. Identical SQL for both twins in 24.3% of pairs.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean | perturbed |
|---|---|---|
| p_gm | 0.704 | 0.614 |
| p_min | 0.735 | 0.714 |
| seq_logprob | 0.824 | 0.773 |
| neg_entropy | 0.667 | 0.598 |

`p_gm` AUROC 95% CI: clean [0.651, 0.762], perturbed [0.543, 0.685].

## 3. Calibration of `p_gm`, pooled

| | mean confidence | accuracy | ECE | Brier |
|---|---|---|---|---|
| clean | 0.861 | 0.393 | 0.468 | 0.442 |
| perturbed | 0.850 | 0.267 | 0.584 | 0.530 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 118 clean-correct pairs, 50 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | AUROC flip from −Δ | AUROC flip from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0210 | -0.0298 | 0.540 | 0.594 | 0.4606 | 0.320 |
| p_min | -0.0439 | -0.0827 | 0.577 | 0.676 | 0.1528 | 0.320 |
| seq_logprob | -0.5638 | -2.1645 | 0.617 | 0.649 | 0.0304 | 0.300 |
| neg_entropy | -0.0466 | -0.0752 | 0.544 | 0.593 | 0.4218 | 0.340 |

Flip-AUROC (from −Δ) 95% CIs: `p_gm` [0.421, 0.646], `p_min` [0.460, 0.688], `seq_logprob` [0.500, 0.736], `neg_entropy` [0.421, 0.651]. Blind-spot rate = share of R→W flips whose confidence did not decrease.

Length control (clean-correct pairs): mean Δ output tokens R→R 1.18, R→W 10.88; AUROC of flip from length increase alone 0.547 (95% CI [0.432, 0.671]).

Truncated decodes (no closing fence within max_new_tokens): clean 3.3%, perturbed 2.0% of pairs. Execution errors: clean 42.3%, perturbed 45.3%.

Figures: `figures/`.
