# cp2_e3_qwen2.5-coder-7b: auto-generated results summary (label `exec_ok`)

Model `Qwen/Qwen2.5-Coder-7B-Instruct` (rev `c03e6d35`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_column_synonym, NLQ_value_synonym. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 72.4% | 63.4% | 9.0 | 116 | 30 | 3.92e-13 |
| column_synonym | 563 | 68.0% | 54.2% | 13.9 | 102 | 24 | 1.21e-12 |
| value_synonym | 506 | 71.5% | 50.0% | 21.5 | 118 | 9 | 2.25e-25 |
| **pooled** | 2022 | 71.0% | 57.5% | 13.5 | 336 | 63 | 3.96e-46 |

Pooled drop 95% CI: [10.6, 16.4] pts (615 distinct clean questions). Identical SQL for both twins in 43.3% of pairs. No closing fence (cap or loop guard): clean 0.0%, perturbed 0.1%. Execution errors: clean 4.5%, perturbed 5.4%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.719 [0.667, 0.767] | 0.607 [0.570, 0.645] |
| p_min | 0.679 [0.625, 0.729] | 0.619 [0.577, 0.658] |
| seq_logprob | 0.809 [0.758, 0.853] | 0.706 [0.663, 0.745] |
| neg_entropy | 0.732 [0.679, 0.777] | 0.629 [0.591, 0.668] |
| sc_agree | 0.828 [0.780, 0.872] | 0.725 [0.692, 0.756] |
| sc_majority | 0.825 [0.776, 0.870] | 0.724 [0.691, 0.753] |
| sc_neg_entropy | 0.821 [0.772, 0.866] | 0.713 [0.680, 0.742] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.960 | 0.710 | 0.250 | 0.261 |
| p_gm | perturbed | 0.954 | 0.575 | 0.380 | 0.384 |
| sc_agree | clean | 0.861 | 0.710 | 0.155 | 0.155 |
| sc_agree | perturbed | 0.821 | 0.575 | 0.248 | 0.264 |
| sc_majority | clean | 0.894 | 0.710 | 0.184 | 0.173 |
| sc_majority | perturbed | 0.853 | 0.575 | 0.278 | 0.275 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 1435 clean-correct pairs, 336 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0060 | -0.0160 | 0.623 [0.575, 0.668] | 0.576 | 0.0000 | 0.253 |
| p_min | -0.0312 | -0.0737 | 0.563 [0.513, 0.614] | 0.584 | 0.0005 | 0.348 |
| seq_logprob | -0.1515 | -0.6548 | 0.662 [0.613, 0.709] | 0.626 | 0.0000 | 0.247 |
| neg_entropy | -0.0126 | -0.0345 | 0.655 [0.607, 0.700] | 0.596 | 0.0000 | 0.202 |
| sc_agree | -0.0322 | -0.1795 | 0.631 [0.593, 0.667] | 0.663 | 0.0000 | 0.571 |
| sc_majority | -0.0291 | -0.1351 | 0.629 [0.592, 0.665] | 0.661 | 0.0000 | 0.571 |
| sc_neg_entropy | -0.0627 | -0.2534 | 0.611 [0.571, 0.649] | 0.657 | 0.0000 | 0.586 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.76, R→W 3.88; flip AUROC from length increase alone 0.591 [0.535, 0.650].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 690 | 116 | 0.654 | 0.233 |
| column_synonym | 383 | 102 | 0.568 | 0.265 |
| value_synonym | 362 | 118 | 0.632 | 0.263 |

Figures: `figures/`.
