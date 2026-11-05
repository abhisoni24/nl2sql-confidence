# cp2_e1_deepseek-coder-6.7b: auto-generated results summary (label `exec_ok`)

Model `deepseek-ai/deepseek-coder-6.7b-instruct` (rev `e5d64add`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 51.2% | 45.2% | 6.0 | 115 | 58 | 1.75e-05 |
| keyword_carrier | 399 | 77.7% | 76.2% | 1.5 | 20 | 14 | 0.392 |
| column_synonym | 563 | 53.5% | 38.5% | 14.9 | 110 | 26 | 1.71e-13 |
| column_carrier | 579 | 71.8% | 40.1% | 31.8 | 197 | 13 | 2.21e-43 |
| column_attribute | 119 | 57.1% | 36.1% | 21.0 | 28 | 3 | 4.65e-06 |
| column_value | 304 | 75.7% | 53.6% | 22.0 | 74 | 7 | 3.17e-15 |
| value_synonym | 506 | 60.3% | 39.9% | 20.4 | 123 | 20 | 2.78e-19 |
| multitype | 1351 | 60.3% | 40.4% | 19.8 | 322 | 54 | 1.42e-47 |
| others | 2819 | 67.8% | 59.9% | 7.8 | 340 | 119 | 1.05e-25 |
| **pooled** | 7593 | 63.8% | 50.4% | 13.4 | 1329 | 314 | 2.07e-148 |

Pooled drop 95% CI: [11.0, 15.9] pts (1020 distinct clean questions). Identical SQL for both twins in 36.3% of pairs. No closing fence (cap or loop guard): clean 0.1%, perturbed 0.3%. Execution errors: clean 3.9%, perturbed 4.1%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.564 [0.520, 0.608] | 0.581 [0.553, 0.610] |
| p_min | 0.652 [0.613, 0.693] | 0.629 [0.605, 0.653] |
| seq_logprob | 0.788 [0.752, 0.820] | 0.766 [0.744, 0.787] |
| neg_entropy | 0.571 [0.528, 0.614] | 0.583 [0.553, 0.612] |
| p_gm_platt | 0.490 [0.446, 0.534] | 0.529 [0.502, 0.558] |
| probe_lr | 0.832 [0.802, 0.862] | 0.823 [0.803, 0.843] |
| probe_mlp | 0.796 [0.762, 0.827] | 0.775 [0.751, 0.798] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.947 | 0.638 | 0.309 | 0.325 |
| p_gm | perturbed | 0.942 | 0.504 | 0.438 | 0.439 |
| p_gm_platt | clean | 0.646 | 0.638 | 0.047 | 0.234 |
| p_gm_platt | perturbed | 0.639 | 0.504 | 0.135 | 0.268 |
| probe_lr | clean | 0.647 | 0.638 | 0.129 | 0.177 |
| probe_lr | perturbed | 0.538 | 0.504 | 0.143 | 0.197 |
| probe_mlp | clean | 0.633 | 0.638 | 0.109 | 0.184 |
| probe_mlp | perturbed | 0.566 | 0.504 | 0.110 | 0.210 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 4842 clean-correct pairs, 1329 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0040 | -0.0098 | 0.570 [0.534, 0.602] | 0.631 | 0.0000 | 0.366 |
| p_min | -0.0179 | -0.0315 | 0.540 [0.501, 0.576] | 0.632 | 0.0000 | 0.418 |
| seq_logprob | -0.1523 | -1.0824 | 0.695 [0.660, 0.728] | 0.736 | 0.0000 | 0.244 |
| neg_entropy | -0.0089 | -0.0210 | 0.587 [0.552, 0.619] | 0.624 | 0.0000 | 0.318 |
| p_gm_platt | -0.0062 | -0.0143 | 0.566 [0.531, 0.597] | 0.603 | 0.0000 | 0.366 |
| probe_lr | -0.0581 | -0.4000 | 0.806 [0.780, 0.834] | 0.805 | 0.0000 | 0.100 |
| probe_mlp | -0.0379 | -0.2296 | 0.754 [0.721, 0.786] | 0.737 | 0.0000 | 0.141 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.57, R→W 11.46; flip AUROC from length increase alone 0.663 [0.622, 0.705].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 488 | 115 | 0.555 | 0.400 |
| keyword_carrier | 310 | 20 | 0.605 | 0.300 |
| column_synonym | 301 | 110 | 0.518 | 0.391 |
| column_carrier | 416 | 197 | 0.559 | 0.294 |
| column_attribute | 68 | 28 | 0.454 | 0.607 |
| column_value | 230 | 74 | 0.626 | 0.257 |
| value_synonym | 305 | 123 | 0.498 | 0.496 |
| multitype | 814 | 322 | 0.574 | 0.348 |
| others | 1910 | 340 | 0.573 | 0.365 |

Figures: `figures/`.
