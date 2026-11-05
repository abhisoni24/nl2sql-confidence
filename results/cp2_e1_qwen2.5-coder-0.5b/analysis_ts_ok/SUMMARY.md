# cp2_e1_qwen2.5-coder-0.5b: auto-generated results summary (label `ts_ok`)

Model `Qwen/Qwen2.5-Coder-0.5B-Instruct` (rev `ea3f2471`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 40.4% | 31.8% | 8.6 | 124 | 42 | 1.36e-10 |
| keyword_carrier | 399 | 54.6% | 52.9% | 1.8 | 23 | 16 | 0.337 |
| column_synonym | 563 | 28.1% | 18.5% | 9.6 | 75 | 21 | 2.71e-08 |
| column_carrier | 579 | 42.7% | 31.4% | 11.2 | 102 | 37 | 3.22e-08 |
| column_attribute | 119 | 42.0% | 30.3% | 11.8 | 18 | 4 | 0.00434 |
| column_value | 304 | 30.3% | 16.1% | 14.1 | 48 | 5 | 7.08e-10 |
| value_synonym | 506 | 28.9% | 13.6% | 15.2 | 92 | 15 | 1.11e-14 |
| multitype | 1351 | 38.9% | 23.9% | 15.0 | 263 | 60 | 1.86e-31 |
| others | 2819 | 41.3% | 34.8% | 6.4 | 309 | 128 | 2.36e-18 |
| **pooled** | 7593 | 39.3% | 29.8% | 9.6 | 1054 | 328 | 5.52e-89 |

Pooled drop 95% CI: [7.1, 12.0] pts (1020 distinct clean questions). Identical SQL for both twins in 31.9% of pairs. No closing fence (cap or loop guard): clean 1.0%, perturbed 1.0%. Execution errors: clean 36.1%, perturbed 37.1%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.666 [0.624, 0.707] | 0.624 [0.596, 0.654] |
| p_min | 0.746 [0.703, 0.784] | 0.716 [0.686, 0.744] |
| seq_logprob | 0.839 [0.805, 0.867] | 0.818 [0.793, 0.841] |
| neg_entropy | 0.648 [0.605, 0.688] | 0.615 [0.587, 0.646] |
| p_gm_platt | 0.656 [0.614, 0.698] | 0.616 [0.588, 0.646] |
| probe_lr | 0.775 [0.736, 0.812] | 0.770 [0.741, 0.795] |
| probe_mlp | 0.805 [0.768, 0.838] | 0.791 [0.764, 0.816] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.867 | 0.393 | 0.473 | 0.451 |
| p_gm | perturbed | 0.855 | 0.298 | 0.558 | 0.513 |
| p_gm_platt | clean | 0.475 | 0.393 | 0.083 | 0.229 |
| p_gm_platt | perturbed | 0.444 | 0.298 | 0.147 | 0.226 |
| probe_lr | clean | 0.416 | 0.393 | 0.156 | 0.215 |
| probe_lr | perturbed | 0.346 | 0.298 | 0.152 | 0.205 |
| probe_mlp | clean | 0.468 | 0.393 | 0.087 | 0.182 |
| probe_mlp | perturbed | 0.421 | 0.298 | 0.124 | 0.184 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 2985 clean-correct pairs, 1054 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0113 | -0.0310 | 0.616 [0.577, 0.658] | 0.600 | 0.0000 | 0.284 |
| p_min | -0.0208 | -0.0701 | 0.622 [0.584, 0.660] | 0.689 | 0.0000 | 0.299 |
| seq_logprob | -0.2303 | -1.9333 | 0.737 [0.696, 0.780] | 0.768 | 0.0000 | 0.195 |
| neg_entropy | -0.0316 | -0.0824 | 0.625 [0.587, 0.663] | 0.596 | 0.0000 | 0.256 |
| p_gm_platt | -0.0365 | -0.0809 | 0.608 [0.569, 0.651] | 0.602 | 0.0000 | 0.284 |
| probe_lr | -0.0599 | -0.2661 | 0.699 [0.663, 0.737] | 0.720 | 0.0000 | 0.160 |
| probe_mlp | -0.0305 | -0.2099 | 0.766 [0.732, 0.801] | 0.735 | 0.0000 | 0.126 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.34, R→W 8.39; flip AUROC from length increase alone 0.656 [0.609, 0.707].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 385 | 124 | 0.603 | 0.355 |
| keyword_carrier | 218 | 23 | 0.841 | 0.087 |
| column_synonym | 158 | 75 | 0.547 | 0.267 |
| column_carrier | 247 | 102 | 0.616 | 0.176 |
| column_attribute | 50 | 18 | 0.576 | 0.222 |
| column_value | 92 | 48 | 0.602 | 0.354 |
| value_synonym | 146 | 92 | 0.548 | 0.283 |
| multitype | 526 | 263 | 0.564 | 0.278 |
| others | 1163 | 309 | 0.618 | 0.307 |

Figures: `figures/`.
