# cp2_e1_qwen2.5-coder-7b: auto-generated results summary (label `ts_ok`)

Model `Qwen/Qwen2.5-Coder-7B-Instruct` (rev `c03e6d35`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 66.5% | 56.6% | 10.0 | 127 | 32 | 1.35e-14 |
| keyword_carrier | 399 | 77.9% | 73.4% | 4.5 | 31 | 13 | 0.00956 |
| column_synonym | 563 | 64.3% | 46.9% | 17.4 | 115 | 17 | 4.62e-19 |
| column_carrier | 579 | 76.2% | 59.1% | 17.1 | 112 | 13 | 8.12e-21 |
| column_attribute | 119 | 69.7% | 62.2% | 7.6 | 24 | 15 | 0.2 |
| column_value | 304 | 76.6% | 43.4% | 33.2 | 105 | 4 | 1.78e-26 |
| value_synonym | 506 | 68.0% | 44.3% | 23.7 | 128 | 8 | 5.76e-29 |
| multitype | 1351 | 71.9% | 51.0% | 20.9 | 331 | 49 | 1.58e-52 |
| others | 2819 | 71.1% | 63.5% | 7.6 | 297 | 82 | 1.02e-29 |
| **pooled** | 7593 | 70.9% | 57.3% | 13.7 | 1270 | 233 | 8.69e-173 |

Pooled drop 95% CI: [11.0, 15.9] pts (1020 distinct clean questions). Identical SQL for both twins in 49.4% of pairs. No closing fence (cap or loop guard): clean 0.0%, perturbed 0.0%. Execution errors: clean 3.9%, perturbed 3.9%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.699 [0.657, 0.738] | 0.641 [0.615, 0.670] |
| p_min | 0.707 [0.666, 0.741] | 0.672 [0.644, 0.699] |
| seq_logprob | 0.809 [0.773, 0.841] | 0.758 [0.732, 0.784] |
| neg_entropy | 0.707 [0.663, 0.747] | 0.659 [0.634, 0.687] |
| p_gm_platt | 0.679 [0.635, 0.719] | 0.630 [0.603, 0.659] |
| probe_lr | 0.813 [0.780, 0.843] | 0.752 [0.727, 0.776] |
| probe_mlp | 0.795 [0.756, 0.827] | 0.755 [0.728, 0.780] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.961 | 0.709 | 0.252 | 0.263 |
| p_gm | perturbed | 0.955 | 0.573 | 0.383 | 0.385 |
| p_gm_platt | clean | 0.796 | 0.709 | 0.091 | 0.199 |
| p_gm_platt | perturbed | 0.778 | 0.573 | 0.205 | 0.274 |
| probe_lr | clean | 0.801 | 0.709 | 0.136 | 0.175 |
| probe_lr | perturbed | 0.729 | 0.573 | 0.216 | 0.256 |
| probe_mlp | clean | 0.753 | 0.709 | 0.078 | 0.165 |
| probe_mlp | perturbed | 0.705 | 0.573 | 0.139 | 0.222 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 5384 clean-correct pairs, 1270 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0045 | -0.0171 | 0.640 [0.602, 0.677] | 0.641 | 0.0000 | 0.269 |
| p_min | -0.0274 | -0.0821 | 0.596 [0.560, 0.634] | 0.658 | 0.0000 | 0.313 |
| seq_logprob | -0.1007 | -0.6776 | 0.681 [0.643, 0.718] | 0.714 | 0.0000 | 0.248 |
| neg_entropy | -0.0103 | -0.0366 | 0.661 [0.622, 0.699] | 0.660 | 0.0000 | 0.243 |
| p_gm_platt | -0.0159 | -0.0554 | 0.637 [0.600, 0.676] | 0.638 | 0.0000 | 0.269 |
| probe_lr | -0.0390 | -0.2289 | 0.711 [0.679, 0.740] | 0.714 | 0.0000 | 0.170 |
| probe_mlp | -0.0372 | -0.1270 | 0.665 [0.626, 0.703] | 0.723 | 0.0000 | 0.206 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.37, R→W 3.84; flip AUROC from length increase alone 0.588 [0.545, 0.636].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 634 | 127 | 0.645 | 0.252 |
| keyword_carrier | 311 | 31 | 0.687 | 0.226 |
| column_synonym | 362 | 115 | 0.568 | 0.270 |
| column_carrier | 441 | 112 | 0.622 | 0.268 |
| column_attribute | 83 | 24 | 0.639 | 0.167 |
| column_value | 233 | 105 | 0.541 | 0.390 |
| value_synonym | 344 | 128 | 0.626 | 0.266 |
| multitype | 971 | 331 | 0.658 | 0.221 |
| others | 2005 | 297 | 0.657 | 0.300 |

Figures: `figures/`.
