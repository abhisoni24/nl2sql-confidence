# cp2_e1_qwen2.5-coder-1.5b: auto-generated results summary (label `ts_ok`)

Model `Qwen/Qwen2.5-Coder-1.5B-Instruct` (rev `2e1fd397`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 50.6% | 42.9% | 7.7 | 115 | 42 | 4.85e-09 |
| keyword_carrier | 399 | 71.7% | 65.7% | 6.0 | 32 | 8 | 0.000182 |
| column_synonym | 563 | 45.5% | 32.5% | 13.0 | 100 | 27 | 4.73e-11 |
| column_carrier | 579 | 47.0% | 39.4% | 7.6 | 83 | 39 | 8.41e-05 |
| column_attribute | 119 | 26.9% | 29.4% | -2.5 | 9 | 12 | 0.664 |
| column_value | 304 | 38.8% | 23.0% | 15.8 | 60 | 12 | 8.06e-09 |
| value_synonym | 506 | 46.6% | 31.8% | 14.8 | 93 | 18 | 2.26e-13 |
| multitype | 1351 | 44.3% | 32.0% | 12.3 | 238 | 72 | 6.82e-22 |
| others | 2819 | 55.8% | 48.2% | 7.7 | 315 | 99 | 2.52e-27 |
| **pooled** | 7593 | 50.8% | 41.3% | 9.4 | 1045 | 329 | 5.14e-87 |

Pooled drop 95% CI: [7.2, 11.7] pts (1020 distinct clean questions). Identical SQL for both twins in 44.5% of pairs. No closing fence (cap or loop guard): clean 0.0%, perturbed 0.1%. Execution errors: clean 20.3%, perturbed 21.2%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.598 [0.553, 0.639] | 0.592 [0.560, 0.620] |
| p_min | 0.740 [0.702, 0.774] | 0.708 [0.680, 0.734] |
| seq_logprob | 0.831 [0.802, 0.858] | 0.803 [0.781, 0.825] |
| neg_entropy | 0.539 [0.494, 0.579] | 0.535 [0.505, 0.563] |
| p_gm_platt | 0.578 [0.535, 0.616] | 0.579 [0.548, 0.606] |
| probe_lr | 0.804 [0.772, 0.835] | 0.777 [0.751, 0.800] |
| probe_mlp | 0.837 [0.809, 0.865] | 0.808 [0.786, 0.829] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.904 | 0.508 | 0.396 | 0.402 |
| p_gm | perturbed | 0.895 | 0.413 | 0.481 | 0.470 |
| p_gm_platt | clean | 0.637 | 0.508 | 0.131 | 0.262 |
| p_gm_platt | perturbed | 0.618 | 0.413 | 0.208 | 0.281 |
| probe_lr | clean | 0.607 | 0.508 | 0.154 | 0.212 |
| probe_lr | perturbed | 0.543 | 0.413 | 0.194 | 0.235 |
| probe_mlp | clean | 0.611 | 0.508 | 0.109 | 0.178 |
| probe_mlp | perturbed | 0.572 | 0.413 | 0.161 | 0.204 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 3854 clean-correct pairs, 1045 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0089 | -0.0241 | 0.619 [0.578, 0.663] | 0.637 | 0.0000 | 0.262 |
| p_min | -0.0190 | -0.0797 | 0.609 [0.566, 0.648] | 0.709 | 0.0000 | 0.340 |
| seq_logprob | -0.1865 | -1.4132 | 0.708 [0.662, 0.750] | 0.751 | 0.0000 | 0.227 |
| neg_entropy | -0.0273 | -0.0644 | 0.615 [0.578, 0.654] | 0.599 | 0.0000 | 0.242 |
| p_gm_platt | -0.0179 | -0.0451 | 0.614 [0.572, 0.658] | 0.636 | 0.0000 | 0.262 |
| probe_lr | -0.0505 | -0.2421 | 0.714 [0.680, 0.746] | 0.716 | 0.0000 | 0.169 |
| probe_mlp | -0.0312 | -0.1770 | 0.727 [0.691, 0.760] | 0.738 | 0.0000 | 0.180 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.14, R→W 7.01; flip AUROC from length increase alone 0.636 [0.587, 0.680].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 482 | 115 | 0.624 | 0.296 |
| keyword_carrier | 286 | 32 | 0.769 | 0.094 |
| column_synonym | 256 | 100 | 0.597 | 0.230 |
| column_carrier | 272 | 83 | 0.629 | 0.217 |
| column_attribute | 32 | 9 | 0.667 | 0.222 |
| column_value | 118 | 60 | 0.533 | 0.267 |
| value_synonym | 236 | 93 | 0.584 | 0.258 |
| multitype | 598 | 238 | 0.581 | 0.282 |
| others | 1574 | 315 | 0.622 | 0.276 |

Figures: `figures/`.
