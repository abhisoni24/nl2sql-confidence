# cp2_e1_qwen2.5-coder-3b: auto-generated results summary (label `ts_ok`)

Model `Qwen/Qwen2.5-Coder-3B-Instruct` (rev `488639f1`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 56.2% | 48.2% | 8.1 | 105 | 28 | 1.12e-11 |
| keyword_carrier | 399 | 77.7% | 69.4% | 8.3 | 36 | 3 | 3.61e-08 |
| column_synonym | 563 | 49.2% | 36.4% | 12.8 | 99 | 27 | 7.47e-11 |
| column_carrier | 579 | 59.4% | 50.6% | 8.8 | 84 | 33 | 2.67e-06 |
| column_attribute | 119 | 47.1% | 37.0% | 10.1 | 19 | 7 | 0.029 |
| column_value | 304 | 63.8% | 36.8% | 27.0 | 90 | 8 | 1.09e-18 |
| value_synonym | 506 | 59.5% | 38.1% | 21.3 | 115 | 7 | 2.68e-26 |
| multitype | 1351 | 59.1% | 41.2% | 17.9 | 279 | 37 | 4.3e-47 |
| others | 2819 | 60.4% | 53.2% | 7.2 | 324 | 122 | 3.33e-22 |
| **pooled** | 7593 | 59.5% | 47.9% | 11.6 | 1151 | 272 | 9.81e-129 |

Pooled drop 95% CI: [9.4, 13.8] pts (1020 distinct clean questions). Identical SQL for both twins in 44.6% of pairs. No closing fence (cap or loop guard): clean 0.1%, perturbed 0.0%. Execution errors: clean 12.2%, perturbed 13.3%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.674 [0.633, 0.715] | 0.655 [0.627, 0.682] |
| p_min | 0.731 [0.692, 0.769] | 0.718 [0.693, 0.743] |
| seq_logprob | 0.842 [0.809, 0.871] | 0.804 [0.780, 0.825] |
| neg_entropy | 0.654 [0.614, 0.695] | 0.639 [0.611, 0.667] |
| p_gm_platt | 0.652 [0.613, 0.693] | 0.641 [0.613, 0.668] |
| probe_lr | 0.834 [0.803, 0.861] | 0.793 [0.771, 0.815] |
| probe_mlp | 0.858 [0.830, 0.884] | 0.805 [0.783, 0.826] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.932 | 0.595 | 0.336 | 0.346 |
| p_gm | perturbed | 0.922 | 0.479 | 0.443 | 0.438 |
| p_gm_platt | clean | 0.679 | 0.595 | 0.094 | 0.232 |
| p_gm_platt | perturbed | 0.645 | 0.479 | 0.167 | 0.261 |
| probe_lr | clean | 0.691 | 0.595 | 0.136 | 0.185 |
| probe_lr | perturbed | 0.620 | 0.479 | 0.184 | 0.230 |
| probe_mlp | clean | 0.698 | 0.595 | 0.107 | 0.161 |
| probe_mlp | perturbed | 0.646 | 0.479 | 0.168 | 0.213 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 4519 clean-correct pairs, 1151 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0075 | -0.0267 | 0.649 [0.605, 0.692] | 0.658 | 0.0000 | 0.264 |
| p_min | -0.0238 | -0.0660 | 0.569 [0.526, 0.609] | 0.682 | 0.0000 | 0.373 |
| seq_logprob | -0.1725 | -1.2569 | 0.728 [0.683, 0.769] | 0.737 | 0.0000 | 0.215 |
| neg_entropy | -0.0186 | -0.0650 | 0.680 [0.644, 0.715] | 0.645 | 0.0000 | 0.192 |
| p_gm_platt | -0.0297 | -0.0883 | 0.643 [0.599, 0.687] | 0.658 | 0.0000 | 0.264 |
| probe_lr | -0.0472 | -0.2909 | 0.728 [0.696, 0.761] | 0.743 | 0.0000 | 0.189 |
| probe_mlp | -0.0346 | -0.1959 | 0.702 [0.668, 0.735] | 0.725 | 0.0000 | 0.194 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.14, R→W 5.47; flip AUROC from length increase alone 0.605 [0.560, 0.652].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 536 | 105 | 0.623 | 0.314 |
| keyword_carrier | 310 | 36 | 0.814 | 0.111 |
| column_synonym | 277 | 99 | 0.675 | 0.222 |
| column_carrier | 344 | 84 | 0.665 | 0.119 |
| column_attribute | 56 | 19 | 0.637 | 0.158 |
| column_value | 194 | 90 | 0.664 | 0.278 |
| value_synonym | 301 | 115 | 0.550 | 0.400 |
| multitype | 799 | 279 | 0.635 | 0.269 |
| others | 1702 | 324 | 0.643 | 0.265 |

Figures: `figures/`.
