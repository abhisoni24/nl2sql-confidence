# cp2_e1_deepseek-coder-6.7b: auto-generated results summary (label `ts_ok`)

Model `deepseek-ai/deepseek-coder-6.7b-instruct` (rev `e5d64add`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 45.8% | 38.4% | 7.3 | 122 | 52 | 1.15e-07 |
| keyword_carrier | 399 | 69.4% | 66.9% | 2.5 | 25 | 15 | 0.154 |
| column_synonym | 563 | 47.4% | 33.7% | 13.7 | 106 | 29 | 1.66e-11 |
| column_carrier | 579 | 61.1% | 29.7% | 31.4 | 193 | 11 | 4e-44 |
| column_attribute | 119 | 55.5% | 26.9% | 28.6 | 36 | 2 | 5.4e-09 |
| column_value | 304 | 69.7% | 35.5% | 34.2 | 109 | 5 | 1.48e-26 |
| value_synonym | 506 | 56.9% | 33.2% | 23.7 | 138 | 18 | 4.24e-24 |
| multitype | 1351 | 55.7% | 33.7% | 22.0 | 348 | 51 | 1.85e-55 |
| others | 2819 | 58.3% | 49.7% | 8.6 | 350 | 108 | 7.46e-31 |
| **pooled** | 7593 | 56.6% | 41.6% | 15.0 | 1427 | 291 | 1.13e-179 |

Pooled drop 95% CI: [12.5, 17.8] pts (1020 distinct clean questions). Identical SQL for both twins in 36.3% of pairs. No closing fence (cap or loop guard): clean 0.1%, perturbed 0.3%. Execution errors: clean 3.9%, perturbed 4.1%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.548 [0.506, 0.592] | 0.566 [0.538, 0.595] |
| p_min | 0.638 [0.600, 0.676] | 0.616 [0.592, 0.642] |
| seq_logprob | 0.799 [0.765, 0.832] | 0.789 [0.767, 0.809] |
| neg_entropy | 0.563 [0.522, 0.603] | 0.574 [0.546, 0.602] |
| p_gm_platt | 0.492 [0.451, 0.535] | 0.526 [0.498, 0.555] |
| probe_lr | 0.821 [0.792, 0.851] | 0.830 [0.811, 0.850] |
| probe_mlp | 0.798 [0.764, 0.828] | 0.795 [0.771, 0.818] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.947 | 0.566 | 0.381 | 0.390 |
| p_gm | perturbed | 0.942 | 0.416 | 0.526 | 0.518 |
| p_gm_platt | clean | 0.646 | 0.566 | 0.083 | 0.256 |
| p_gm_platt | perturbed | 0.639 | 0.416 | 0.223 | 0.293 |
| probe_lr | clean | 0.647 | 0.566 | 0.159 | 0.201 |
| probe_lr | perturbed | 0.538 | 0.416 | 0.169 | 0.209 |
| probe_mlp | clean | 0.633 | 0.566 | 0.115 | 0.195 |
| probe_mlp | perturbed | 0.566 | 0.416 | 0.158 | 0.217 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 4295 clean-correct pairs, 1427 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0037 | -0.0104 | 0.577 [0.542, 0.608] | 0.621 | 0.0000 | 0.359 |
| p_min | -0.0157 | -0.0400 | 0.563 [0.525, 0.599] | 0.623 | 0.0000 | 0.397 |
| seq_logprob | -0.1088 | -1.0689 | 0.714 [0.678, 0.747] | 0.758 | 0.0000 | 0.224 |
| neg_entropy | -0.0085 | -0.0220 | 0.591 [0.558, 0.623] | 0.616 | 0.0000 | 0.315 |
| p_gm_platt | -0.0061 | -0.0151 | 0.573 [0.538, 0.604] | 0.605 | 0.0000 | 0.359 |
| probe_lr | -0.0456 | -0.3826 | 0.806 [0.776, 0.833] | 0.820 | 0.0000 | 0.118 |
| probe_mlp | -0.0305 | -0.2182 | 0.755 [0.720, 0.789] | 0.763 | 0.0000 | 0.156 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.28, R→W 11.08; flip AUROC from length increase alone 0.671 [0.628, 0.715].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 436 | 122 | 0.534 | 0.418 |
| keyword_carrier | 277 | 25 | 0.670 | 0.240 |
| column_synonym | 267 | 106 | 0.527 | 0.349 |
| column_carrier | 354 | 193 | 0.540 | 0.311 |
| column_attribute | 66 | 36 | 0.408 | 0.639 |
| column_value | 212 | 109 | 0.721 | 0.239 |
| value_synonym | 288 | 138 | 0.534 | 0.457 |
| multitype | 752 | 348 | 0.586 | 0.342 |
| others | 1643 | 350 | 0.570 | 0.363 |

Figures: `figures/`.
