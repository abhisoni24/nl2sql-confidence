# cp2_e1_qwen2.5-coder-14b: auto-generated results summary (label `ts_ok`)

Model `Qwen/Qwen2.5-Coder-14B-Instruct` (rev `aedcc2d4`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 69.7% | 57.8% | 11.9 | 131 | 18 | 2.28e-22 |
| keyword_carrier | 399 | 84.2% | 78.9% | 5.3 | 28 | 7 | 0.000508 |
| column_synonym | 563 | 71.2% | 54.5% | 16.7 | 109 | 15 | 8.67e-19 |
| column_carrier | 579 | 72.4% | 58.7% | 13.6 | 90 | 11 | 1.42e-16 |
| column_attribute | 119 | 79.8% | 53.8% | 26.1 | 35 | 4 | 3.35e-07 |
| column_value | 304 | 76.0% | 48.4% | 27.6 | 89 | 5 | 5.86e-21 |
| value_synonym | 506 | 69.6% | 44.1% | 25.5 | 134 | 5 | 1.2e-33 |
| multitype | 1351 | 73.4% | 50.6% | 22.8 | 332 | 24 | 1.84e-70 |
| others | 2819 | 72.8% | 65.7% | 7.1 | 267 | 67 | 2.1e-29 |
| **pooled** | 7593 | 73.0% | 59.0% | 13.9 | 1215 | 156 | 1.48e-203 |

Pooled drop 95% CI: [11.9, 16.1] pts (1020 distinct clean questions). Identical SQL for both twins in 52.8% of pairs. No closing fence (cap or loop guard): clean 0.0%, perturbed 0.0%. Execution errors: clean 0.4%, perturbed 0.9%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.633 [0.587, 0.680] | 0.634 [0.604, 0.663] |
| p_min | 0.663 [0.620, 0.705] | 0.654 [0.628, 0.683] |
| seq_logprob | 0.748 [0.705, 0.792] | 0.731 [0.702, 0.759] |
| neg_entropy | 0.666 [0.616, 0.712] | 0.666 [0.635, 0.693] |
| p_gm_platt | 0.611 [0.564, 0.656] | 0.617 [0.586, 0.647] |
| probe_lr | 0.757 [0.715, 0.797] | 0.745 [0.721, 0.770] |
| probe_mlp | 0.717 [0.665, 0.760] | 0.716 [0.682, 0.746] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.963 | 0.730 | 0.233 | 0.248 |
| p_gm | perturbed | 0.959 | 0.590 | 0.368 | 0.372 |
| p_gm_platt | clean | 0.816 | 0.730 | 0.087 | 0.198 |
| p_gm_platt | perturbed | 0.806 | 0.590 | 0.215 | 0.279 |
| probe_lr | clean | 0.830 | 0.730 | 0.161 | 0.194 |
| probe_lr | perturbed | 0.771 | 0.590 | 0.222 | 0.262 |
| probe_mlp | clean | 0.854 | 0.730 | 0.133 | 0.194 |
| probe_mlp | perturbed | 0.807 | 0.590 | 0.220 | 0.265 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 5542 clean-correct pairs, 1215 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0031 | -0.0131 | 0.604 [0.559, 0.646] | 0.656 | 0.0000 | 0.355 |
| p_min | -0.0123 | -0.0422 | 0.544 [0.504, 0.588] | 0.633 | 0.0000 | 0.411 |
| seq_logprob | -0.0607 | -0.4380 | 0.637 [0.593, 0.685] | 0.690 | 0.0000 | 0.327 |
| neg_entropy | -0.0082 | -0.0298 | 0.626 [0.584, 0.668] | 0.681 | 0.0000 | 0.307 |
| p_gm_platt | -0.0084 | -0.0267 | 0.589 [0.546, 0.632] | 0.645 | 0.0000 | 0.355 |
| probe_lr | -0.0321 | -0.1793 | 0.715 [0.679, 0.750] | 0.726 | 0.0000 | 0.186 |
| probe_mlp | -0.0303 | -0.1282 | 0.703 [0.667, 0.739] | 0.713 | 0.0000 | 0.190 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.20, R→W 1.78; flip AUROC from length increase alone 0.557 [0.511, 0.603].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 664 | 131 | 0.650 | 0.298 |
| keyword_carrier | 336 | 28 | 0.681 | 0.286 |
| column_synonym | 401 | 109 | 0.584 | 0.321 |
| column_carrier | 419 | 90 | 0.613 | 0.378 |
| column_attribute | 95 | 35 | 0.610 | 0.314 |
| column_value | 231 | 89 | 0.500 | 0.461 |
| value_synonym | 352 | 134 | 0.538 | 0.418 |
| multitype | 992 | 332 | 0.616 | 0.340 |
| others | 2052 | 267 | 0.605 | 0.352 |

Figures: `figures/`.
