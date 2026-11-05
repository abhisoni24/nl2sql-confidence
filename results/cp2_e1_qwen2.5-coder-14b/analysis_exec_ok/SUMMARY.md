# cp2_e1_qwen2.5-coder-14b: auto-generated results summary (label `exec_ok`)

Model `Qwen/Qwen2.5-Coder-14B-Instruct` (rev `aedcc2d4`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 77.1% | 66.6% | 10.5 | 118 | 18 | 3.29e-19 |
| keyword_carrier | 399 | 87.2% | 83.5% | 3.8 | 19 | 4 | 0.0026 |
| column_synonym | 563 | 78.7% | 60.9% | 17.8 | 113 | 13 | 4.52e-21 |
| column_carrier | 579 | 85.1% | 73.2% | 11.9 | 80 | 11 | 4.41e-14 |
| column_attribute | 119 | 89.9% | 63.0% | 26.9 | 32 | 0 | 4.66e-10 |
| column_value | 304 | 85.9% | 65.1% | 20.7 | 69 | 6 | 1.16e-14 |
| value_synonym | 506 | 76.3% | 53.4% | 22.9 | 129 | 13 | 3.46e-25 |
| multitype | 1351 | 81.3% | 60.0% | 21.3 | 305 | 17 | 1.95e-69 |
| others | 2819 | 81.8% | 75.8% | 6.1 | 242 | 71 | 5.5e-23 |
| **pooled** | 7593 | 81.4% | 68.8% | 12.6 | 1107 | 153 | 8.75e-179 |

Pooled drop 95% CI: [10.7, 14.6] pts (1020 distinct clean questions). Identical SQL for both twins in 52.8% of pairs. No closing fence (cap or loop guard): clean 0.0%, perturbed 0.0%. Execution errors: clean 0.4%, perturbed 0.9%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.625 [0.572, 0.679] | 0.635 [0.602, 0.667] |
| p_min | 0.647 [0.593, 0.700] | 0.641 [0.609, 0.673] |
| seq_logprob | 0.719 [0.664, 0.769] | 0.701 [0.665, 0.732] |
| neg_entropy | 0.651 [0.596, 0.707] | 0.659 [0.627, 0.689] |
| p_gm_platt | 0.602 [0.550, 0.651] | 0.619 [0.587, 0.650] |
| probe_lr | 0.771 [0.729, 0.811] | 0.731 [0.704, 0.758] |
| probe_mlp | 0.739 [0.696, 0.778] | 0.712 [0.683, 0.740] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.963 | 0.814 | 0.149 | 0.171 |
| p_gm | perturbed | 0.959 | 0.688 | 0.270 | 0.283 |
| p_gm_platt | clean | 0.816 | 0.814 | 0.015 | 0.148 |
| p_gm_platt | perturbed | 0.806 | 0.688 | 0.118 | 0.221 |
| probe_lr | clean | 0.830 | 0.814 | 0.119 | 0.150 |
| probe_lr | perturbed | 0.771 | 0.688 | 0.179 | 0.223 |
| probe_mlp | clean | 0.854 | 0.814 | 0.088 | 0.143 |
| probe_mlp | perturbed | 0.807 | 0.688 | 0.144 | 0.213 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 6179 clean-correct pairs, 1107 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0031 | -0.0147 | 0.614 [0.569, 0.662] | 0.665 | 0.0000 | 0.346 |
| p_min | -0.0115 | -0.0432 | 0.543 [0.500, 0.587] | 0.630 | 0.0000 | 0.413 |
| seq_logprob | -0.0623 | -0.4616 | 0.635 [0.591, 0.680] | 0.673 | 0.0000 | 0.333 |
| neg_entropy | -0.0083 | -0.0329 | 0.635 [0.590, 0.681] | 0.683 | 0.0000 | 0.300 |
| p_gm_platt | -0.0082 | -0.0290 | 0.597 [0.551, 0.644] | 0.656 | 0.0000 | 0.346 |
| probe_lr | -0.0373 | -0.1847 | 0.715 [0.679, 0.751] | 0.709 | 0.0000 | 0.173 |
| probe_mlp | -0.0326 | -0.1255 | 0.690 [0.649, 0.729] | 0.693 | 0.0000 | 0.193 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.12, R→W 1.22; flip AUROC from length increase alone 0.547 [0.501, 0.589].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 735 | 118 | 0.679 | 0.271 |
| keyword_carrier | 348 | 19 | 0.694 | 0.263 |
| column_synonym | 443 | 113 | 0.588 | 0.310 |
| column_carrier | 493 | 80 | 0.631 | 0.388 |
| column_attribute | 107 | 32 | 0.583 | 0.344 |
| column_value | 261 | 69 | 0.547 | 0.464 |
| value_synonym | 386 | 129 | 0.547 | 0.403 |
| multitype | 1099 | 305 | 0.617 | 0.338 |
| others | 2307 | 242 | 0.619 | 0.339 |

Figures: `figures/`.
