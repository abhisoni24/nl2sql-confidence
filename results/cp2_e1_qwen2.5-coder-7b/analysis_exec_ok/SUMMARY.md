# cp2_e1_qwen2.5-coder-7b: auto-generated results summary (label `exec_ok`)

Model `Qwen/Qwen2.5-Coder-7B-Instruct` (rev `c03e6d35`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 72.4% | 63.4% | 9.0 | 116 | 30 | 3.92e-13 |
| keyword_carrier | 399 | 83.7% | 80.7% | 3.0 | 21 | 9 | 0.0428 |
| column_synonym | 563 | 68.0% | 54.2% | 13.9 | 102 | 24 | 1.21e-12 |
| column_carrier | 579 | 87.2% | 74.3% | 13.0 | 85 | 10 | 5.76e-16 |
| column_attribute | 119 | 70.6% | 69.7% | 0.8 | 19 | 18 | 1 |
| column_value | 304 | 79.6% | 59.5% | 20.1 | 68 | 7 | 1.17e-13 |
| value_synonym | 506 | 71.5% | 50.0% | 21.5 | 118 | 9 | 2.25e-25 |
| multitype | 1351 | 76.1% | 59.1% | 17.0 | 279 | 49 | 3.16e-40 |
| others | 2819 | 79.9% | 73.0% | 7.0 | 272 | 76 | 5.27e-27 |
| **pooled** | 7593 | 77.5% | 66.3% | 11.2 | 1080 | 232 | 5.66e-131 |

Pooled drop 95% CI: [8.7, 13.2] pts (1020 distinct clean questions). Identical SQL for both twins in 49.4% of pairs. No closing fence (cap or loop guard): clean 0.0%, perturbed 0.0%. Execution errors: clean 3.9%, perturbed 3.9%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.708 [0.659, 0.751] | 0.634 [0.606, 0.663] |
| p_min | 0.699 [0.657, 0.736] | 0.659 [0.630, 0.687] |
| seq_logprob | 0.809 [0.768, 0.846] | 0.733 [0.703, 0.762] |
| neg_entropy | 0.715 [0.667, 0.759] | 0.648 [0.620, 0.676] |
| p_gm_platt | 0.682 [0.635, 0.726] | 0.621 [0.593, 0.651] |
| probe_lr | 0.838 [0.808, 0.870] | 0.763 [0.738, 0.788] |
| probe_mlp | 0.799 [0.760, 0.833] | 0.727 [0.697, 0.754] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.961 | 0.775 | 0.186 | 0.203 |
| p_gm | perturbed | 0.955 | 0.663 | 0.292 | 0.304 |
| p_gm_platt | clean | 0.796 | 0.775 | 0.049 | 0.164 |
| p_gm_platt | perturbed | 0.778 | 0.663 | 0.115 | 0.228 |
| probe_lr | clean | 0.801 | 0.775 | 0.094 | 0.137 |
| probe_lr | perturbed | 0.729 | 0.663 | 0.165 | 0.212 |
| probe_mlp | clean | 0.753 | 0.775 | 0.065 | 0.145 |
| probe_mlp | perturbed | 0.705 | 0.663 | 0.101 | 0.206 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 5881 clean-correct pairs, 1080 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0046 | -0.0183 | 0.651 [0.617, 0.685] | 0.629 | 0.0000 | 0.251 |
| p_min | -0.0261 | -0.0855 | 0.604 [0.566, 0.641] | 0.639 | 0.0000 | 0.302 |
| seq_logprob | -0.1214 | -0.7639 | 0.695 [0.660, 0.729] | 0.682 | 0.0000 | 0.226 |
| neg_entropy | -0.0106 | -0.0399 | 0.681 [0.649, 0.714] | 0.646 | 0.0000 | 0.211 |
| p_gm_platt | -0.0161 | -0.0586 | 0.649 [0.614, 0.683] | 0.626 | 0.0000 | 0.251 |
| probe_lr | -0.0439 | -0.2519 | 0.724 [0.689, 0.757] | 0.725 | 0.0000 | 0.161 |
| probe_mlp | -0.0395 | -0.1344 | 0.675 [0.635, 0.713] | 0.685 | 0.0000 | 0.193 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.46, R→W 4.69; flip AUROC from length increase alone 0.607 [0.564, 0.648].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 690 | 116 | 0.654 | 0.233 |
| keyword_carrier | 334 | 21 | 0.714 | 0.238 |
| column_synonym | 383 | 102 | 0.568 | 0.265 |
| column_carrier | 505 | 85 | 0.631 | 0.247 |
| column_attribute | 84 | 19 | 0.599 | 0.211 |
| column_value | 242 | 68 | 0.640 | 0.309 |
| value_synonym | 362 | 118 | 0.632 | 0.263 |
| multitype | 1028 | 279 | 0.659 | 0.201 |
| others | 2253 | 272 | 0.657 | 0.290 |

Figures: `figures/`.
