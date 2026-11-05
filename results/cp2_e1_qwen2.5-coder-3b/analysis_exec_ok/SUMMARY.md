# cp2_e1_qwen2.5-coder-3b: auto-generated results summary (label `exec_ok`)

Model `Qwen/Qwen2.5-Coder-3B-Instruct` (rev `488639f1`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 62.3% | 55.0% | 7.3 | 105 | 35 | 2.56e-09 |
| keyword_carrier | 399 | 83.0% | 78.9% | 4.0 | 24 | 8 | 0.007 |
| column_synonym | 563 | 56.7% | 42.1% | 14.6 | 107 | 25 | 2.8e-13 |
| column_carrier | 579 | 71.5% | 63.4% | 8.1 | 86 | 39 | 3.18e-05 |
| column_attribute | 119 | 55.5% | 47.1% | 8.4 | 22 | 12 | 0.121 |
| column_value | 304 | 71.7% | 50.0% | 21.7 | 75 | 9 | 4.31e-14 |
| value_synonym | 506 | 64.0% | 43.9% | 20.2 | 110 | 8 | 4.74e-24 |
| multitype | 1351 | 66.2% | 48.9% | 17.3 | 279 | 45 | 2.23e-42 |
| others | 2819 | 69.6% | 63.7% | 5.9 | 294 | 127 | 2.41e-16 |
| **pooled** | 7593 | 67.5% | 57.0% | 10.5 | 1102 | 308 | 6.95e-105 |

Pooled drop 95% CI: [8.2, 12.7] pts (1020 distinct clean questions). Identical SQL for both twins in 44.6% of pairs. No closing fence (cap or loop guard): clean 0.1%, perturbed 0.0%. Execution errors: clean 12.2%, perturbed 13.3%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.663 [0.620, 0.709] | 0.647 [0.620, 0.673] |
| p_min | 0.708 [0.668, 0.746] | 0.696 [0.672, 0.722] |
| seq_logprob | 0.805 [0.767, 0.837] | 0.772 [0.745, 0.797] |
| neg_entropy | 0.647 [0.605, 0.691] | 0.634 [0.608, 0.661] |
| p_gm_platt | 0.636 [0.593, 0.678] | 0.627 [0.600, 0.653] |
| probe_lr | 0.827 [0.796, 0.857] | 0.791 [0.768, 0.812] |
| probe_mlp | 0.835 [0.804, 0.863] | 0.783 [0.761, 0.805] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.932 | 0.675 | 0.257 | 0.279 |
| p_gm | perturbed | 0.922 | 0.570 | 0.352 | 0.362 |
| p_gm_platt | clean | 0.679 | 0.675 | 0.030 | 0.210 |
| p_gm_platt | perturbed | 0.645 | 0.570 | 0.091 | 0.240 |
| probe_lr | clean | 0.691 | 0.675 | 0.116 | 0.174 |
| probe_lr | perturbed | 0.620 | 0.570 | 0.150 | 0.209 |
| probe_mlp | clean | 0.698 | 0.675 | 0.066 | 0.155 |
| probe_mlp | perturbed | 0.646 | 0.570 | 0.103 | 0.200 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 5123 clean-correct pairs, 1102 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0079 | -0.0242 | 0.632 [0.592, 0.676] | 0.646 | 0.0000 | 0.281 |
| p_min | -0.0224 | -0.0637 | 0.571 [0.527, 0.610] | 0.670 | 0.0000 | 0.368 |
| seq_logprob | -0.1890 | -1.2693 | 0.703 [0.661, 0.744] | 0.712 | 0.0000 | 0.244 |
| neg_entropy | -0.0204 | -0.0576 | 0.665 [0.628, 0.705] | 0.634 | 0.0000 | 0.201 |
| p_gm_platt | -0.0300 | -0.0829 | 0.627 [0.586, 0.671] | 0.640 | 0.0000 | 0.281 |
| probe_lr | -0.0501 | -0.2763 | 0.716 [0.679, 0.751] | 0.749 | 0.0000 | 0.190 |
| probe_mlp | -0.0399 | -0.1790 | 0.689 [0.658, 0.720] | 0.717 | 0.0000 | 0.202 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.03, R→W 5.61; flip AUROC from length increase alone 0.601 [0.557, 0.645].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 594 | 105 | 0.637 | 0.305 |
| keyword_carrier | 331 | 24 | 0.825 | 0.083 |
| column_synonym | 319 | 107 | 0.637 | 0.262 |
| column_carrier | 414 | 86 | 0.649 | 0.140 |
| column_attribute | 66 | 22 | 0.598 | 0.182 |
| column_value | 218 | 75 | 0.604 | 0.307 |
| value_synonym | 324 | 110 | 0.519 | 0.409 |
| multitype | 894 | 279 | 0.613 | 0.276 |
| others | 1963 | 294 | 0.637 | 0.296 |

Figures: `figures/`.
