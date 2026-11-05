# cp2_e1_qwen2.5-coder-1.5b: auto-generated results summary (label `exec_ok`)

Model `Qwen/Qwen2.5-Coder-1.5B-Instruct` (rev `2e1fd397`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 61.5% | 53.3% | 8.2 | 118 | 40 | 3.97e-10 |
| keyword_carrier | 399 | 80.2% | 75.9% | 4.3 | 29 | 12 | 0.0115 |
| column_synonym | 563 | 53.3% | 41.9% | 11.4 | 100 | 36 | 3.72e-08 |
| column_carrier | 579 | 58.9% | 53.0% | 5.9 | 81 | 47 | 0.00337 |
| column_attribute | 119 | 37.8% | 38.7% | -0.8 | 12 | 13 | 1 |
| column_value | 304 | 61.5% | 44.7% | 16.8 | 68 | 17 | 2.18e-08 |
| value_synonym | 506 | 53.4% | 38.1% | 15.2 | 95 | 18 | 7.97e-14 |
| multitype | 1351 | 52.8% | 40.4% | 12.4 | 246 | 78 | 2.03e-21 |
| others | 2819 | 65.8% | 58.5% | 7.3 | 315 | 108 | 1.44e-24 |
| **pooled** | 7593 | 60.8% | 51.7% | 9.2 | 1064 | 369 | 3.11e-78 |

Pooled drop 95% CI: [6.9, 11.5] pts (1020 distinct clean questions). Identical SQL for both twins in 44.5% of pairs. No closing fence (cap or loop guard): clean 0.0%, perturbed 0.1%. Execution errors: clean 20.3%, perturbed 21.2%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.599 [0.553, 0.640] | 0.606 [0.576, 0.634] |
| p_min | 0.726 [0.689, 0.763] | 0.693 [0.669, 0.716] |
| seq_logprob | 0.801 [0.768, 0.829] | 0.768 [0.743, 0.790] |
| neg_entropy | 0.544 [0.494, 0.587] | 0.556 [0.525, 0.584] |
| p_gm_platt | 0.570 [0.527, 0.610] | 0.584 [0.554, 0.612] |
| probe_lr | 0.781 [0.745, 0.814] | 0.758 [0.733, 0.784] |
| probe_mlp | 0.798 [0.768, 0.830] | 0.755 [0.731, 0.781] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.904 | 0.608 | 0.296 | 0.320 |
| p_gm | perturbed | 0.895 | 0.517 | 0.378 | 0.387 |
| p_gm_platt | clean | 0.637 | 0.608 | 0.037 | 0.236 |
| p_gm_platt | perturbed | 0.618 | 0.517 | 0.105 | 0.255 |
| probe_lr | clean | 0.607 | 0.608 | 0.157 | 0.211 |
| probe_lr | perturbed | 0.543 | 0.517 | 0.165 | 0.231 |
| probe_mlp | clean | 0.611 | 0.608 | 0.069 | 0.181 |
| probe_mlp | perturbed | 0.572 | 0.517 | 0.108 | 0.212 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 4618 clean-correct pairs, 1064 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0082 | -0.0257 | 0.640 [0.602, 0.681] | 0.652 | 0.0000 | 0.245 |
| p_min | -0.0207 | -0.0691 | 0.584 [0.545, 0.625] | 0.684 | 0.0000 | 0.360 |
| seq_logprob | -0.2427 | -1.3891 | 0.687 [0.646, 0.729] | 0.719 | 0.0000 | 0.239 |
| neg_entropy | -0.0245 | -0.0678 | 0.638 [0.599, 0.675] | 0.617 | 0.0000 | 0.232 |
| p_gm_platt | -0.0164 | -0.0475 | 0.636 [0.598, 0.677] | 0.642 | 0.0000 | 0.245 |
| probe_lr | -0.0547 | -0.2220 | 0.694 [0.665, 0.724] | 0.701 | 0.0000 | 0.172 |
| probe_mlp | -0.0392 | -0.1496 | 0.687 [0.647, 0.725] | 0.690 | 0.0000 | 0.201 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.59, R→W 5.48; flip AUROC from length increase alone 0.582 [0.538, 0.628].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 586 | 118 | 0.691 | 0.220 |
| keyword_carrier | 320 | 29 | 0.771 | 0.103 |
| column_synonym | 300 | 100 | 0.621 | 0.220 |
| column_carrier | 341 | 81 | 0.642 | 0.185 |
| column_attribute | 45 | 12 | 0.715 | 0.250 |
| column_value | 187 | 68 | 0.583 | 0.265 |
| value_synonym | 270 | 95 | 0.591 | 0.284 |
| multitype | 714 | 246 | 0.617 | 0.264 |
| others | 1855 | 315 | 0.636 | 0.260 |

Figures: `figures/`.
