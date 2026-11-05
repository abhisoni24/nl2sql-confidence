# cp2_e3_qwen2.5-coder-7b: auto-generated results summary (label `ts_ok`)

Model `Qwen/Qwen2.5-Coder-7B-Instruct` (rev `c03e6d35`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_column_synonym, NLQ_value_synonym. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 66.5% | 56.6% | 10.0 | 127 | 32 | 1.35e-14 |
| column_synonym | 563 | 64.3% | 46.9% | 17.4 | 115 | 17 | 4.62e-19 |
| value_synonym | 506 | 68.0% | 44.3% | 23.7 | 128 | 8 | 5.76e-29 |
| **pooled** | 2022 | 66.3% | 50.8% | 15.5 | 370 | 57 | 2.88e-57 |

Pooled drop 95% CI: [12.4, 18.6] pts (615 distinct clean questions). Identical SQL for both twins in 43.3% of pairs. No closing fence (cap or loop guard): clean 0.0%, perturbed 0.1%. Execution errors: clean 4.5%, perturbed 5.4%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.719 [0.670, 0.765] | 0.620 [0.582, 0.656] |
| p_min | 0.693 [0.641, 0.746] | 0.640 [0.599, 0.679] |
| seq_logprob | 0.817 [0.773, 0.857] | 0.740 [0.702, 0.776] |
| neg_entropy | 0.729 [0.678, 0.775] | 0.640 [0.603, 0.677] |
| sc_agree | 0.809 [0.765, 0.850] | 0.712 [0.679, 0.741] |
| sc_majority | 0.806 [0.761, 0.848] | 0.711 [0.678, 0.740] |
| sc_neg_entropy | 0.802 [0.759, 0.844] | 0.703 [0.669, 0.731] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.960 | 0.663 | 0.297 | 0.304 |
| p_gm | perturbed | 0.954 | 0.508 | 0.446 | 0.444 |
| sc_agree | clean | 0.861 | 0.663 | 0.198 | 0.191 |
| sc_agree | perturbed | 0.821 | 0.508 | 0.314 | 0.316 |
| sc_majority | clean | 0.894 | 0.663 | 0.231 | 0.210 |
| sc_majority | perturbed | 0.853 | 0.508 | 0.345 | 0.329 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 1340 clean-correct pairs, 370 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0058 | -0.0161 | 0.617 [0.566, 0.664] | 0.591 | 0.0000 | 0.262 |
| p_min | -0.0330 | -0.0769 | 0.559 [0.513, 0.607] | 0.608 | 0.0008 | 0.341 |
| seq_logprob | -0.1245 | -0.6251 | 0.661 [0.611, 0.709] | 0.658 | 0.0000 | 0.254 |
| neg_entropy | -0.0120 | -0.0334 | 0.642 [0.596, 0.689] | 0.609 | 0.0000 | 0.224 |
| sc_agree | -0.0313 | -0.1659 | 0.623 [0.586, 0.663] | 0.650 | 0.0000 | 0.592 |
| sc_majority | -0.0279 | -0.1276 | 0.620 [0.584, 0.657] | 0.649 | 0.0000 | 0.595 |
| sc_neg_entropy | -0.0601 | -0.2410 | 0.603 [0.566, 0.642] | 0.646 | 0.0000 | 0.608 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.55, R→W 3.58; flip AUROC from length increase alone 0.599 [0.543, 0.656].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 634 | 127 | 0.645 | 0.252 |
| column_synonym | 362 | 115 | 0.568 | 0.270 |
| value_synonym | 344 | 128 | 0.626 | 0.266 |

Figures: `figures/`.
