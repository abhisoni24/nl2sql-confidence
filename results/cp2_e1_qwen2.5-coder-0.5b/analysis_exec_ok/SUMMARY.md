# cp2_e1_qwen2.5-coder-0.5b: auto-generated results summary (label `exec_ok`)

Model `Qwen/Qwen2.5-Coder-0.5B-Instruct` (rev `ea3f2471`), greedy decoding. Sets: NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others. Seed 42. 95% CIs: cluster bootstrap over `spider_qid`, 1000 resamples. Every number below is computed by `scripts/analyze.py`.

## 1. Robustness (pair level)

| set | pairs | EX clean | EX perturbed | drop (pts) | R→W | W→R | McNemar p |
|---|---|---|---|---|---|---|---|
| keyword_synonym | 953 | 45.5% | 38.0% | 7.6 | 127 | 55 | 9.76e-08 |
| keyword_carrier | 399 | 63.2% | 62.4% | 0.8 | 25 | 22 | 0.771 |
| column_synonym | 563 | 36.8% | 25.4% | 11.4 | 86 | 22 | 3.99e-10 |
| column_carrier | 579 | 50.3% | 40.1% | 10.2 | 109 | 50 | 3.31e-06 |
| column_attribute | 119 | 50.4% | 40.3% | 10.1 | 21 | 9 | 0.0428 |
| column_value | 304 | 53.6% | 35.9% | 17.8 | 60 | 6 | 2.73e-12 |
| value_synonym | 506 | 33.2% | 17.8% | 15.4 | 96 | 18 | 4.72e-14 |
| multitype | 1351 | 45.0% | 30.3% | 14.7 | 272 | 74 | 9.85e-28 |
| others | 2819 | 48.9% | 42.5% | 6.4 | 344 | 163 | 6.44e-16 |
| **pooled** | 7593 | 46.9% | 37.4% | 9.5 | 1140 | 419 | 4.16e-77 |

Pooled drop 95% CI: [7.0, 12.1] pts (1020 distinct clean questions). Identical SQL for both twins in 31.9% of pairs. No closing fence (cap or loop guard): clean 1.0%, perturbed 1.0%. Execution errors: clean 36.1%, perturbed 37.1%.

## 2. Discrimination: AUROC (correct vs wrong), pooled

| confidence | clean [95% CI] | perturbed [95% CI] |
|---|---|---|
| p_gm | 0.671 [0.630, 0.713] | 0.633 [0.607, 0.662] |
| p_min | 0.735 [0.693, 0.772] | 0.713 [0.687, 0.737] |
| seq_logprob | 0.822 [0.786, 0.853] | 0.803 [0.778, 0.828] |
| neg_entropy | 0.640 [0.597, 0.682] | 0.613 [0.585, 0.644] |
| p_gm_platt | 0.656 [0.615, 0.698] | 0.617 [0.591, 0.647] |
| probe_lr | 0.754 [0.716, 0.792] | 0.743 [0.718, 0.770] |
| probe_mlp | 0.763 [0.724, 0.799] | 0.736 [0.707, 0.764] |

## 3. Calibration, pooled

| confidence | side | mean conf | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| p_gm | clean | 0.867 | 0.469 | 0.397 | 0.394 |
| p_gm | perturbed | 0.855 | 0.374 | 0.481 | 0.457 |
| p_gm_platt | clean | 0.475 | 0.469 | 0.037 | 0.232 |
| p_gm_platt | perturbed | 0.444 | 0.374 | 0.072 | 0.232 |
| probe_lr | clean | 0.416 | 0.469 | 0.189 | 0.240 |
| probe_lr | perturbed | 0.346 | 0.374 | 0.176 | 0.232 |
| probe_mlp | clean | 0.468 | 0.469 | 0.066 | 0.203 |
| probe_mlp | perturbed | 0.421 | 0.374 | 0.086 | 0.208 |

## 4. Core test: does confidence drop when a rewrite breaks a correct answer?

Among 3562 clean-correct pairs, 1140 flip to wrong (R→W).

| confidence | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | flip AUROC from perturbed conf | Mann-Whitney p | blind-spot rate |
|---|---|---|---|---|---|---|
| p_gm | -0.0124 | -0.0273 | 0.587 [0.549, 0.624] | 0.611 | 0.0000 | 0.310 |
| p_min | -0.0239 | -0.0589 | 0.589 [0.553, 0.623] | 0.685 | 0.0000 | 0.326 |
| seq_logprob | -0.2902 | -1.9444 | 0.701 [0.657, 0.744] | 0.754 | 0.0000 | 0.232 |
| neg_entropy | -0.0341 | -0.0731 | 0.600 [0.562, 0.635] | 0.599 | 0.0000 | 0.277 |
| p_gm_platt | -0.0388 | -0.0705 | 0.576 [0.538, 0.613] | 0.606 | 0.0000 | 0.310 |
| probe_lr | -0.0622 | -0.2494 | 0.678 [0.641, 0.715] | 0.698 | 0.0000 | 0.188 |
| probe_mlp | -0.0396 | -0.1815 | 0.708 [0.668, 0.746] | 0.687 | 0.0000 | 0.176 |

Blind-spot rate = share of R→W flips whose confidence did not decrease. Length control: mean Δ output tokens R→R 0.42, R→W 8.91; flip AUROC from length increase alone 0.645 [0.601, 0.692].

## 5. Flip detection by perturbation set (E4), primary score

| set | clean-correct | flips | flip AUROC (p_gm) | blind-spot rate |
|---|---|---|---|---|
| keyword_synonym | 434 | 127 | 0.585 | 0.354 |
| keyword_carrier | 252 | 25 | 0.813 | 0.120 |
| column_synonym | 207 | 86 | 0.511 | 0.337 |
| column_carrier | 291 | 109 | 0.552 | 0.239 |
| column_attribute | 60 | 21 | 0.621 | 0.143 |
| column_value | 163 | 60 | 0.512 | 0.400 |
| value_synonym | 168 | 96 | 0.570 | 0.271 |
| multitype | 608 | 272 | 0.544 | 0.309 |
| others | 1379 | 344 | 0.594 | 0.328 |

Figures: `figures/`.
