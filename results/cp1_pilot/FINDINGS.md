# CP1 findings (pilot, 2026-10-05)

Hand-written reading of `SUMMARY.md` / `metrics.json`. Every number is copied from those
files; `scripts/verify_results.py` recomputes the headline ones independently (all match).

**Setup.** Qwen2.5-Coder-0.5B-Instruct, greedy, CPU. 300 Dr.Spider NLQ pairs
(100 each: keyword_synonym, column_synonym, value_synonym; at most one rewrite per clean
question per set). 569 unique generations in about 52 minutes on 2 CPU cores.

## Five observations

**1. Rewrites hurt.** Pooled EX 39.3% clean vs 26.7% perturbed (drop 12.7 pts,
95% CI [8.0, 17.3]). Discordant pairs 50 R→W vs 12 W→R, exact McNemar p = 1.2e-6.
Per set: column_synonym −17.0 (p = 2.2e-4), value_synonym −12.0 (p = 0.004),
keyword_synonym −9.0 (p = 0.108, not significant at n = 100).

**2. Confidence ranks answers worse after a rewrite.** AUROC (correct vs wrong):

| score | clean | perturbed |
|---|---|---|
| p_gm (primary) | 0.704 [0.651, 0.762] | 0.614 [0.543, 0.685] |
| seq_logprob | 0.824 | 0.773 |
| p_min | 0.735 | 0.714 |
| neg_entropy | 0.667 | 0.598 |

All four scores lose discrimination on perturbed inputs.

**3. Confidence does not follow the accuracy drop.** Mean p_gm 0.861 → 0.850 (−0.011)
while accuracy fell 0.393 → 0.267 (−0.127). ECE 0.468 → 0.584. Caveat: p_gm is an average
token probability, not a calibrated P(correct); the gap mainly says it cannot be read as one.

**4. Core test: when a rewrite breaks a correct answer, p_gm barely moves.**
118 clean-correct pairs, 50 flip to wrong.

| score | mean Δ R→R | mean Δ R→W | flip AUROC from −Δ [95% CI] | MW p | conf did not drop |
|---|---|---|---|---|---|
| p_gm | −0.021 | −0.030 | 0.540 [0.421, 0.646] | 0.46 | 32% of flips |
| seq_logprob | −0.56 | −2.16 | 0.617 [0.500, 0.736] | 0.030 | 30% |

p_gm cannot tell a broken answer from a surviving one (CI covers 0.5). seq_logprob shows a
weak signal; output-length change alone gives 0.547 [0.432, 0.671], so length explains part
but not obviously all of it. Not settled at this n.

**5. Absolute perturbed-side confidence beats the paired change.** Flip AUROC from
perturbed confidence alone: p_min 0.676, seq_logprob 0.649, p_gm 0.594 (vs 0.577 / 0.617 /
0.540 from Δ). The clean twin adds little beyond the perturbed score itself.

## Examples (R→W flips where p_gm went *up*)

```
value_synonym   Δp_gm = +0.054
q : What are the number of votes from state 'NY' or 'CA'?
q': Show the number of vote from state New York (or California).
pred(q') : ... WHERE state = 'New York' OR state = 'California'      (DB stores 'NY', 'CA')

keyword_synonym Δp_gm = +0.037
q : Which district has both stores with less than 3000 ... and stores with more than 10000 ...?
q': Show me the districts that have stores with fewer than 3000 products, and those with over 10000
pred(q') : ... UNION ...        (gold: INTERSECT; arguably the rewrite itself is ambiguous)
```

## Failure modes worth noting

- **Confident loops.** 14 / 569 decodes (2.5%) hit the 192-token cap repeating a subquery;
  their mean p_gm is 0.901 vs 0.855 for the rest. Repetition looks like certainty.
- **Schema linking dominates errors.** Execution errors in 42.3% (clean) / 45.3% (perturbed)
  of pairs; "no such column" is 207 of the 246 error predictions.

## Limits (say these in class)

0.5B model; 300 pairs, so only 50 flips and wide CIs; single-DB execution match can accept
wrong SQL (e.g. a bare `MIN(date), other_details` that happens to return the right row);
27 clean questions appear in two or three sets, so pooled pairs are not fully independent; some
Dr.Spider rewrites may change meaning (the INTERSECT example), a label-quality question we
plan to audit in CP2.
