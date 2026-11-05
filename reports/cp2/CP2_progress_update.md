# Progress update: can a text-to-SQL model tell when rewording a question has broken its answer?

ECE/CS 559 · Abhishek and Shubham Harale · November 5, 2026 · Code: https://github.com/abhisoni24/nl2sql-confidence

**What we did.** Text-to-SQL models turn a question into a database query. Asking the same question in different words should give the same answer, but often does not. We test whether the model's confidence warns us when that happens. Using the Dr.Spider benchmark, we took 7,593 pairs of an original and a reworded question that share the same correct query (1,020 original questions, 9 kinds of rewording) and ran six open models: Qwen2.5-Coder at five sizes (0.5B to 14B parameters) and DeepSeek-Coder (6.7B). An answer counts as correct if its query returns the right result. We compared four confidence signals: the model's own word probabilities, a corrected (recalibrated) version of them, a small neural network trained to read the model's internal activity, and agreement among 10 sampled answers. The *warning score* asks whether confidence falls more when rewording turns a correct answer wrong than when it stays correct (0.5 = coin flip, 1.0 = perfect warning).

## Results so far

| Model | Accuracy (original → reworded) | Stricter check (original → reworded) | Warning score: model's own probabilities | Warning score: small network |
|---|---|---|---|---|
| Qwen-0.5B | 46.9% → 37.4% | 39.3% → 29.8% | 0.59  (0.55 to 0.62) | 0.68  (0.64 to 0.72) |
| Qwen-1.5B | 60.8% → 51.7% | 50.8% → 41.3% | 0.64  (0.60 to 0.68) | 0.69  (0.67 to 0.72) |
| Qwen-3B | 67.5% → 57.0% | 59.5% → 47.9% | 0.63  (0.59 to 0.68) | 0.72  (0.68 to 0.75) |
| Qwen-7B | 77.5% → 66.3% | 70.9% → 57.3% | 0.65  (0.62 to 0.69) | 0.72  (0.69 to 0.76) |
| DeepSeek-6.7B | 63.8% → 50.4% | 56.6% → 41.6% | 0.57  (0.53 to 0.60) | 0.81  (0.78 to 0.83) |
| Qwen-14B | 81.4% → 68.8% | 73.0% → 59.0% | 0.61  (0.57 to 0.66) | 0.72  (0.68 to 0.75) |

Ranges in brackets are 95% confidence intervals. The stricter check reruns each query on about 30 to 60 altered copies of the database.

![figure](cp2_update_figure.png)

Lines: Qwen2.5-Coder; diamonds: DeepSeek-Coder-6.7B. Bars: 95% confidence intervals.

## What we found and what it means

- **Bigger models are more accurate but no harder to throw off.** Accuracy on original questions rises from 47% to 81%, yet rewording costs 9 to 13 points at every size. Scaling up does not fix sensitivity to phrasing.
- **The model's own confidence is a weak alarm.** Its warning score is only 0.57 to 0.65, and in 25 to 37% of the cases where rewording broke a correct answer, confidence did not drop at all. Users cannot rely on it to flag these errors.
- **The warning signal exists inside the model.** A small network that reads the model's internal activity, trained only on original questions and tested on unseen databases, scores 0.68 to 0.81 and beats the model's own confidence for every model, so a learned confidence check is a practical safeguard.
- **Corrections tuned on standard questions do not fully carry over.** After recalibration, the gap between stated confidence and actual accuracy is 0.02 to 0.05 on original questions but 0.07 to 0.14 on reworded ones (lower is better). Confidence should be checked on reworded inputs, not only on the standard test set.
- **Asking the model 10 times helps rank answers, not catch rewording failures.** For the 7B model, agreement among samples separates right from wrong answers best (0.83 vs 0.72 for its own probabilities), but it warns no better (0.63 vs 0.62) at ten times the cost.

## Problems encountered

- **Answers that are right by accident.** Checking a query on a single database overstates accuracy by 7 to 10 points; we report the stricter check too.
- **Rewordings are not independent.** One original question has up to 37 rewordings (typically 6), so we compute uncertainty by resampling whole questions, not single pairs.
- **Longer answers look less confident.** The probability of a whole query falls as the query gets longer, and longer answers are more often wrong; answer length alone scores 0.55 to 0.66, so part of that signal is just length.
- **Coarse agreement scores.** With 10 samples, agreement moves in 10% steps, so many cases tie.
- **Runaway outputs.** Small models sometimes repeated text to the length limit (2.5% in our pilot); a stopping rule now caps this at 1%.

## Experiments that remain

- **Test a second benchmark besides Dr.Spider** with the same six models: Spider-Syn first (1,034 original/reworded pairs with the same correct query), then a dataset built on different databases, to check that the findings are not specific to one benchmark.
- Separate the effect of answer length from genuine confidence.
- Test the small network on kinds of rewording, and on database and query changes, it never saw in training.
- Hand-check about 100 broken answers to see how often a rewording actually changed the meaning.
- Final presentation (Dec 1 or 3) and final paper with code (Dec 4).
