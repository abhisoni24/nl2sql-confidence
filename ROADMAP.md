# ROADMAP: working notes for the next steps

Guiding document for whoever picks this project up next (including Claude in a new session).
`PLAN.md` holds the design and checkpoint history; this file holds **where we are, what we
believe, what is still unproven, and what to do next**. Update it at the end of every work session.

Last updated: 2026-11-05, after Checkpoint 2.

---

## 1. One-paragraph status

Text-to-SQL models often break when a question is reworded without changing its meaning. On
Dr.Spider (9 rewrite types, 7,593 original/reworded pairs with the same gold SQL, 1,020 original
questions) and six open models (Qwen2.5-Coder 0.5B to 14B, DeepSeek-Coder-6.7B), we found that
(1) rewording costs 9 to 13 accuracy points at every size, (2) the model's own token probabilities
are a weak warning of these breaks (warning score 0.57 to 0.65), (3) a linear probe on hidden
states, trained on original questions only, warns much better (0.68 to 0.81), and (4) calibration
fit on original questions degrades on rewrites (ECE 0.015 to 0.049 → 0.072 to 0.135). All
numbers: `results/cp2_summary/<exec_ok|ts_ok>/CP2_SUMMARY.md`.

## 2. Glossary (terms used in code vs. in the write-ups)

| code / paper term | plain term used in reports | meaning |
|---|---|---|
| pair, twin | original + reworded question | same gold SQL, different wording |
| flip, R→W | rewording broke a correct answer | right on original, wrong on rewrite |
| flip AUROC | warning score | does confidence drop more for flips than for R→R pairs (0.5 = chance) |
| `p_gm` | model's own word probabilities | geometric-mean token probability of the SQL |
| `seq_logprob` | probability of the whole query | sum of token log-probs (length-biased) |
| `probe_lr`, `probe_mlp` | small network reading internal activity | linear / MLP probe on hidden states |
| `p_gm_platt` | recalibrated confidence | Platt scaling fit on original questions |
| `sc_*` | agreement among samples | k = 10 sampled queries, execution-result agreement |
| `exec_ok` / `ts_ok` | accuracy / stricter check | single-DB execution match / test-suite match |

## 3. What is done (with evidence)

| checkpoint | what | evidence |
|---|---|---|
| CP0 | repo, conda env, configs, tests | `PLAN.md` §4, `pytest`: 21 fast + 2 slow tests |
| CP1 (Oct 5) | CPU pilot, 0.5B, 300 pairs | `results/cp1_pilot/FINDINGS.md` |
| CP2 (Nov 5) | E1 to E6 on GPU, 6 models | `results/cp2_summary/`, `reports/cp2/CP2_progress_update.pdf` |
| verification | 140/140 independent re-checks | `scripts/verify_results.py`, run on a different machine |
| code | public repo | https://github.com/abhisoni24/nl2sql-confidence (user pushes; large outputs git-ignored) |

## 4. Claims and how solid they are

| # | claim | strength | what would strengthen it |
|---|---|---|---|
| C1 | Scale raises accuracy but not robustness to rewording | solid on Dr.Spider (CIs overlap across sizes) | second dataset (G3) |
| C2 | Token-probability confidence is a weak warning | solid; 25 to 37% of flips show no confidence drop | second dataset; add rescaled full-sequence baseline (§6, N2) |
| C3 | Hidden-state probe warns better than output probabilities | promising; held-out databases, preset layer best in 5/6 models | transfer to unseen rewrite types and datasets; length control |
| C4 | Calibration fit on original questions degrades on rewrites | solid for Platt on `p_gm` | other calibrators; second dataset |
| C5 | Sample agreement ranks answers but does not warn better | weak (one model, k = 10, ties) | tie-aware metric, larger k |

## 5. Known problems and threats to validity

- **Length confound:** answer-length change alone gives warning score 0.55 to 0.66, so part of
  `seq_logprob`'s edge is length. Needs a length-controlled comparison.
- **Single benchmark:** everything so far is Dr.Spider on Spider-dev databases (instructor asked for a second dataset).
- **Label noise:** some Dr.Spider rewrites may change meaning (CP1 example: dropping "both" turns
  INTERSECT into UNION). Needs a manual audit.
- **Lucky matches:** single-DB execution overstates accuracy by 6.5 to 10.1 points; always report both labels.
- **Dependence:** up to 37 rewrites per original question (median 6); keep the cluster bootstrap.
- **Hidden-state features were lost:** they lived only on the Colab runtime, which has since stopped.
  Re-create them with a feature-only pass (§7, task T0) instead of re-running generation.

## 6. Path to a novel finding

Preliminary literature scan (2026-11-05, not a full novelty check):

- Ramachandran & Sarawagi, *Text-to-SQL Calibration: No Need to Ask, Just Rescale Model
  Probabilities* (arXiv:2411.16742): rescaled full-sequence probability beats self-check and
  verbalized confidence for text-to-SQL calibration. They do not study rewording shift.
- *Node-Level Uncertainty Estimation in LLM-Generated SQL* (arXiv:2511.13984): supervised per-node
  error classifier from schema and lexical features; not hidden states, not paired rewrites.
- *LLM NL2SQL Robustness: Surface Noise vs. Linguistic Variation* (arXiv:2603.17017): a robustness
  benchmark for frontier LLMs. It measures accuracy, not confidence.
- *Calibrating LLMs for Text-to-SQL Parsing by Leveraging Sub-clause Frequencies* (EMNLP 2025) and
  *SUN* (COLING 2022): calibration and uncertainty for text-to-SQL, not under paired rewording.

In this scan we found **no paper that measures whether text-to-SQL confidence warns of failures
caused by meaning-preserving rewrites using paired twins**. Run a proper novelty check (Semantic Scholar /
alphaXiv, forward citations of Dr.Spider and of 2411.16742) before claiming it.

Candidate contributions (our own hypotheses, ordered by feasibility):

| id | candidate finding | test | status |
|---|---|---|---|
| N1 | **Paired-twin warning score as an evaluation protocol**: confidence methods that look calibrated on standard test sets fail exactly where rewording breaks answers | already shown on Dr.Spider; needs G3 | half done |
| N2 | **Hidden states beat the strongest output-probability baseline under rewording**, including the rescaled full-sequence probability of 2411.16742 | add that baseline (Platt/temperature on `seq_logprob`), length control, second dataset | open |
| N3 | **Fragility probe** (our idea): from the original question alone, predict which correct answers will break when reworded | probe on last-prompt-token features, label = share of an original question's rewrites that flip; no new generation needed | open, cheap |
| N4 | **Which rewrite operators create blind spots**: controlled deterministic rewrites (ASSERT-style typed IR, the user's ICSE 2027 work) to attribute warning failures to specific rewrite types | needs ASSERT perturbation module on BIRD or Spider schemas | stretch |

## 7. Next tasks (prioritized)

| id | task | why | done when | est. cost |
|---|---|---|---|---|
| T0 | `scripts/extract_features.py`: teacher-forced pass over saved predictions (`labeled.jsonl` `raw_output`) to rebuild hidden-state features | features were lost with the runtime; needed by T3, T4 | features for all 6 models; probe AUROC within ±0.01 of CP2 | ~30 to 45 min GPU |
| T1 | **Second dataset (G3): Spider-Syn** loader + configs, run all 6 models through the same pipeline | instructor request; tests generality | `results/cp3_spidersyn_*` + cross-dataset table | ~40 min GPU (≤2,068 questions per model) |
| T2 | Rescaled full-sequence baseline + length control | needed for N2; answers the length confound | warning score of length-matched / length-residualized scores reported with CIs | CPU only |
| T3 | Probe transfer: leave-one-rewrite-type-out; train on Dr.Spider, test on Spider-Syn | C3 → N2 | table of transfer warning scores | CPU/GPU minutes |
| T4 | Fragility probe (N3) | potential novel result | AUROC for predicting flip-prone original questions, with CIs | CPU minutes |
| T5 | Manual audit of ~100 flips (meaning changed? gold wrong?) | label noise | audited CSV + error-rate estimate | ~2 h human |
| T6 | Third dataset on different databases (BIRD or Spider-Realistic) | stronger G3 | as T1 | see §8 |
| T7 | Tie-aware sampling analysis (E3) | C5 | tie-corrected warning score | CPU |
| T8 | Final talk slides (Dec 1/3), paper + frozen configs (Dec 4) | CP4 | `make reproduce` regenerates every table/figure | |

## 8. Second-dataset options (verified 2026-11-05)

| dataset | what it is | pairs | databases | license | fit |
|---|---|---|---|---|---|
| **Spider-Syn** (Gan et al., ACL 2021; github.com/ygan/Spider-Syn) | Spider dev questions with schema words replaced by synonyms; fields `SpiderQuestion`, `SpiderSynQuestion`, `query` | 1,034 dev | Spider dev (same as Dr.Spider) | MIT | drop-in, recommended first |
| Spider-Realistic (Deng et al., NAACL 2021; zenodo 5205322) | 508 Spider dev questions with explicit column mentions removed; pair with Spider dev by `db_id` + `query` | 508 | Spider dev | CC BY-SA 4.0 | easy, small |
| BIRD dev + our own deterministic rewrites | different, larger databases; rewrites generated by rules (no LLM), as in ASSERT | design choice | BIRD | check BIRD terms | strongest generality claim, most work |

Caveat to state in the paper: Spider-Syn and Spider-Realistic reuse Spider-dev databases, so they
test new rewordings, not new databases. Decision needed from the team: is Spider-Syn enough for the
instructor, or should T6 (different databases) be in scope?

## 9. How to run things (gotchas learned the hard way)

- **GPU:** Colab A100 driven from Claude's workspace with the Colab CLI (`uv tool install
  google-colab-cli`; OAuth copy-paste flow, token in `~/.config/colab-cli/token.json`, per session).
  Adopt an existing browser runtime by adding its endpoint to `colab_cli.state.StateStore` under a
  name. **Never run bare `colab ssh` or `colab new` without the user's OK**: they allocate a new paid runtime.
- Start long jobs detached (`setsid nohup ...`) through `colab exec`; a CLI timeout kills only the CLI.
- Always run `configs/cp2_plan_smoke.yaml` first (it caught a bundle bug that dropped `configs/data/`).
- `--batch-scale 2` is safe on 80 GB; OOM backoff halves batches automatically.
- Hidden-state features are large (about 1.5 GB for 6 models). Copy them off the runtime before it stops.
- **Laptop transfer:** the commit tool caps 50 files / 30 MB per file; for big result sets, tar,
  split into 25 MB parts, extract with device_bash, then delete the parts (needs delete permission).
- The laptop VM has 3 GB RAM and no persistent background processes; do compute in the cloud workspace or on Colab.
- `Makefile` cannot be written by the commit tool; write it with device_bash (base64).
- The user pushes git; Claude has no GitHub credentials. Large outputs are git-ignored by design.

## 10. Conventions for write-ups

No em dashes (use commas, colons, parentheses). "Prof." before professors' names. Brevity,
examples over long text, plain language in class documents. Every number must trace to a results
file; never cite a source that has not been checked. Label our own ideas as ours.

## 11. Key dates

| date | milestone |
|---|---|
| Nov 5 | CP2 progress update submitted (`reports/cp2/CP2_progress_update.pdf`) |
| Dec 1 or 3 | final presentation |
| Dec 4 | final paper and code |
