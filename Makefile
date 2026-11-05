# Pipeline shortcuts. Override the experiment with: make cp1 EXP=configs/experiment/other.yaml
# Extra config overrides:                      make infer ARGS="generation.batch_size=2"
EXP  ?= configs/experiment/cp1_pilot.yaml
PY   ?= python
ARGS ?=

.PHONY: env test data infer eval analyze verify cp1 cp2-smoke cp2 cp2-status bundle

env:      ## create the conda environment
	conda env create -f environment.yml

test:     ## unit tests
	$(PY) -m pytest -q

data:     ## download, verify, sample pairs
	$(PY) scripts/prepare_data.py --config $(EXP) $(ARGS)

infer:    ## generate SQL + token confidences (resumable)
	$(PY) scripts/run_inference.py --config $(EXP) $(ARGS)

eval:     ## execution-match labels
	$(PY) scripts/evaluate.py --config $(EXP) $(ARGS)

analyze:  ## metrics, figures, SUMMARY.md
	$(PY) scripts/analyze.py --config $(EXP) $(ARGS)

verify:   ## independent re-check of headline numbers
	$(PY) scripts/verify_results.py --config $(EXP) $(ARGS)

cp1: test data infer eval analyze verify  ## full checkpoint-1 pipeline

cp2-smoke: ## every CP2 stage on 0.5B with ~50 questions (minutes on a GPU)
	$(PY) scripts/run_cp2.py --plan configs/cp2_plan_smoke.yaml $(ARGS)

cp2:       ## full Checkpoint 2 plan (configs/cp2_plan.yaml), resumable
	$(PY) scripts/run_cp2.py $(ARGS)

cp2-status: ## progress table for the CP2 plan
	$(PY) scripts/run_cp2.py --status $(ARGS)

bundle:    ## zip the code for Colab (colab/nl2sql_confidence_code.zip)
	$(PY) colab/make_bundle.py
