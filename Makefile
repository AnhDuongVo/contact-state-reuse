# Common commands. Run `make help` for a list.
PY ?= python

.PHONY: help install lint format test test-fast demo quick medium full revision paper clean

help:
	@echo "install     editable install with dev tools"
	@echo "lint        ruff lint + format check"
	@echo "format      ruff format and autofix"
	@echo "test        full pytest suite (about 25 s)"
	@echo "test-fast   pytest without tests marked slow (CI)"
	@echo "demo        tiny end-to-end pipeline with figures (about 1 min on CPU)"
	@echo "quick       experiments A-D at quick scale"
	@echo "medium      experiments A-D at the white-paper scale (about 2 h on 2 cores)"
	@echo "full        experiments A-D at full scale (many hours on CPU)"
	@echo "revision    all v0.2 experiments (E, T, L, G1, G2, R), summaries (about 6 h on 2 cores)"
	@echo "paper       v0.2 figures and PDFs from results/revision"
	@echo "clean       remove generated data and outputs"

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow"

demo:
	$(PY) scripts/demo.py --config configs/demo.yaml

quick:
	for e in A_modality B_objective C_force_timing D_reuse; do \
		$(PY) scripts/run_experiment.py configs/experiments/$$e.yaml --scale quick || exit 1; done

medium:
	for e in A_modality B_objective C_force_timing D_reuse; do \
		$(PY) scripts/run_experiment.py configs/experiments/$$e.yaml --scale medium || exit 1; done

full:
	for e in A_modality B_objective C_force_timing D_reuse; do \
		$(PY) scripts/run_experiment.py configs/experiments/$$e.yaml --scale full || exit 1; done

revision:
	bash scripts/run_revision.sh
	$(PY) scripts/analyze_force_input.py --run outputs/E_factorial_medium/run1 --out results/revision
	$(PY) scripts/summarize_revision.py --out results/revision

paper:
	$(PY) scripts/make_revision_figures.py --results results/revision --out docs/whitepaper/figures
	$(PY) docs/whitepaper/build_pdf.py

clean:
	rm -rf data outputs .pytest_cache .ruff_cache
