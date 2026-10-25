# Common commands. Run `make help` for a list.
PY ?= python

.PHONY: help install lint format test test-fast demo quick full clean

help:
	@echo "install     editable install with dev tools"
	@echo "lint        ruff lint + format check"
	@echo "format      ruff format and autofix"
	@echo "test        full pytest suite (about 15 s)"
	@echo "test-fast   pytest without tests marked slow (CI)"
	@echo "demo        tiny end-to-end pipeline with figures (about 1 min on CPU)"
	@echo "quick       experiments A-D at quick scale"
	@echo "full        experiments A-D at full scale (hours on CPU)"
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

full:
	for e in A_modality B_objective C_force_timing D_reuse; do \
		$(PY) scripts/run_experiment.py configs/experiments/$$e.yaml --scale full || exit 1; done

clean:
	rm -rf data outputs .pytest_cache .ruff_cache
