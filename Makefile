.PHONY: help install install-gpu install-dev lint figures extended-views test verify aggregate clean

help:
	@echo "Targets:"
	@echo "  install         Install the CPU analysis package"
	@echo "  install-gpu     Add model scoring and probing dependencies"
	@echo "  install-dev     Add tests and linting"
	@echo "  lint            Check Python style and common errors"
	@echo "  figures         Rebuild paper figures from cached aggregates"
	@echo "  extended-views  Rebuild full-coverage figures not in the paper"
	@echo "  test            Run the software test suite"
	@echo "  verify          Run lint, tests, and rebuild paper figures"
	@echo "  aggregate       Re-aggregate parquets from raw JSONs (requires raw data)"
	@echo "  clean           Remove generated figures + caches"

install:
	python -m pip install -e .

install-gpu:
	python -m pip install -e ".[gpu]"

install-dev:
	python -m pip install -e ".[dev]"

lint:
	ruff check src scripts tests

figures:
	python scripts/figures/chat_template_dumbbell.py
	python scripts/figures/bbq_quadrant.py
	python scripts/figures/probing_curves.py

extended-views:
	python scripts/figures/extended_views/probing_8panels.py

test:
	python -m pytest tests/

verify: lint test figures

aggregate:
	python scripts/aggregate.py

clean:
	rm -rf figures/*.png figures/*.pdf
	rm -rf figures/extended_views/*.png figures/extended_views/*.pdf
	rm -rf .pytest_cache __pycache__ src/**/__pycache__
