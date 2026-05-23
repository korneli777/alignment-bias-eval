.PHONY: help install install-gpu install-dev figures extended-views test aggregate clean

help:
	@echo "Targets:"
	@echo "  install         pip install -e . (CPU deps)"
	@echo "  install-gpu     pip install -e .[gpu] (adds torch + transformers)"
	@echo "  install-dev     pip install -e .[dev,tracking]"
	@echo "  figures         Rebuild paper figures from cached aggregates"
	@echo "  extended-views  Rebuild full-coverage figures not in the paper"
	@echo "  test            Unit + integration tests (incl. paper-number gate)"
	@echo "  aggregate       Re-aggregate parquets from raw JSONs (requires raw data)"
	@echo "  clean           Remove generated figures + caches"

install:
	pip install -e .

install-gpu:
	pip install -e .[gpu]

install-dev:
	pip install -e .[dev,tracking]

figures:
	python scripts/figures/chat_template_dumbbell.py
	python scripts/figures/bbq_quadrant.py
	python scripts/figures/probing_curves.py

extended-views:
	python scripts/figures/extended_views/probing_8panels.py

test:
	pytest tests/

aggregate:
	python scripts/aggregate.py

clean:
	rm -rf figures/*.png figures/*.pdf
	rm -rf figures/extended_views/*.png figures/extended_views/*.pdf
	rm -rf .pytest_cache __pycache__ src/**/__pycache__
