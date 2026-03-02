PYTHON := $(shell if [ -x .venv/bin/python ]; then printf '%s' .venv/bin/python; else command -v python3; fi)

.PHONY: run test watch hooks

run:
	$(PYTHON) -m src.bot

test:
	$(PYTHON) -m unittest discover -s tests -t .

watch:
	$(PYTHON) scripts/watch_tests.py

hooks:
	git config core.hooksPath .githooks
