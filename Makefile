.PHONY: install test simulations tennis all

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m unittest discover -s tests -v

simulations:
	$(PYTHON) scripts/run_simulations.py

tennis:
	$(PYTHON) scripts/run_real_data.py

all:
	$(PYTHON) scripts/build_all.py
