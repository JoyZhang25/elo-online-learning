.PHONY: install test simulations real-data all

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m unittest discover -s tests -v

simulations:
	$(PYTHON) scripts/run_simulations.py

real-data:
	$(PYTHON) scripts/run_real_data.py

all:
	$(PYTHON) scripts/build_all.py
