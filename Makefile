PYTHON ?= python3
TECTONIC ?= tectonic

.DEFAULT_GOAL := pdf
.PHONY: pdf check archives

pdf:
	$(TECTONIC) --keep-intermediates --keep-logs main.tex

check:
	$(PYTHON) anc/run_all.py

archives: pdf check
	$(PYTHON) package_submission.py
