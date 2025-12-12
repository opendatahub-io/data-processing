.PHONY: format-python format-notebook format-python-check format-notebooks-check test-notebook-parameters

USE_CASES := $(wildcard notebooks/use-cases/*.ipynb)
TUTORIALS := $(wildcard notebooks/tutorials/*.ipynb)
ALL_NOTEBOOKS := $(USE_CASES) $(TUTORIALS)

COMMON := $(wildcard kubeflow-pipelines/common/*.py)
DOCLING_STANDARD_PIPELINE := $(wildcard kubeflow-pipelines/docling-standard/*.py)
DOCLING_VLM_PIPELINE := $(wildcard kubeflow-pipelines/docling-vlm/*.py)
ALL_PYTHON_FILES := $(COMMON) $(DOCLING_STANDARD_PIPELINE) $(DOCLING_VLM_PIPELINE)

format-python:
	@echo "Running ruff format..."
	ruff format $(ALL_PYTHON_FILES)

format-notebooks:
	@echo "Running nbstripout..."
	nbstripout --keep-id $(ALL_NOTEBOOKS)

format-notebooks-check:
	nbstripout --keep-id --verify $(ALL_NOTEBOOKS)
	@echo "nbstripout check passed :)"

format-python-check:
	ruff format --check $(ALL_PYTHON_FILES)
	@echo "ruff format check passed :)"

test-notebook-parameters:
	@echo "Running notebook parameters validation..."
	pytest tests/test_notebook_parameters.py -v
	@echo "Notebook parameters test passed :)"
