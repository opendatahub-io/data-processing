.PHONY: test unittest integration-test test-all test-notebook-parameters test-notebook-execution test-notebooks format-python format-notebooks lint-python format-python-check format-notebooks-check typecheck lint coverage help

USE_CASES := $(wildcard notebooks/use-cases/*.ipynb)
TUTORIALS := $(wildcard notebooks/tutorials/*.ipynb)
ALL_NOTEBOOKS := $(USE_CASES) $(TUTORIALS)

COMMON := $(wildcard kubeflow-pipelines/common/*.py)
DOCLING_STANDARD_PIPELINE := $(wildcard kubeflow-pipelines/docling-standard/*.py)
DOCLING_VLM_PIPELINE := $(wildcard kubeflow-pipelines/docling-vlm/*.py)
ALL_PYTHON_FILES := $(COMMON) $(DOCLING_STANDARD_PIPELINE) $(DOCLING_VLM_PIPELINE)

##@ Testing

test:                          ## Run all tests (unit + integration)
	pytest tests/ -v

unittest:                      ## Run unit tests only (no GPU, network, or Docker needed)
	pytest tests/unit/ -v

integration-test:              ## Run integration tests (notebook execution, may need GPU)
	pytest tests/integration/ -v

test-notebook-parameters:      ## Validate notebooks have papermill 'parameters' cell
	pytest tests/test_notebook_parameters.py -v

test-notebook-execution:       ## Execute all notebooks via papermill (slow, may need GPU)
	pytest tests/test_notebook_execution.py -v

test-notebooks: format-notebooks-check test-notebook-parameters test-notebook-execution  ## Run all notebook validations

coverage:                      ## Run unit tests with coverage report
	pytest tests/unit/ --cov --cov-report=term-missing --cov-report=html

test-all: lint test            ## Run everything: lint + all tests

##@ Formatting

format-python:                 ## Auto-format Python files with ruff
	ruff format $(ALL_PYTHON_FILES)

format-notebooks:              ## Strip notebook outputs with nbstripout
	nbstripout --keep-id $(ALL_NOTEBOOKS)

##@ Checks

lint-python:                   ## Run ruff linter on KFP Python files
	ruff check kubeflow-pipelines/

format-python-check:           ## Check Python formatting (ruff, no changes)
	ruff format --check $(ALL_PYTHON_FILES)

format-notebooks-check:        ## Check notebook outputs are stripped
	nbstripout --keep-id --verify $(ALL_NOTEBOOKS)

typecheck:                     ## Run mypy type checks on KFP code
	mypy kubeflow-pipelines/

lint: lint-python format-python-check  ## Run all lint + format checks

##@ Help

help:                          ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
