# Tests

This directory contains automated tests for the ODH Data Processing project.

## Overview

Tests validate project components including notebooks, Python modules, and configurations to ensure they work correctly in development and CI/CD environments.

## Current Tests

- **`test_notebook_parameters.py`** - Validates notebooks have required parameters cells for papermill execution
- **`conftest.py`** - Shared test configuration and utilities

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_*.py -v

# Run via Makefile (where available)
make test-notebook-parameters
```

Tests also run automatically in CI/CD via GitHub Actions workflows.

## Setup

Install dependencies:
```bash
pip install -r requirements-dev.txt
```

## Adding New Tests

1. Create new test files following `test_*.py` naming convention
2. Add shared utilities to `conftest.py` if needed  
3. Update this README to document new test categories
4. Add Makefile targets for convenient test execution

## Configuration

Test configuration is in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = ["--tb=short", "-v"]
```

## Troubleshooting

Common issues:
- **Test discovery**: Run from project root where `pyproject.toml` exists
- **Import errors**: Install dependencies with `pip install -r requirements-dev.txt`
- **Test failures**: Check error messages for specific validation requirements