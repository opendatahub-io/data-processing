# One-Command Test Execution

> Principle: Tests must be runnable with a single command.

## Changes

### 1. Makefile — Reorganized with clear test hierarchy

| Target | Command | What it runs | Speed |
|---|---|---|---|
| `make test` | `pytest tests/ -v` | All tests (unit + integration) | Varies |
| `make unittest` | `pytest tests/ -m unit -v` | Unit tests only (no GPU/network/Docker) | Fast (~5s) |
| `make integration-test` | `pytest tests/ -m integration -v` | Integration tests (notebook execution) | Slow (minutes) |
| `make test-all` | `lint` + `typecheck` + `test` | Everything: formatting, types, all tests | Slowest |
| `make lint` | `format-python-check` + `format-notebooks-check` | All lint/format checks | Fast |
| `make help` | — | Show all targets with descriptions | — |

**Key improvement**: `make test` now runs _only_ tests (no formatting/lint mixed in). The old `test-all` mixed pytest with `format-python-check`, which meant a formatting issue would block test results. Now `test-all` calls `lint`, `typecheck`, and `test` as separate stages.

Granular targets (`test-notebook-parameters`, `test-notebook-execution`, `typecheck`) are preserved for CI workflows that need them.

### 2. pyproject.toml — Added pytest markers

```toml
[tool.pytest.ini_options]
markers = [
  "unit: Fast unit tests with no external dependencies (GPU, network, cluster)",
  "integration: Tests requiring external resources (notebook execution, GPU, Docker)",
]
```

### 3. conftest.py — Auto-applies markers by file name

Instead of requiring `@pytest.mark.unit` on every test class/function, `conftest.py` auto-applies markers based on the test file name:

```python
UNIT_TEST_FILES = {
    "test_subset_selection_config.py",
    "test_subset_selection_processor.py",
    "test_cli.py",
    "test_encoder_registry.py",
    "test_kfp_constants.py",
}

INTEGRATION_TEST_FILES = {
    "test_notebook_execution.py",
}
```

This uses `pytest_collection_modifyitems` to tag items at collection time. Adding a new test file just means adding its name to the appropriate set.

### 4. CLAUDE.md — Updated test documentation

Test Commands section now documents `make test`, `make unittest`, `make integration-test`, and explains the marker system.

## Files Modified

| File | Change |
|---|---|
| `Makefile` | Added `test`, `unittest`, `integration-test`, `lint`, `help` targets; reorganized with `##@` section headers |
| `pyproject.toml` | Added `markers` to `[tool.pytest.ini_options]` |
| `tests/conftest.py` | Added `pytest_collection_modifyitems` hook for auto-marking; added `UNIT_TEST_FILES` and `INTEGRATION_TEST_FILES` sets |
| `CLAUDE.md` | Updated Test Commands section with new targets and marker documentation |

## Quick Reference

```bash
# Developer daily workflow (fast feedback)
make unittest          # ~5 seconds, no GPU needed

# Pre-push check
make test-all          # lint + typecheck + all tests

# CI granular
make test-notebook-parameters
make test-notebook-execution
make typecheck
make lint
```
