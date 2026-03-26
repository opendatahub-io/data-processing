# Test Isolation

> Principle: Separate unit tests (fast, no dependencies) from integration/e2e tests.

## Before

All test files flat in `tests/`:
```
tests/
  conftest.py                        # filename-based marker logic
  test_cli.py
  test_encoder_registry.py
  test_kfp_constants.py
  test_subset_selection_config.py
  test_subset_selection_processor.py
  test_notebook_parameters.py        # (in repo)
  test_notebook_execution.py         # (in repo)
```

Isolation was marker-based only (`pytest -m unit`), enforced by maintaining a `UNIT_TEST_FILES` set in `conftest.py`. Fragile — adding a test file without updating the set meant it ran in neither category.

## After

Directory-based isolation with auto-applied markers:
```
tests/
  conftest.py                        # shared fixtures (notebook discovery)
  fixtures/                          # sample data for tests
  unit/                              # fast tests, no external deps
    __init__.py
    conftest.py                      # auto-applies @pytest.mark.unit
    test_cli.py
    test_encoder_registry.py
    test_kfp_constants.py
    test_subset_selection_config.py
    test_subset_selection_processor.py
  integration/                       # slow tests, need GPU/network/Docker
    __init__.py
    conftest.py                      # auto-applies @pytest.mark.integration
    test_notebook_parameters.py      # (existing, in repo)
    test_notebook_execution.py       # (existing, in repo)
```

## How It Works

Each subdirectory has a minimal `conftest.py` that auto-applies the marker:

```python
# tests/unit/conftest.py
def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.unit)
```

**Adding a new test**: just place it in `tests/unit/` or `tests/integration/`. No need to annotate with `@pytest.mark.unit` or update any registry — the directory is the isolation boundary.

## Three Ways to Run

| Method | Unit tests | Integration tests | All tests |
|---|---|---|---|
| **Directory** (primary) | `pytest tests/unit/` | `pytest tests/integration/` | `pytest tests/` |
| **Marker** (alternative) | `pytest -m unit` | `pytest -m integration` | `pytest` |
| **Makefile** (recommended) | `make unittest` | `make integration-test` | `make test` |

## Changes Made

| File | Change |
|---|---|
| `tests/unit/` | **New directory** — moved 5 unit test files here |
| `tests/unit/__init__.py` | **New** — makes it a package for pytest discovery |
| `tests/unit/conftest.py` | **New** — auto-applies `@pytest.mark.unit` |
| `tests/integration/` | **New directory** — integration test files go here |
| `tests/integration/__init__.py` | **New** — makes it a package |
| `tests/integration/conftest.py` | **New** — auto-applies `@pytest.mark.integration` |
| `tests/conftest.py` | **Updated** — removed `UNIT_TEST_FILES`/`INTEGRATION_TEST_FILES` sets and filename-based marker logic; kept shared fixtures |
| `pyproject.toml` | **Updated** — `testpaths = ["tests/unit", "tests/integration"]` |
| `Makefile` | **Updated** — targets use directory paths (`tests/unit/`, `tests/integration/`) instead of `-m unit` |
| `.github/workflows/ci.yml` | **Updated** — runs `pytest tests/unit/` instead of `pytest -m unit` |
| `CLAUDE.md` | **Updated** — Key Directories, Test Commands, Test Isolation sections |

## Migration Guide for Existing Tests

The existing repo test files (`test_notebook_parameters.py`, `test_notebook_execution.py`) should be moved from `tests/` to `tests/integration/`:

```bash
mv tests/test_notebook_parameters.py tests/integration/
mv tests/test_notebook_execution.py tests/integration/
```

The `from conftest import get_notebook_files` import in these files will continue to work — pytest makes the root `conftest.py` available to all subdirectories automatically.
