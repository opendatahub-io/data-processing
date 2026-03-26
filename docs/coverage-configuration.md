# Coverage Configuration

> Principle: Code coverage reporting helps identify untested code paths.

## What Was Added

### 1. `pytest-cov` dependency

Added to `requirements-dev.txt`:
```
pytest-cov>=4.1.0
```

### 2. Coverage configuration in `pyproject.toml`

```toml
[tool.coverage.run]
source = [
  "scripts/subset_selection",
  "kubeflow-pipelines/common",
]
omit = [
  "tests/*",
  "*/__pycache__/*",
  "*/local_run.py",
]

[tool.coverage.report]
show_missing = true
skip_empty = true
fail_under = 40
exclude_lines = [
  "pragma: no cover",
  "if __name__ == .__main__.",
  "if TYPE_CHECKING:",
]

[tool.coverage.html]
directory = "htmlcov"
```

**Design decisions**:

- **`source`** covers the two subsystems with unit-testable business logic: `scripts/subset_selection/` (config, CLI, processor, encoders, utils) and `kubeflow-pipelines/common/` (constants). KFP component files and pipeline definitions are excluded because they require container runtime for meaningful testing.

- **`fail_under = 40`** — a pragmatic floor. The current unit tests cover config validation, CLI parsing, processor methods, and encoder registry. GPU-dependent code paths (embedding generation, subset selection) can't be covered without hardware. 40% is achievable now and can be raised as more tests are added.

- **`omit`** excludes test files themselves, bytecode, and `local_run.py` (Docker orchestration scripts that are integration-tested, not unit-tested).

- **`exclude_lines`** skips `if __name__ == "__main__"` blocks and `TYPE_CHECKING` imports — standard exclusions that don't represent meaningful untested logic.

### 3. Makefile target

```makefile
coverage:  ## Run unit tests with coverage report
	pytest tests/ -m unit --cov --cov-report=term-missing --cov-report=html
```

Generates both a terminal report (with missing line numbers) and an HTML report in `htmlcov/`.

### 4. CI workflow integration

Updated `.github/workflows/ci.yml`:

```yaml
- name: Run unit tests with coverage
  run: pytest tests/ -m unit --cov --cov-report=term-missing --cov-report=xml:coverage.xml

- name: Upload coverage to Codecov
  if: github.event_name == 'pull_request'
  uses: codecov/codecov-action@v5
  with:
    files: coverage.xml
    fail_ci_if_error: false
```

- Generates XML coverage report for Codecov upload
- `fail_ci_if_error: false` — Codecov upload failures don't block PRs (token may not be configured yet)
- Only uploads on `pull_request` events (not `workflow_dispatch`)

### 5. `.gitignore` update

Added coverage artifacts:
```
htmlcov/
coverage.xml
.coverage
```

## Files Modified

| File | Change |
|---|---|
| `requirements-dev.txt` | Added `pytest-cov>=4.1.0` |
| `pyproject.toml` | Added `[tool.coverage.run]`, `[tool.coverage.report]`, `[tool.coverage.html]` sections |
| `Makefile` | Added `coverage` target |
| `.github/workflows/ci.yml` | Changed `make unittest` to `pytest --cov`, added Codecov upload step |
| `.gitignore` | Added `htmlcov/`, `coverage.xml`, `.coverage` |
| `CLAUDE.md` | Added coverage section under Test Commands, added `make coverage` to granular targets |

## Usage

```bash
# Local: terminal report with missing lines
make coverage

# Local: just the summary
pytest -m unit --cov

# Local: specific module
pytest tests/test_cli.py --cov=scripts/subset_selection/cli

# CI: automatically runs on every PR via ci.yml
```

## Codecov Setup (One-Time)

To enable Codecov PR comments and badge:

1. Go to [codecov.io](https://codecov.io) and add the `opendatahub-io/data-processing` repository
2. Add the `CODECOV_TOKEN` secret to the GitHub repository settings
3. Update the CI workflow to pass the token:
   ```yaml
   - uses: codecov/codecov-action@v5
     with:
       files: coverage.xml
       token: ${{ secrets.CODECOV_TOKEN }}
   ```
4. Optionally add a `codecov.yml` at repo root for PR comment configuration
