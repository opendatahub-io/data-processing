# Linting in CI

> Principle: Automated code style enforcement catches formatting issues in AI-generated code.

## Gap

The existing `validate-python.yml` only ran `ruff format --check` — this checks **formatting** (whitespace, line length, quote style) but not **linting** (actual code issues: unused imports, undefined names, mutable default arguments, missing imports, deprecated syntax).

The Makefile `lint` target had the same gap, and only covered `kubeflow-pipelines/` files — missing `scripts/` and `tests/`.

| What | Formatting (`ruff format`) | Linting (`ruff check`) |
|---|---|---|
| Catches style/whitespace | Yes | No |
| Catches undefined names | No | Yes (F821) |
| Catches unused imports | No | Yes (F401) |
| Catches mutable defaults | No | Yes (B006) |
| Catches import ordering | No | Yes (I) |
| Catches deprecated syntax | No | Yes (UP) |

## Changes

### 1. `validate-python.yml` — Added `ruff check` step

```yaml
- name: Run ruff linter
  run: ruff check scripts/ tests/ kubeflow-pipelines/

- name: Run ruff format check
  run: ruff format --check scripts/ tests/ kubeflow-pipelines/
```

Also:
- Renamed workflow from "Python Formatting Validation" to "Python Lint & Format"
- Removed `pull_request_target` trigger (security concern flagged in earlier audit)
- Expanded scope from `$(ALL_PYTHON_FILES)` (kubeflow-pipelines only) to all Python directories

### 2. Makefile — Added `lint-python` target

```makefile
lint-python:                   ## Run ruff linter on all Python files
	ruff check scripts/ tests/ kubeflow-pipelines/
```

Updated `lint` target to include linting:
```makefile
lint: lint-python format-python-check format-notebooks-check
```

Updated `format-python-check` to cover all directories:
```makefile
format-python-check:
	ruff format --check scripts/ tests/ kubeflow-pipelines/
```

### 3. Ruff rules (already configured)

The `pyproject.toml` already had comprehensive lint rules — they just weren't being run in CI:

```toml
lint.select = ["E","F","I","B","W","UP"]
```

| Rule | What it catches |
|---|---|
| E | pycodestyle errors |
| F | pyflakes (undefined names, unused imports, redefined variables) |
| I | import sorting (isort-compatible) |
| B | flake8-bugbear (common bug patterns) |
| W | warnings |
| UP | pyupgrade (modernize syntax for Python 3.12) |

## CI Pipeline After Change

```
PR opened
  → validate-python.yml
      1. ruff check (LINT)        ← NEW
      2. ruff format --check (FORMAT)
  → ci.yml
      3. pytest tests/unit/ (UNIT TESTS)
      4. pytest test_notebook_parameters.py
  → typecheck.yml
      5. mypy (TYPE CHECK)
```

## Files Modified

| File | Change |
|---|---|
| `.github/workflows/validate-python.yml` | Added `ruff check` step, removed `pull_request_target`, expanded file scope |
| `Makefile` | Added `lint-python` target, updated `lint` to include it, expanded `format-python-check` scope |
| `CLAUDE.md` | Updated CI table and lint command descriptions |
