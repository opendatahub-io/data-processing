# CI Runs Tests on PRs

> Requirement: At least one GitHub Actions workflow must trigger on `pull_request` (not `pull_request_target`) AND contain a step that runs tests — both in the same file scores 100.

## Audit of Existing Workflows

| Workflow | `pull_request` trigger? | Runs tests (`pytest`/`make test`)? | Scores 100? |
|---|---|---|---|
| `validate-python.yml` | Yes (also has `pull_request_target`) | No — runs `make format-python-check` | No |
| `validate-notebooks.yml` | Yes | Partial — runs `make test-notebook-parameters` | Partial |
| `execute-all-notebooks.yml` | No — `push` + `workflow_dispatch` only | Yes — runs pytest | No |
| `compile-kfp.yml` | Yes | No — compiles YAML and diffs | No |
| `execute-kfp-localrunners.yml` | Yes | Yes — runs `local_run.py` | No — not pytest |
| `typecheck.yml` | Yes | No — runs mypy | No |

**Gap**: No workflow triggers on `pull_request` and runs `make test` or `pytest tests/`. The existing workflows are split by concern (format, notebooks, pipelines, types) with none running the unit test suite.

### Additional issue: `validate-python.yml` uses `pull_request_target`

This workflow triggers on both `pull_request` and `pull_request_target`. The `pull_request_target` trigger runs in the context of the **base** branch, not the PR branch, which is a security concern for workflows that check out PR code (it does: `ref: ${{ github.event.pull_request.head.sha }}`). This is safe here since it only has `contents: read` and runs `ruff`, but it's worth noting — the rubric explicitly says to use `pull_request`, not `pull_request_target`.

## Solution: New `ci.yml` Workflow

Created `.github/workflows/ci.yml` — a single workflow that satisfies both criteria:

```yaml
name: CI

on:
  workflow_dispatch:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v6

      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
          pip install -r scripts/subset_selection/requirements.txt

      - name: Run unit tests
        run: make unittest

      - name: Run notebook parameter validation
        run: make test-notebook-parameters
```

### Design decisions

1. **Triggers on every PR** — no `paths` filter. Unit tests are fast (~5s) and should catch regressions regardless of what files changed.

2. **Runs `make unittest`** — uses the `unittest` Makefile target which runs `pytest -m unit`. This executes all unit tests (config, CLI, processor, encoder registry, constants) without requiring GPU, Docker, or network access.

3. **Also runs `test-notebook-parameters`** — this fast validation (checks for papermill `parameters` cells) catches a common contributor mistake and doesn't need notebooks to execute.

4. **Does NOT run integration tests** — notebook execution and pipeline local runners need EC2 GPU instances. Those are already handled by `execute-all-notebooks.yml` (on push to main) and `execute-kfp-localrunners.yml` (on pipeline PRs).

5. **Uses `pip cache`** — speeds up repeated runs on the same PR.

6. **Installs subset_selection requirements** — needed for the unit tests that import from `scripts/subset_selection/`.

## Workflow Coverage After Change

| What gets tested | On PR? | On push to main? |
|---|---|---|
| Unit tests (config, CLI, processor, registry, constants) | `ci.yml` | `ci.yml` (via merge) |
| Python formatting (ruff) | `validate-python.yml` | — |
| Notebook formatting (nbstripout) | `validate-notebooks.yml` | `validate-notebooks.yml` |
| Notebook parameters | `ci.yml` + `validate-notebooks.yml` | `validate-notebooks.yml` |
| Notebook execution | — | `execute-all-notebooks.yml` |
| Pipeline compilation | `compile-kfp.yml` | `compile-kfp.yml` |
| Pipeline local runners | `execute-kfp-localrunners.yml` | `execute-kfp-localrunners.yml` |
| Type checking (mypy) | `typecheck.yml` | — |

## Files Created/Modified

| File | Change |
|---|---|
| `.github/workflows/ci.yml` | **New** — PR-triggered workflow running `make unittest` |
| `CLAUDE.md` | Added `ci.yml` to CI/CD table |
