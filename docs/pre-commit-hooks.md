# Pre-commit / Local Hooks

> Principle: Fast local validation before pushing code.

## Before

```yaml
# .pre-commit-config.yaml (original)
repos:
  - repo: ruff-pre-commit
    hooks:
      - id: ruff-format        # formatting only, no linting
  - repo: nbstripout
    hooks:
      - id: nbstripout
```

Only 2 hooks — formatting and notebook output stripping. No linting, no YAML validation, no type checking, no hygiene checks.

## After

```yaml
# .pre-commit-config.yaml (enhanced)
repos:
  - repo: ruff-pre-commit
    hooks:
      - id: ruff               # lint with autofixes (NEW)
      - id: ruff-format        # format
  - repo: nbstripout
    hooks:
      - id: nbstripout
  - repo: pre-commit-hooks     # (NEW)
    hooks:
      - id: check-yaml         # catch broken YAML
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-merge-conflict
      - id: check-added-large-files  # block >500KB files
  - repo: mirrors-mypy         # (NEW, pre-push only)
    hooks:
      - id: mypy               # type checking
```

**8 hooks** total across two stages: 7 on commit (fast), 1 on push (slower).

## Hook Stages

| Stage | Hooks | Speed | What it catches |
|---|---|---|---|
| **pre-commit** | ruff-lint, ruff-format, nbstripout, check-yaml, end-of-file-fixer, trailing-whitespace, check-merge-conflict, check-added-large-files | <5s | Style issues, lint errors, broken YAML, merge conflicts, large binaries |
| **pre-push** | mypy | ~15s | Type errors across the codebase |

The two-stage design keeps commits fast while ensuring type safety before pushing.

## New Hooks Explained

### `ruff` (lint)
The original config only had `ruff-format`. Adding `ruff` (the linter) catches actual bugs before commit:
- **E/F**: pycodestyle + pyflakes (undefined names, unused imports)
- **I**: import sorting (isort-compatible)
- **B**: flake8-bugbear (common bug patterns like mutable defaults)
- **W**: warnings
- **UP**: pyupgrade (modernize syntax for Python 3.12)

The `--fix` flag auto-corrects safe issues; `--exit-non-zero-on-fix` ensures the commit is blocked so the developer reviews the fix.

### `check-yaml`
Validates YAML syntax. Critical for this project — broken `.github/workflows/*.yml` or `*_compiled.yaml` files are caught before commit, not in CI. Uses `--unsafe` to allow GitHub Actions custom tags (`${{ }}`).

### `check-added-large-files`
Blocks files >500KB. Prevents accidentally committing model weights, large PDFs, or notebook outputs that bypass nbstripout.

### `check-merge-conflict`
Catches committed merge conflict markers (`<<<<<<<`). Common when resolving conflicts in compiled YAML files.

### `mypy` (pre-push)
Runs on push, not commit, because it takes ~15s and requires resolving all imports. Uses the same config as `make typecheck`. Catches type errors before they hit CI.

## Exclude Pattern

```yaml
exclude: |
  (?x)(
    ^\.git/|
    ^\.venv/|^venv/|
    ^\.pytest_cache/|^\.mypy_cache/|
    ^\.ipynb_checkpoints/|
    _compiled\.yaml$
  )
```

Added `_compiled\.yaml$` to skip generated pipeline YAML files — these are auto-generated and should not be linted or modified by hooks.

## Installation

```bash
# Install both commit and push hooks
pre-commit install && pre-commit install --hook-type pre-push

# Run all hooks against all files (first-time or CI)
pre-commit run --all-files

# Run only specific hooks
pre-commit run ruff --all-files
pre-commit run check-yaml --all-files
```

## Files Modified

| File | Change |
|---|---|
| `.pre-commit-config.yaml` | Added ruff lint, check-yaml, end-of-file-fixer, trailing-whitespace, check-merge-conflict, check-added-large-files, mypy (pre-push); added `_compiled.yaml` to exclude |
| `CLAUDE.md` | Updated Formatting/Linting section with full hook list and two-stage explanation; updated install command in Contribution Workflow |
