## Description

<!-- What does this PR do? Link the related issue(s). -->

Fixes #

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactoring (no behavior change)
- [ ] Documentation
- [ ] CI/CD or tooling
- [ ] Pipeline component change
- [ ] Notebook change

## Testing Checklist

<!-- Check all that apply. CI will also run these automatically. -->

- [ ] `make unittest` passes (unit tests, no GPU needed)
- [ ] `make lint` passes (ruff lint + format check)
- [ ] `make typecheck` passes (mypy)
- [ ] New/modified code has tests in `tests/unit/` or `tests/integration/`

### If you changed KFP pipelines or components:

- [ ] Recompiled YAML and committed it (`python *_convert_pipeline.py`)
- [ ] Tested locally with `local_run.py` (or confirmed CI will cover it)

### If you changed notebooks:

- [ ] Notebook has a code cell tagged `parameters` for papermill
- [ ] Cell outputs are stripped (pre-commit hook handles this)
- [ ] Tested execution locally or confirmed CI will cover it

### If you changed dependencies:

- [ ] Updated the correct scoped `requirements.txt` (not a monolithic file)
- [ ] Pinning strategy matches existing pattern (exact for KFP, floor for scripts)

## Test Evidence

<!-- Paste relevant output from test runs, screenshots, or logs. -->
<!-- For AI-generated PRs: include the full test command and output. -->

```
<paste test output here>
```

## Additional Context

<!-- Anything else reviewers should know? Breaking changes, migration steps, etc. -->
