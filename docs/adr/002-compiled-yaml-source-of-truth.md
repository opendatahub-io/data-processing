# ADR-002: Compiled YAML as Source of Truth

## Status

Accepted

## Context

KFP pipelines are defined in Python (`*_convert_pipeline.py`) but deployed as compiled YAML files (`*_compiled.yaml`). Users install pipelines by uploading the compiled YAML to a KFP instance — they never run the Python source directly in production.

Two approaches exist for managing compiled YAML:
1. **Commit the compiled YAML** alongside source code
2. **Generate YAML in CI only** and publish as a release artifact

## Decision

Compiled YAML files are committed to the repository and treated as checked-in artifacts.

CI enforces consistency: the `compile-kfp.yml` workflow recompiles from source and diffs against the committed YAML. If they differ, CI fails with instructions to regenerate locally.

```
standard_convert_pipeline_compiled.yaml   ← committed, GENERATED
vlm_convert_pipeline_compiled.yaml        ← committed, GENERATED
```

## Consequences

- **Positive**: Users can install pipelines directly from the repository without building from source — just download the YAML via URL
- **Positive**: PR reviewers can see YAML changes alongside Python changes
- **Positive**: The `main` branch always has a deployable pipeline
- **Negative**: Contributors must remember to regenerate YAML after pipeline code changes (`python *_convert_pipeline.py`)
- **Negative**: PRs that modify pipeline code will have larger diffs (Python changes + YAML changes)
- **Mitigation**: CI catches stale YAML and provides the exact regeneration command in the failure message
