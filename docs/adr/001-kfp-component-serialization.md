# ADR-001: KFP Component Serialization

## Status

Accepted

## Context

Kubeflow Pipelines (KFP) SDK v2 serializes `@dsl.component`-decorated functions into self-contained container steps. The KFP compiler inspects the function source code, extracts it, and packages it to run inside a container image at runtime.

This means the function must be entirely self-contained — any imports, helper functions, or constants it uses must either:
1. Be defined inside the function body
2. Come from packages listed in `packages_to_install`
3. Already exist in the `base_image`

Project-level modules (like a shared `logging_config.py` or utility library) are **not available at runtime** because the container only has the serialized function code, not the full repository.

## Decision

All KFP component functions place their imports inside the function body, not at module top level.

```python
@dsl.component(base_image=DOCLING_BASE_IMAGE)
def docling_convert_standard(...):
    from pathlib import Path                    # Inside function
    from docling.document_converter import ...  # Inside function
    ...
```

The `# pylint: disable=import-outside-toplevel` comments throughout `components.py`, `standard_components.py`, and `vlm_components.py` exist because of this constraint, not as a style choice.

The `common/` module (`components.py`, `constants.py`) is used at **compile time only** — when `python standard_convert_pipeline.py` runs, it imports the component functions to build the pipeline graph and compile to YAML. At runtime in Kubernetes, each component runs in isolation.

## Consequences

- **Positive**: Components are fully portable — the compiled YAML works on any KFP instance without needing the source repository
- **Positive**: Dependencies are explicit per component via `base_image` and `packages_to_install`
- **Negative**: Code cannot be shared between components at runtime (some duplication is unavoidable)
- **Negative**: Import statements look unusual (inside functions) and trigger linter warnings
- **Constraint**: Any shared logging, error handling, or utility code must be duplicated in each component function or provided by the base image
