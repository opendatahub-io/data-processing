# Code Navigability Guide for KFP Pipelines

> Reference: [AI Bug Automation Readiness - Code Navigability](https://github.com/ugiordan/ai-bug-automation-readiness/blob/main/docs/TEAM_ACTION_GUIDE.md#6-code-navigability-5)

## KFP Component Navigation

### Understanding the Import Pattern

KFP `@dsl.component` functions are serialized into isolated containers. This creates a unique navigation pattern:

```python
# At module top (compile-time only — NOT available at runtime):
from common.constants import DOCLING_BASE_IMAGE

# Inside function body (runtime — must be self-contained):
@dsl.component(base_image=DOCLING_BASE_IMAGE)
def import_pdfs(...):
    import os                    # These run inside the container
    from pathlib import Path     # at pipeline execution time
    import boto3
```

**Key insight**: When navigating to a component's implementation, look inside the function body for the actual logic. Module-level imports are only used for compile-time configuration (base images, packages_to_install).

### Component Location Map

| Component | File | Pipeline |
|---|---|---|
| `import_pdfs` | `kubeflow-pipelines/common/components.py:8` | Both |
| `create_pdf_splits` | `kubeflow-pipelines/common/components.py:142` | Both |
| `download_docling_models` | `kubeflow-pipelines/common/components.py:166` | Both |
| `docling_chunk` | `kubeflow-pipelines/common/components.py:246` | Both |
| `docling_convert_standard` | `kubeflow-pipelines/docling-standard/standard_components.py:14` | Standard |
| `docling_convert_vlm` | `kubeflow-pipelines/docling-vlm/vlm_components.py:14` | VLM |

### Pipeline Definition Files

Each pipeline variant has a definition file that wires components together:

| File | What it does |
|---|---|
| `docling-standard/standard_convert_pipeline.py` | Defines standard pipeline DAG + compiles to YAML |
| `docling-vlm/vlm_convert_pipeline.py` | Defines VLM pipeline DAG + compiles to YAML |

Running these files as scripts (`python standard_convert_pipeline.py`) produces the `*_compiled.yaml`.

### Generated vs. Source Files

**Rule**: If a file starts with `# CODE GENERATED` or ends in `_compiled.yaml`, do not edit it. Find the source `.py` file and modify that instead.

## Refactoring Opportunities

### 1. Extract S3 secret reading into a helper

`common/components.py` lines 44-104 contain repetitive S3 secret file reading. This pattern repeats for each secret key:

```python
# Current: 6 nearly identical blocks
s3_endpoint_url_file_path = os.path.join(s3_secret_mount_path, "S3_ENDPOINT_URL")
if os.path.isfile(s3_endpoint_url_file_path):
    with open(s3_endpoint_url_file_path) as f:
        s3_endpoint_url = f.read()
else:
    raise ValueError(...)
```

**Note**: Because of KFP serialization, this helper must be defined inside the component function body.

### 2. Add common/README.md

The `kubeflow-pipelines/common/` directory would benefit from a README explaining the compile-time vs. runtime distinction and why `__init__.py` exports components.
