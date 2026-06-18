# Dependency Audit — KFP Pipelines

> Reference: [AI Bug Automation Readiness - Dependency Complexity](https://github.com/ugiordan/ai-bug-automation-readiness/blob/main/docs/TEAM_ACTION_GUIDE.md)

## Dependency Strategy

The project uses **scoped requirements files** — each pipeline variant has its own `requirements.txt` with exact pinning for reproducible builds.

## Requirements File Inventory

| File | Scope | Pinning Strategy |
|---|---|---|
| `requirements-dev.txt` | Dev tools (ruff, pytest, nbstripout, mypy) | Floor (`>=`) — latest compatible |
| `kubeflow-pipelines/docling-standard/requirements.txt` | Standard pipeline compilation | Exact (`==`) — reproducible KFP builds |
| `kubeflow-pipelines/docling-vlm/requirements.txt` | VLM pipeline compilation | Exact (`==`) — reproducible KFP builds |

## KFP Pipeline Dependencies (exact-pinned)

| Package | Version | Used By | Purpose |
|---|---|---|---|
| `docling` | 2.57.0 | Both | Document conversion engine |
| `kfp` | 2.14.6 | Both | Kubeflow Pipelines SDK |
| `kfp-kubernetes` | 2.14.6 | Both | KFP Kubernetes extensions |
| `boto3` | 1.40.52 | Both | S3 storage access |
| `tesserocr` | 2.9.1 | Standard only | OCR engine binding |

## Container Base Images

These are not in requirements files — they're defined in `kubeflow-pipelines/common/constants.py` and configurable via environment variables:

| Image | Default | Components Using It |
|---|---|---|
| `PYTHON_BASE_IMAGE` | `registry.access.redhat.com/ubi9/python-311:9.6-*` | `import_pdfs`, `create_pdf_splits` |
| `DOCLING_BASE_IMAGE` | `quay.io/fabianofranz/docling-ubi9:2.54.0` | `download_docling_models`, `docling_convert_*`, `docling_chunk` |

## KFP Component Runtime Dependencies

KFP components specify additional runtime dependencies via `packages_to_install` in `@dsl.component()`. These are installed inside the container at pipeline execution time:

| Component | `packages_to_install` | Container |
|---|---|---|
| `import_pdfs` | `["boto3", "requests"]` | PYTHON_BASE_IMAGE |
| `create_pdf_splits` | (none) | PYTHON_BASE_IMAGE |
| `download_docling_models` | (none) | DOCLING_BASE_IMAGE |
| `docling_convert_*` | (none) | DOCLING_BASE_IMAGE |
| `docling_chunk` | (none) | DOCLING_BASE_IMAGE |

## Recommendations

1. **Keep exact pinning for KFP dependencies** — pipeline reproducibility is critical
2. **Keep floor pinning for dev tools** — always use latest compatible
3. **Dependabot** is already configured (`.github/dependabot.yml`) to monitor pip dependencies weekly
4. **No monolithic requirements.txt** — each subsystem manages its own dependencies
