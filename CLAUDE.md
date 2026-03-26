# Data Processing — Kubeflow Pipelines

Document conversion and chunking pipelines using Docling, packaged as Kubeflow Pipelines (KFP) for Open Data Hub / Red Hat OpenShift AI.

Repository: [opendatahub-io/data-processing](https://github.com/opendatahub-io/data-processing)

## Architecture

```
PDF/Documents → Import (S3/HTTP) → Split → Docling Convert → Chunk (optional) → JSONL output
```

**Core flow**: KFP pipelines import PDFs, split them for parallel processing, convert via Docling (standard OCR or VLM), and optionally chunk output using HybridChunker for RAG.

Two pipeline variants:
- **Standard** (`docling-standard/`): Traditional Docling models (layout + tableformer)
- **VLM** (`docling-vlm/`): Vision-language models (SmolVLM/SmolDocling), supports local or remote inference

Shared KFP components (`common/components.py`): `import_pdfs`, `create_pdf_splits`, `download_docling_models`, `docling_chunk`

## Key Directories

| Directory | Purpose |
|---|---|
| `kubeflow-pipelines/common/` | Shared KFP components and constants (`components.py`, `constants.py`) |
| `kubeflow-pipelines/docling-standard/` | Standard Docling conversion pipeline |
| `kubeflow-pipelines/docling-vlm/` | VLM-based conversion pipeline |
| `tests/unit/` | Fast unit tests (no GPU, network, or Docker needed) |
| `tests/integration/` | Integration tests (pipeline validation) |
| `tests/fixtures/` | Sample data for tests |

## Generated Code — Do Not Edit

The following files are auto-generated. They contain a `# CODE GENERATED ... DO NOT EDIT` header. Do not modify them directly — edit the source and regenerate instead.

| Generated File | Source | Regenerate Command |
|---|---|---|
| `kubeflow-pipelines/docling-standard/standard_convert_pipeline_compiled.yaml` | `standard_convert_pipeline.py` | `cd kubeflow-pipelines/docling-standard && python standard_convert_pipeline.py` |
| `kubeflow-pipelines/docling-vlm/vlm_convert_pipeline_compiled.yaml` | `vlm_convert_pipeline.py` | `cd kubeflow-pipelines/docling-vlm && python vlm_convert_pipeline.py` |

CI enforces that committed YAML matches compiled output (`.github/workflows/compile-kfp.yml`).

## Build & Setup

**Python**: 3.12
**Dev dependencies**: `pip install -r requirements-dev.txt`
**Pipeline dependencies**: `pip install -r kubeflow-pipelines/<pipeline>/requirements.txt`

Container base images are configurable via env vars:
- `PYTHON_BASE_IMAGE` (default: `registry.access.redhat.com/ubi9/python-311:9.6-*`)
- `DOCLING_BASE_IMAGE` (default: `quay.io/fabianofranz/docling-ubi9:2.54.0`)

See `kubeflow-pipelines/common/constants.py:1-9` for defaults.

## Test Commands

```bash
make test-all                    # Run everything: lint + format + tests
make format-python-check         # Check Python formatting (ruff)
make help                        # Show all available targets
```

Run directly with pytest:
```bash
pytest tests/unit/ -v            # Unit tests only
pytest tests/unit/test_kfp_constants.py -v  # Specific test file
```

## Formatting & Linting

Handled by pre-commit hooks (`.pre-commit-config.yaml`). Install with `pre-commit install`.

- **ruff-format**: Code formatting (line-length 88, target Python 3.12)
- **ruff-lint**: Linting with autofixes (rules: E, F, I, B, W, UP)

Ruff config: `pyproject.toml`

## CI/CD (GitHub Actions)

| Workflow | Trigger | What it does |
|---|---|---|
| `validate-python.yml` | PR with `*.py` changes | Ruff lint + format check |
| `compile-kfp.yml` | PR with pipeline changes | Compile pipelines and diff YAML |
| `execute-kfp-localrunners.yml` | PR with pipeline changes | Run pipelines on EC2 GPU runner |

## Navigation Notes

- KFP components use inline imports (required by KFP serialization — `@dsl.component` functions are serialized into isolated containers)
- Pipeline-specific components live in their own directory (e.g., `standard_components.py`)
- Shared components are in `common/components.py` — imported by both pipelines
- `common/constants.py` defines container base images used across all pipelines

## Debugging Guide

**Pipeline compile failures**: Regenerate YAML and commit it — see "Generated Code" section above.

**KFP component imports**: Components use `pylint: disable=import-outside-toplevel` because KFP serializes component functions — imports must be inside the function body.

**Local pipeline testing**: Each pipeline has a `local_run.py` for Docker-based local execution without a Kubernetes cluster.
