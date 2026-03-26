# Architecture

This document describes the high-level design of the `data-processing` repository, focused on the Kubeflow Pipelines (KFP) subsystem.

## System Purpose

This repository provides reference data-processing pipelines for Open Data Hub / Red Hat OpenShift AI. It converts documents (primarily PDFs) into structured formats (JSON, Markdown) and optionally chunks them for RAG workflows, using the [Docling](https://docling-project.github.io/docling/) toolkit.

## High-Level Data Flow

```
                          ┌─────────────────────────────────────────────────────┐
                          │              Kubeflow Pipeline (KFP)                │
                          │                                                     │
  PDF Source              │  ┌────────────┐   ┌──────────────┐   ┌───────────┐ │   Output
  (HTTP/S3)  ────────────►│  │ import_pdfs│──►│create_splits │──►│ ParallelFor│ │──► JSON + MD
                          │  └────────────┘   └──────────────┘   │           │ │   (+ JSONL
                          │                                       │ ┌───────┐ │ │    chunks)
                          │  ┌──────────────────┐                 │ │convert│ │ │
                          │  │download_models   │────────────────►│ │  ──►  │ │ │
                          │  └──────────────────┘                 │ │ chunk │ │ │
                          │                                       │ └───────┘ │ │
                          │                                       └───────────┘ │
                          └─────────────────────────────────────────────────────┘
```

### Pipeline Steps

1. **`import_pdfs`** — Downloads PDFs from HTTP URLs or S3-compatible storage
2. **`create_pdf_splits`** — Divides PDF list into N splits for parallel processing
3. **`download_docling_models`** — Downloads ML models (layout, tableformer, or VLM)
4. **`docling_convert_*`** — Converts PDFs to Docling JSON + Markdown (runs in parallel via `ParallelFor`)
5. **`docling_chunk`** *(optional)* — Splits converted documents into semantic chunks (JSONL output)

## KFP Pipeline Architecture (`kubeflow-pipelines/`)

Two pipeline variants exist:

| Pipeline | Conversion Strategy | Models Used |
|---|---|---|
| **Standard** (`docling-standard/`) | Traditional OCR + layout analysis | Layout model, TableFormer |
| **VLM** (`docling-vlm/`) | Vision-language model inference | SmolVLM, SmolDocling, or remote API |

**Key architectural constraint**: KFP `@dsl.component` functions are serialized by the KFP SDK and executed in isolated containers. This means:
- All imports must be inside the function body (not at module top)
- Components cannot import from project-level modules at runtime
- Each component is self-contained with its own dependencies
- The `common/` module is only used at **compile time** to define shared component functions

See [ADR-001](docs/adr/001-kfp-component-serialization.md) for details.

**Component ownership**:
- `common/components.py` — Shared components used by both pipelines: `import_pdfs`, `create_pdf_splits`, `download_docling_models`, `docling_chunk`
- `docling-standard/standard_components.py` — Standard-specific: `docling_convert_standard`
- `docling-vlm/vlm_components.py` — VLM-specific: `docling_convert_vlm`

**Compiled YAML**: Each pipeline has a `*_compiled.yaml` file that is the KFP-deployable artifact. These are **generated code** — produced by running `python *_convert_pipeline.py`. CI enforces that committed YAML matches what the compiler produces.

## Container Images

Two base images are used across all KFP components, configurable via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PYTHON_BASE_IMAGE` | `registry.access.redhat.com/ubi9/python-311:9.6-*` | Lightweight components (import, split) |
| `DOCLING_BASE_IMAGE` | `quay.io/fabianofranz/docling-ubi9:2.54.0` | Components needing Docling + ML models |

Defined in `kubeflow-pipelines/common/constants.py`.

## Secrets Architecture

KFP pipelines use Kubernetes Secrets mounted as volumes (not environment variables):

| Secret Name | Mount Path | Used When | Keys |
|---|---|---|---|
| `data-processing-docling-pipeline` | `/mnt/secrets` | `pdf_from_s3=True` | `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`, `S3_PREFIX` |
| `data-processing-docling-pipeline` | `/mnt/secrets` | `remote_model_enabled=True` (VLM only) | `REMOTE_MODEL_ENDPOINT_URL`, `REMOTE_MODEL_API_KEY`, `REMOTE_MODEL_NAME` |

Both use the same secret name. Components read individual keys as files from the mount path.

## Key Design Decisions

Documented as ADRs in `docs/adr/`:

- [ADR-001: KFP Component Serialization](docs/adr/001-kfp-component-serialization.md) — Why imports are inside function bodies
- [ADR-002: Compiled YAML as Source of Truth](docs/adr/002-compiled-yaml-source-of-truth.md) — Why generated YAML is committed
- [ADR-003: Shared Secret Name](docs/adr/003-shared-secret-name.md) — Why S3 and VLM configs share one secret

## Directory Map

```
data-processing/
├── kubeflow-pipelines/
│   ├── common/                      ← Shared KFP components + constants
│   │   ├── components.py            ← import_pdfs, create_pdf_splits, download_docling_models, docling_chunk
│   │   └── constants.py             ← Base image defaults
│   ├── docling-standard/            ← Standard conversion pipeline
│   │   ├── standard_components.py   ← docling_convert_standard
│   │   ├── standard_convert_pipeline.py ← Pipeline definition + compiler
│   │   ├── standard_convert_pipeline_compiled.yaml ← GENERATED — do not edit
│   │   └── local_run.py             ← Docker-based local testing
│   └── docling-vlm/                 ← VLM conversion pipeline
│       ├── vlm_components.py        ← docling_convert_vlm
│       ├── vlm_convert_pipeline.py  ← Pipeline definition + compiler
│       ├── vlm_convert_pipeline_compiled.yaml ← GENERATED — do not edit
│       └── local_run.py             ← Docker-based local testing
├── tests/                           ← Pytest suite
├── docs/
│   └── adr/                         ← Architecture Decision Records
├── Makefile                         ← Build/test/format targets
├── pyproject.toml                   ← Ruff + pytest config
└── .pre-commit-config.yaml          ← Pre-commit hooks
```
