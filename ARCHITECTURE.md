# Architecture

This document describes the high-level design of the `data-processing` repository. It is intended for contributors and AI agents working with the codebase.

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

## Three Subsystems

The repository contains three independent subsystems that share no runtime code:

### 1. Kubeflow Pipelines (`kubeflow-pipelines/`)

KFP pipelines that run on Kubernetes. Two pipeline variants exist:

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

### 2. Jupyter Notebooks (`notebooks/`)

Interactive examples organized by purpose:

- `tutorials/` — Step-by-step guides (e.g., RAG dataset preparation)
- `use-cases/` — Targeted demonstrations (document conversion, chunking, extraction, subset selection)

Notebooks follow the [papermill](https://papermill.readthedocs.io/) convention: each notebook must have a code cell tagged `parameters` to allow parameterized execution in CI.

### 3. Subset Selection Scripts (`scripts/subset_selection/`)

A standalone Python package for selecting representative subsets from large datasets using facility location maximization. This is independent from the KFP pipelines — it has its own `requirements.txt`, CLI, and GPU-accelerated processing pipeline.

**Internal architecture**:
- `cli.py` — Entry point, argument parsing
- `subset_selection.py` — Core logic: dataset loading, embedding generation, fold-based selection
- `encoders/` — Pluggable encoder registry (currently: Arctic embeddings)
- `utils/` — Retry logic, pairwise similarity computation

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

## CI/CD Architecture

```
PR opened/updated
    │
    ├── validate-python.yml ──────── ruff format check on *.py files
    ├── validate-notebooks.yml ───── nbstripout check + parameter validation
    ├── execute-all-notebooks.yml ── Full notebook execution via papermill
    ├── compile-kfp.yml ──────────── Compile pipelines, diff against committed YAML
    └── execute-kfp-localrunners.yml ── Run pipelines on EC2 GPU instance (upstream only)
```

- **Mergify** auto-merges PRs with 1+ approval, passing CI, no conflicts, and no `do-not-merge` label (squash merge)
- **Dependabot** monitors GitHub Actions and pip dependencies weekly (patch/minor only)
- Pipeline local runners use EC2 `g6e.xlarge` GPU instances via self-hosted runners (upstream repo only)

## Testing Strategy

| Test Type | What It Validates | Speed |
|---|---|---|
| **Notebook parameters** (`test_notebook_parameters.py`) | Every notebook has a `parameters`-tagged cell | Fast (~seconds) |
| **Notebook execution** (`test_notebook_execution.py`) | Notebooks run end-to-end via papermill | Slow (minutes, may need GPU) |
| **Python formatting** (`make format-python-check`) | Ruff format compliance | Fast |
| **Notebook formatting** (`make format-notebooks-check`) | Outputs stripped via nbstripout | Fast |
| **Pipeline compilation** (`compile-kfp.yml`) | Compiled YAML matches committed version | Fast |
| **Pipeline execution** (`execute-kfp-localrunners.yml`) | Pipelines run locally via Docker on GPU | Slow (~15 min) |

## Key Design Decisions

Documented as ADRs in `docs/adr/`:

- [ADR-001: KFP Component Serialization](docs/adr/001-kfp-component-serialization.md) — Why imports are inside function bodies
- [ADR-002: Compiled YAML as Source of Truth](docs/adr/002-compiled-yaml-source-of-truth.md) — Why generated YAML is committed
- [ADR-003: Shared Secret Name](docs/adr/003-shared-secret-name.md) — Why S3 and VLM configs share one secret

## Directory Map

```
data-processing/
├── ARCHITECTURE.md                  ← You are here
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
├── notebooks/
│   ├── tutorials/                   ← Step-by-step guides
│   └── use-cases/                   ← Targeted demonstrations
├── scripts/
│   └── subset_selection/            ← Standalone subset selection package
│       ├── cli.py                   ← CLI entry point
│       ├── subset_selection.py      ← Core selection logic
│       ├── encoders/                ← Pluggable embedding encoders
│       └── utils/                   ← Retry logic, similarity computation
├── tests/                           ← Pytest suite
│   ├── conftest.py                  ← Notebook discovery + skip list
│   ├── test_notebook_parameters.py  ← Parameter cell validation
│   └── test_notebook_execution.py   ← Full notebook execution
├── docs/
│   ├── adr/                         ← Architecture Decision Records
│   └── maintainers/                 ← Release strategy
├── Makefile                         ← Build/test/format targets
├── pyproject.toml                   ← Ruff + pytest config
└── .pre-commit-config.yaml          ← ruff-format + nbstripout hooks
```
