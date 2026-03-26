# Common KFP Components

Shared [Kubeflow Pipeline](https://www.kubeflow.org/docs/components/pipelines/) components and constants used by both the standard and VLM pipelines.

## Files

| File | Purpose |
|---|---|
| `components.py` | Shared `@dsl.component` functions used by both pipelines |
| `constants.py` | Base container image defaults (`PYTHON_BASE_IMAGE`, `DOCLING_BASE_IMAGE`) |
| `__init__.py` | Re-exports component functions for pipeline imports |

## Components

| Component | Description | Base Image |
|---|---|---|
| `import_pdfs` | Downloads PDFs from HTTP URLs or S3-compatible storage | `PYTHON_BASE_IMAGE` |
| `create_pdf_splits` | Divides PDF list into N splits for parallel processing | `PYTHON_BASE_IMAGE` |
| `download_docling_models` | Downloads ML models based on pipeline type (standard/VLM) | `DOCLING_BASE_IMAGE` |
| `docling_chunk` | Chunks Docling JSON output into JSONL using HybridChunker | `DOCLING_BASE_IMAGE` |

## Important: Compile-Time Only

These components are imported by `standard_convert_pipeline.py` and `vlm_convert_pipeline.py` at **compile time** to build the pipeline graph. At runtime, each component executes in its own container — see [ADR-001](../../docs/adr/001-kfp-component-serialization.md).

## Modifying Components

After changing code in this directory:

1. Recompile both pipelines:
   ```bash
   cd kubeflow-pipelines/docling-standard && python standard_convert_pipeline.py
   cd kubeflow-pipelines/docling-vlm && python vlm_convert_pipeline.py
   ```
2. Commit the updated `*_compiled.yaml` files
3. CI will fail if compiled YAML doesn't match
