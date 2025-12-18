# Data Processing Kubeflow Pipelines

![Status dev-preview](https://img.shields.io/badge/status-dev--preview-blue)
![GitHub License](https://img.shields.io/github/license/opendatahub-io/data-processing)
![GitHub Commits](https://img.shields.io/github/commit-activity/t/opendatahub-io/data-processing)

In this section of the Data Processing repository, we provide reference **data-processing pipelines** for [Open Data Hub](https://github.com/opendatahub-io) / [Red Hat OpenShift AI](https://www.redhat.com/en/products/ai/openshift-ai), packaged as [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) (KFP).

Two KFP pipelines are included:

- **[docling-standard](docling-standard)**: Standard Docling pipeline for standard conversions, OCR, table structure, enrichments.
- **[docling-vlm](docling-vlm)**: Vision-Language-Model (VLM) pipeline supporting local or remote models.

## 🚀 Key Features

- Docling-based PDF conversion at scale using KFP `ParallelFor` batch splits and parallel executions
- Two customizable pipelines to suit different needs:
  - Standard PDF pipeline (backends, OCR engines, table structure, image export)
  - VLM pipeline (Docling VLM or Granite-Vision pipeline options; remote VLM service supported)
- **Optional document chunking** using Docling's HybridChunker 
- Multiple input sources: HTTP/S URLs or S3/S3-compatible APIs like MinIO
- Secret-based configuration:
  - Remote VLM API configuration via a single mounted Kubernetes Secret
  - S3 endpoint and credentials via a single mounted Kubernetes Secret
- Tunable performance and quality: threads, timeouts, OCR forcing, table mode, PDF backends, enrichments
- Works on OpenShift AI/Kubeflow Pipelines

### Pipeline Architecture

The following diagram shows the overall pipeline flow with optional chunking:

![Pipeline Architecture](/assets/pipeline_architecture.png)

**Input path:**
- **PDF Files** → `import_pdfs` → `docling_convert_standard/vlm` → Markdown + Docling JSON

When `chunk_enabled=True`, the conversion output flows through `docling_chunk` to produce chunked JSON files for RAG workflows.

## 📦 File Structure

```bash
kubeflow-pipelines
|
|- common/
|   |- __init__.py
|   |- components.py          # Shared components (import_pdfs, docling_chunk, etc.)
|   |- constants.py           # Shared constants (base images)
|
|- docling-standard/
|   |- standard_components.py
|   |- standard_convert_pipeline.py
|   |- standard_convert_pipeline_compiled.yaml (generated)
|   |- local_run.py           # Local testing script
|   |- requirements.txt
|
|- docling-vlm/
    |- vlm_components.py
    |- vlm_convert_pipeline.py
    |- vlm_convert_pipeline_compiled.yaml (generated)
    |- local_run.py           # Local testing script
    |- requirements.txt
```

## ✨ Getting Started

### Prerequisites

- Access to a KFP 2.x instance (e.g., Red Hat OpenShift AI)
- Optionally, Kubernetes access to create Secrets in your project/namespace
- Python 3.11+ if you'd like to compile the pipeline for your own needs 

### Installation

Start by importing the compiled YAML file for the desired Docling pipeline (standard or VLM) in the KFP UI.

**For the Standard Pipeline**: 

Download the [compiled YAML file](docling-standard/standard_convert_pipeline_compiled.yaml?raw=1) and upload it on the _Import pipeline_ screen, or import it by URL by pointing it to `https://github.com/opendatahub-io/data-processing/raw/refs/heads/main/kubeflow-pipelines/docling-standard/standard_convert_pipeline_compiled.yaml`.

**For the VLM Pipeline**: 

Download the [compiled YAML file](docling-vlm/vlm_convert_pipeline_compiled.yaml?raw=1) and upload it on the _Import pipeline_ screen, or import it by URL by pointing it to `https://github.com/opendatahub-io/data-processing/raw/refs/heads/main/kubeflow-pipelines/docling-vlm/vlm_convert_pipeline_compiled.yaml`.

Optionally, compile from source to generate the pipeline YAML yourself:

```bash
# Standard pipeline
cd data-processing/kubeflow-pipelines/docling-standard
python standard_convert_pipeline.py

# VLM pipeline
cd data-processing/kubeflow-pipelines/docling-vlm
python vlm_convert_pipeline.py
```

## 🖥️ Running

With the imported pipeline, use the _Create run_ option to configure parameters like the source where your documents are stored and start a conversion.

Once conversion finishes, the converted documents are stored in two output formats: a human-readable **Markdown** representation of the original document, and a **JSON** representing a lossless serialization of the Docling Document. To find where the converted files were stored, check the Graph of your pipeline Run and click the _docling-convert_ box, which should be green indicating the conversion was successful. In the details of _docling-convert_, check the _Output artifacts_ section for the link to the S3-compatible storage where the files were stored. If you need the object storage configurations like the access and secret keys, check the _Pipeline server actions_ > _View pipeline server configuration_ option.

### Configuration options

Both standard and VLM pipelines provide default conversion options that should be a good starting point for most document conversions. For more advanced conversion options, both pipelines expose a set of **runtime parameters** that can be changed to tweak the conversion strategy used by Docling:

- [docling-standard configuration options](docling-standard)
- [docling-vlm configuration options](docling-vlm) 

By default, both pipelines will consume documents stored in an HTTP/S source. To configure the source of the documents you'd like to convert, set the `pdf_base_url` and the `pdf_filenames` (comma-separated list of the file names) parameters. The default values of these parameters point to a sample set of PDFs frequently used to test Docling conversions.

### Example configurations

#### 1) Defaults (Docling with default parameters)

- Standard pipeline defaults include `pdf_backend=dlparse_v4`, `image_export_mode=embedded`, `table_mode=accurate`, `num_threads=4`, `timeout_per_document=300`, `ocr=True`, `force_ocr=False`, `ocr_engine=tesseract`.
- VLM pipeline defaults include `num_threads=4`, `timeout_per_document=300`, `image_export_mode=embedded`, and `remote_model_enabled=False`.

#### 2) Minor tweaks: image and table modes

- Standard pipeline parameters:
  - `docling_image_export_mode`: `embedded` (default), `placeholder`, or `referenced`. In `embedded` mode, the image is  embedded as base64 encoded string. With `placeholder`, only the position of the image is marked in the output. In `referenced` mode, the image is exported in PNG format and referenced from the main exported document.
  - `docling_table_mode`: e.g., `accurate` (default), or `fast`. The mode to use in the table structure model.

#### 3) Forcing OCR

- Standard pipeline:
  - `docling_ocr=True` if enabled, the bitmap content will be processed using OCR.
  - `docling_force_ocr=True` forces full-page OCR regardless of input.
  - `docling_ocr_engine`: `tesseract` (default), `tesserocr`, or `rapidocr`. The OCR engine to use.

#### 4) Using a VLM remotely

- VLM pipeline (`docling-vlm`): set `docling_remote_model_enabled=True` to route processing through a VLM [model service](https://github.com/rh-aiservices-bu/models-aas).
- Configuration for remote VLM models comes from a Kubernetes Secret mounted at `/mnt/secrets` instead of individual KFP parameters.
  - The secret should be present in the same namespace of the pipeline.
  - Secret name: `data-processing-docling-pipeline`.
  - Required keys in the Secret data:
    - `REMOTE_MODEL_ENDPOINT_URL`
    - `REMOTE_MODEL_API_KEY`
    - `REMOTE_MODEL_NAME`
  - The component validates the presence of these files under the mount path when `docling_remote_model_enabled=True`.
  - Example secret creation:
    ```bash
    kubectl create secret generic data-processing-docling-pipeline \
    --from-literal=REMOTE_MODEL_ENDPOINT_URL="https://example/v1/inference" \
    --from-literal=REMOTE_MODEL_API_KEY="REDACTED" \
    --from-literal=REMOTE_MODEL_NAME="granite-vision-3-2"
    ```

#### 5) Consuming documents from S3

If you'd like to consume documents stored in an S3-compatible object storage rather than in an URL:

- Set `pdf_from_s3=True` to download and convert documents from S3 or an S3-compatible service. Set the names of the files to convert in `pdf_filenames`, separated by commas.
- Configuration for the S3 connection comes from a Kubernetes Secret mounted at `/mnt/secrets` instead of individual KFP parameters.
  - The secret should be present in the same namespace of the pipeline.
  - Secret name must be `data-processing-docling-pipeline`.
  - Required keys in the Secret data:
    - `S3_ENDPOINT_URL`
    - `S3_ACCESS_KEY`
    - `S3_SECRET_KEY`
    - `S3_BUCKET`
    - `S3_PREFIX`
  - The pipeline mounts the secret, and the importer reads and uses those values when `pdf_from_s3=True`.
  - Example secret creation:
    ```bash
    kubectl create secret generic data-processing-docling-pipeline \
    --from-literal=S3_ENDPOINT_URL="https://s3.us-east-2.amazonaws.com" \
    --from-literal=S3_ACCESS_KEY="REDACTED" \
    --from-literal=S3_SECRET_KEY="REDACTED" \
    --from-literal=S3_BUCKET="my-bucket" \
    --from-literal=S3_PREFIX="my-pdfs"
    ```

#### 6) Using enrichment models (Standard pipeline)

Toggle enrichments via boolean parameters:
- `docling_enrich_code`, `docling_enrich_formula`, `docling_enrich_picture_classes`, `docling_enrich_picture_description`.

#### 7) Chunking converted documents

Both pipelines support optional document chunking using Docling's [HybridChunker](https://docling-project.github.io/docling/examples/hybrid_chunking/). This splits converted documents into smaller, semantically meaningful chunks ideal for RAG (Retrieval-Augmented Generation) workflows.

**Chunking parameters:**
- `docling_chunk_enabled`: Set to `True` to enable chunking after conversion (default: `False`).
- `docling_chunk_max_tokens`: Maximum tokens per chunk (default: `512`). Adjust based on your embedding model's context limit.
- `docling_chunk_merge_peers`: If `True`, merge adjacent small chunks for better context (default: `True`).

**Tokenizer:** Chunking uses the `sentence-transformers/all-MiniLM-L6-v2` tokenizer for accurate token counting, ensuring chunks are sized appropriately for common embedding models.

**Chunked output location:**
When chunking is enabled, an additional output file is created for each converted document:
- **Filename format**: `{original_name}_chunks.jsonl`
- **Location**: Same output directory as the converted `.json` and `.md` files
- To find the output location, check the Graph of your pipeline Run, click the _docling-chunk_ box, and look in the _Output artifacts_ section.

## 🔧 Advanced customizations

- Increase `num_splits` to **parallelize** across more workers (uses KFP `ParallelFor`).
- Tune `num_threads` and `timeout_per_document`.
- Adjust **container resources** per component, e.g. `set_memory_limit("6G")`, `set_cpu_limit("4")`, in [`docling-standard/standard_convert_pipeline.py`](docling-standard/standard_convert_pipeline.py) or [`docling-vlm/vlm_convert_pipeline.py`](docling-vlm/vlm_convert_pipeline.py).
- Change the value of the `base_image` component parameter ([example](https://github.com/opendatahub-io/data-processing/blob/2bc017c30f862a11fc12c0551c31e8cc93ea6e51/kubeflow-pipelines/docling-standard/standard_components.py#L12)) if you'd like to set a **custom container image** to be used in the pipeline run.
- Recompile the pipeline YAML after code or parameter interface changes to refresh the compiled YAML.
