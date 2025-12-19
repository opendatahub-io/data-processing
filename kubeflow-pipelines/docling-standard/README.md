# Standard Data Processing Kubeflow Pipeline

Standard (non-VLM) Docling [Kubeflow Pipeline](https://www.kubeflow.org/docs/components/pipelines/) (KFP) for [Open Data Hub](https://github.com/opendatahub-io) / [Red Hat OpenShift AI](https://www.redhat.com/en/products/ai/openshift-ai) supporting standard conversions, OCR, table structure, and enrichments.

## Installation

Download the [compiled YAML file](standard_convert_pipeline_compiled.yaml?raw=1) and upload it on the _Import pipeline_ screen, or import it by URL by pointing it to `https://github.com/opendatahub-io/data-processing/raw/refs/heads/main/kubeflow-pipelines/docling-standard/standard_convert_pipeline_compiled.yaml`.

## Configuration options

The following configuration options are available as KFP parameters when you _Create run_:

- `docling_enrich_code`: Enable the code enrichment model in the pipeline.
- `docling_enrich_formula`: Enable the formula enrichment model in the pipeline.
- `docling_enrich_picture_classes`: Enable the picture classification enrichment model in the pipeline.
- `docling_enrich_picture_description`: Enable the picture description model in the pipeline.
- `docling_force_ocr`: Replace any existing text with OCR generated text over the full content.
- `docling_image_export_mode`: Image export mode for the document. In `embedded` mode, the image is embedded as base64 encoded string. With `placeholder`, only the position of the image is marked in the output. In `referenced` mode, the image is exported in PNG format and referenced from the main exported document.
- `docling_num_threads`: Number of threads to be used internally by the Docling engine.
- `docling_ocr`: If enabled, the bitmap content will be processed using OCR.
- `docling_ocr_engine`: The OCR engine to use. Current values are: `easyocr`.
- `docling_pdf_backend`: The PDF backend to use. `pypdfium2`, `dlparse_v1`, `dlparse_v2`, or `dlparse_v4`.
- `docling_table_mode`: The mode to use in the table structure model. `accurate` or `fast`.
- `docling_timeout_per_document`: Timeout for each single document conversion.
- `num_splits`: Number of splits to create from the list of file names. Each split will be one container in KFP processing a chunk of the files. Used for parallelism and horizontal scalability.
- `pdf_base_url`: A publicly accessible HTTP/S base URL where the PDF files listed in `pdf_filenames` are located.
- `pdf_filenames`: List of PDF file names to process, separated by commas.
- `pdf_from_s3`: If `True`, PDF files will be fetched from an S3-compatible object storage rather than `pdf_base_url`. A secret must be configured as described in [docs](../README.md).

### Chunking options

Optional document chunking using Docling's [HybridChunker](https://docling-project.github.io/docling/examples/hybrid_chunking/):

- `docling_chunk_enabled`: If `True`, chunk converted documents into smaller pieces (default: `False`).
- `docling_chunk_max_tokens`: Maximum tokens per chunk (default: `512`).
- `docling_chunk_merge_peers`: If `True`, merge adjacent small chunks for better context (default: `True`).

Chunking uses the `sentence-transformers/all-MiniLM-L6-v2` tokenizer for accurate token counting.

**Chunked output**: When enabled, creates `{filename}_chunks.jsonl` files (one JSON object per line) in the same output directory as the converted documents. See [main docs](../README.md) for output format details.

## Local testing

You can test the pipeline locally using Docker before deploying to KFP.

### Prerequisites

```bash
pip install docker kfp
```

Requires a Docker-compatible daemon (Docker or Podman socket).

### Run locally

```bash
cd data-processing/kubeflow-pipelines/docling-standard
python local_run.py
```

This runs `convert_pipeline_local()` which converts PDFs and chunks the output.

## Compiling from source

### Clone repository, create venv, install dependencies

```bash
git clone https://github.com/opendatahub-io/data-processing.git
cd data-processing/kubeflow-pipelines/docling-standard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Compile the kubeflow pipeline

This generates `standard_convert_pipeline_compiled.yaml`:

```bash
python standard_convert_pipeline.py
```