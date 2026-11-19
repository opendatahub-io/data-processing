# VLM Policy Document Processing Pipeline

Policy-document-specific [Kubeflow Pipeline](https://www.kubeflow.org/docs/components/pipelines/) (KFP) for [Open Data Hub](https://github.com/opendatahub-io) / [Red Hat OpenShift AI](https://www.redhat.com/en/products/ai/openshift-ai) that converts policy documents containing non-text artifacts (risk matrices, decision trees, compliance charts, forms, screenshots) into structured Markdown with high-fidelity artifact interpretations.

## Architecture

This pipeline combines Docling's standard structural document parsing with a Vision Language Model (VLM) served via KServe for targeted artifact interpretation:

```
import_pdfs → create_pdf_splits → download_docling_models (standard)
                     │
              ParallelFor(splits):
                → docling_convert_standard (image_export_mode="referenced")
                → interpret_policy_artifacts (VLM classify + interpret + assemble)
                → [optional] docling_chunk
```

**Why standard Docling + external VLM?** The standard Docling pipeline provides high-quality structural parsing (headings, tables, numbered clauses, cross-references, reading order) and extracts images as separate files. An external VLM then classifies and interprets each extracted image with artifact-type-specific prompts, producing structured Markdown that replaces the image in the final output.

### Data Flow

1. **Import**: PDFs downloaded from HTTP URLs or S3-compatible storage
2. **Split**: File list divided into batches for parallel processing via KFP `ParallelFor`
3. **Convert**: Docling standard pipeline parses document structure, exports images as referenced PNGs
4. **Interpret**: Each image sent to VLM for classification → type-specific structured interpretation
5. **Assemble**: Image references in Markdown replaced with interpreted content wrapped in artifact markers
6. **Chunk** (optional): DoclingDocument JSON chunked with HybridChunker for RAG workflows

### Artifact Type Classification

Each extracted image is classified into one of these categories:

| Type | Description | Output Format |
|------|-------------|---------------|
| `risk_matrix` | Risk assessment matrix or heat map | Markdown table with severity levels |
| `decision_tree` | Decision flowchart with branches | Nested Markdown list with branches/outcomes |
| `chart` | Bar, line, pie, scatter plots | Data description with values, trends, summary table |
| `form` | Forms, questionnaires, templates | Structured field-value pairs table |
| `screenshot` | UI screenshots | Step-by-step textual walkthrough |
| `diagram` | Architecture, process, org charts | Component/relationship description |
| `other` | Anything else | Detailed structured description |

### Output Format

Each interpreted artifact is wrapped in HTML comment markers for downstream identification:

```markdown
<!-- artifact: risk_matrix -->
| | Low Impact | Medium Impact | High Impact |
|---|---|---|---|
| **High Likelihood** | Medium | High | Critical |
| **Medium Likelihood** | Low | Medium | High |
| **Low Likelihood** | Negligible | Low | Medium |
<!-- /artifact -->
```

### Output Files

Per document, the pipeline produces:

| File | Description |
|------|-------------|
| `{name}_policy.md` | Assembled Markdown with artifact interpretations inline |
| `metadata/{name}_artifacts.json` | Metadata: artifact type, interpretation, source image per artifact |
| `{name}.json` | Original DoclingDocument JSON (for HybridChunker / downstream consumers) |

## Prerequisites

- Access to a KFP 2.x instance (Red Hat OpenShift AI / Data Science Pipelines)
- A VLM served via KServe InferenceService with OpenAI-compatible API (see [Deploying the VLM](#deploying-the-vlm))
- Kubernetes Secret with VLM endpoint configuration (see [Secret Configuration](#secret-configuration))

## Deploying the VLM

Deploy a VLM as a KServe InferenceService using the provided manifest:

```bash
# Optional: create a HuggingFace token secret if your model requires authentication
kubectl create secret generic hf-token --from-literal=token="hf_YOUR_TOKEN"

# Deploy the InferenceService
kubectl apply -f kserve/vlm-inferenceservice.yaml
```

The default manifest deploys `ibm-granite/granite-vision-3.2-2b` using `registry.redhat.io/rhaiis/vllm-cuda-rhel9:0.8.5`. Adjust the `--model` argument for a different model (e.g., `ibm-granite/granite-vision-3.2-8b` for higher quality), and update the image tag to the latest available version from the [Red Hat container catalog](https://catalog.redhat.com/).

Wait for the InferenceService to be ready:

```bash
kubectl get inferenceservice policy-vlm -w
```

The VLM will be accessible in-cluster at:
```
http://policy-vlm-predictor.NAMESPACE.svc.cluster.local:8080/v1/chat/completions
```

## Secret Configuration

Create a Kubernetes Secret with the VLM endpoint configuration and optionally S3 credentials:

```bash
# VLM endpoint only
kubectl create secret generic data-processing-docling-pipeline \
  --from-literal=REMOTE_MODEL_ENDPOINT_URL="http://policy-vlm-predictor.NAMESPACE.svc.cluster.local:8080" \
  --from-literal=REMOTE_MODEL_NAME="ibm-granite/granite-vision-3.2-2b" \
  --from-literal=REMOTE_MODEL_API_KEY=""

# With S3 source (add these keys to the same secret)
kubectl create secret generic data-processing-docling-pipeline \
  --from-literal=REMOTE_MODEL_ENDPOINT_URL="http://policy-vlm-predictor.NAMESPACE.svc.cluster.local:8080" \
  --from-literal=REMOTE_MODEL_NAME="ibm-granite/granite-vision-3.2-2b" \
  --from-literal=REMOTE_MODEL_API_KEY="" \
  --from-literal=S3_ENDPOINT_URL="https://s3.us-east-2.amazonaws.com" \
  --from-literal=S3_ACCESS_KEY="REDACTED" \
  --from-literal=S3_SECRET_KEY="REDACTED" \
  --from-literal=S3_BUCKET="my-bucket" \
  --from-literal=S3_PREFIX="policy-documents"
```

## Installation

Download the [compiled YAML file](policy_convert_pipeline_compiled.yaml?raw=1) and upload it on the _Import pipeline_ screen in the KFP UI.

## Configuration Options

### Document source

- `pdf_filenames`: Comma-separated list of PDF file names to process.
- `pdf_base_url`: HTTP/S base URL where the PDF files are located.
- `pdf_from_s3`: Set to `True` to download from S3-compatible storage instead of URLs.

### Conversion

- `num_splits`: Number of parallel processing batches (increase for large corpora).
- `docling_num_threads`: Threads per Docling conversion container (default: `4`).
- `docling_timeout_per_document`: Timeout in seconds per document (default: `300`).
- `docling_ocr`: Enable OCR for scanned content (default: `True`).
- `docling_table_mode`: Table detection mode — `accurate` (default) or `fast`.

### VLM interpretation

- `vlm_max_tokens`: Maximum tokens for each VLM interpretation response (default: `1024`).
- `vlm_timeout`: Timeout in seconds per VLM API call (default: `120`).

### Chunking

- `docling_chunk_enabled`: Enable HybridChunker chunking (default: `False`).
- `docling_chunk_max_tokens`: Maximum tokens per chunk (default: `512`).
- `docling_chunk_merge_peers`: Merge adjacent small chunks (default: `True`).

## Handling Large Corpora (200+ pages)

For processing large policy document corpora:

1. **Increase `num_splits`**: Set to 5-10 for 200+ page corpora to distribute work across parallel containers
2. **Increase `docling_timeout_per_document`**: Set to 600+ for complex documents
3. **Increase `vlm_timeout`**: Large or complex images may require longer VLM processing
4. **Resource tuning**: Adjust container memory/CPU limits in `policy_convert_pipeline.py` if needed

## Sample Documents

For testing, ~27 policy PDFs are available in the [`asago-ai/asago-examples`](https://github.com/asago-ai/asago-examples) repository under `asago-policy-mapper/policy_examples/` (Amadeus, Cisco, eBay, NHS, ICRC, NYC, etc.).

## Downstream Integration: asago-policy-mapper

The `asago-policy-mapper` currently calls `doc.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER, image_placeholder="")` which silently drops images. With this pipeline:

- Use `{name}_policy.md` directly — it contains the full document text with VLM-interpreted artifacts inline
- The `{name}.json` (DoclingDocument) is still available for `HybridChunker`-based workflows
- Artifact markers (`<!-- artifact: type -->`) allow downstream consumers to identify and handle interpreted content programmatically

## Quality Evaluation

To evaluate interpretation quality:

1. **Manual review**: Compare `_policy.md` output against original PDF for a sample of documents
2. **Artifact coverage**: Check `_artifacts.json` — `total_images` is the number of images found in the document; `total_artifacts` is how many were successfully interpreted; `skipped` lists any images that could not be resolved
3. **Classification accuracy**: Review artifact type assignments in `_artifacts.json`
4. **Structural preservation**: Verify headings, numbered clauses, cross-references, and tables are intact in the assembled Markdown

## Local Testing

```bash
pip install docker kfp

# Set VLM endpoint environment variables (forwarded into the Docker container)
export REMOTE_MODEL_ENDPOINT_URL="http://localhost:8080"
export REMOTE_MODEL_NAME="ibm-granite/granite-vision-3.2-2b"
export REMOTE_MODEL_API_KEY=""

cd data-processing/kubeflow-pipelines/docling-vlm-policy
python local_run.py
```

Requires a Docker-compatible daemon and a reachable VLM endpoint. The `REMOTE_MODEL_*` environment variables are forwarded into the Docker container via `DockerRunner(environment=...)`.

## Compiling from Source

```bash
git clone https://github.com/opendatahub-io/data-processing.git
cd data-processing/kubeflow-pipelines/docling-vlm-policy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python policy_convert_pipeline.py
```

This generates `policy_convert_pipeline_compiled.yaml`.
