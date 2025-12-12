# VLM Data Processing Kubeflow Pipeline

Vision-Language-Model (VLM) Docling [Kubeflow Pipeline](https://www.kubeflow.org/docs/components/pipelines/) (KFP) for [Open Data Hub](https://github.com/opendatahub-io) / [Red Hat OpenShift AI](https://www.redhat.com/en/products/ai/openshift-ai) supporting local or remote models.

## Installation

Download the [compiled YAML file](vlm_convert_pipeline_compiled.yaml?raw=1) and upload it on the _Import pipeline_ screen, or import it by URL by pointing it to `https://github.com/opendatahub-io/data-processing/raw/refs/heads/main/kubeflow-pipelines/docling-vlm/vlm_convert_pipeline_compiled.yaml`.

## Configuration options

The following configuration options are available as KFP parameters when you _Create run_:

- `docling_image_export_mode`: Image export mode for the document. In `embedded` mode, the image is embedded as base64 encoded string. With `placeholder`, only the position of the image is marked in the output. In `referenced` mode, the image is exported in PNG format and referenced from the main exported document.
- `docling_num_threads`: Number of threads to be used internally by the Docling engine.
- `docling_remote_model_enabled`: If `True`, models will be served from a remote model service API ([example](https://github.com/rh-aiservices-bu/models-aas)). A secret must be configured as described in [docs](../README.md).
- `docling_timeout_per_document`: Timeout for each single document conversion.
- `num_splits`: Number of splits to create from the list of file names. Each split will be one container in KFP processing a chunk of the files. Used for parallelism and horizontal scalability.
- `pdf_base_url`: A publicly accessible HTTP/S base URL where the PDF files listed in `pdf_filenames` are located.
- `pdf_filenames`: List of PDF file names to process, separated by commas.
- `pdf_from_s3`: If `True`, PDF files will be fetched from an S3-compatible object storage rather than `pdf_base_url`. A secret must be configured as described in [docs](../README.md).
 
## Compiling from source

### Clone repository, create venv, install dependencies

```bash
git clone https://github.com/opendatahub-io/data-processing.git
cd data-processing/kubeflow-pipelines/docling-vlm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Compile the kubeflow pipeline

This generates `vlm_convert_pipeline_compiled.yaml`:

```bash
python vlm_convert_pipeline.py
```

