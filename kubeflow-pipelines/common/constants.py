import os

# Base container images used across all Docling Kubeflow Pipelines
PYTHON_BASE_IMAGE = os.getenv(
    "PYTHON_BASE_IMAGE", "quay.io/amaredia/aipcc-docling-image"
)
DOCLING_BASE_IMAGE = os.getenv(
    "DOCLING_BASE_IMAGE", "quay.io/amaredia/aipcc-docling-image"
)
