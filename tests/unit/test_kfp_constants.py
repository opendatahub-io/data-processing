"""Unit tests for KFP pipeline constants and environment variable defaults."""

import os
from unittest.mock import patch

import pytest


class TestBaseImageConstants:
    def test_python_base_image_default(self):
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to pick up env changes
            import importlib
            import kubeflow_pipelines.common.constants as constants

            importlib.reload(constants)
            assert "ubi9/python-311" in constants.PYTHON_BASE_IMAGE
            assert "registry.access.redhat.com" in constants.PYTHON_BASE_IMAGE

    def test_docling_base_image_default(self):
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import kubeflow_pipelines.common.constants as constants

            importlib.reload(constants)
            assert "docling-ubi9" in constants.DOCLING_BASE_IMAGE

    def test_python_base_image_override(self):
        with patch.dict(os.environ, {"PYTHON_BASE_IMAGE": "my-registry/python:3.12"}):
            import importlib
            import kubeflow_pipelines.common.constants as constants

            importlib.reload(constants)
            assert constants.PYTHON_BASE_IMAGE == "my-registry/python:3.12"

    def test_docling_base_image_override(self):
        with patch.dict(
            os.environ, {"DOCLING_BASE_IMAGE": "my-registry/docling:latest"}
        ):
            import importlib
            import kubeflow_pipelines.common.constants as constants

            importlib.reload(constants)
            assert constants.DOCLING_BASE_IMAGE == "my-registry/docling:latest"
