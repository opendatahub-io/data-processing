"""Tests for KFP pipeline constants."""

import os
import sys
from pathlib import Path
from unittest import mock

# Add kubeflow-pipelines to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kubeflow-pipelines"))


class TestBaseImageConstants:
    """Test that base image constants are properly configured."""

    def test_python_base_image_has_default(self):
        """PYTHON_BASE_IMAGE should have a non-empty default."""
        # Re-import to get fresh defaults (without env var override)
        with mock.patch.dict(os.environ, {}, clear=True):
            # Force reimport
            if "common.constants" in sys.modules:
                del sys.modules["common.constants"]
            from common.constants import PYTHON_BASE_IMAGE

            assert PYTHON_BASE_IMAGE
            assert "python" in PYTHON_BASE_IMAGE.lower()

    def test_docling_base_image_has_default(self):
        """DOCLING_BASE_IMAGE should have a non-empty default."""
        with mock.patch.dict(os.environ, {}, clear=True):
            if "common.constants" in sys.modules:
                del sys.modules["common.constants"]
            from common.constants import DOCLING_BASE_IMAGE

            assert DOCLING_BASE_IMAGE
            assert "docling" in DOCLING_BASE_IMAGE.lower()

    def test_python_base_image_overridable_via_env(self):
        """PYTHON_BASE_IMAGE should be overridable via environment variable."""
        custom_image = "my-registry/python:3.12"
        with mock.patch.dict(os.environ, {"PYTHON_BASE_IMAGE": custom_image}):
            if "common.constants" in sys.modules:
                del sys.modules["common.constants"]
            from common.constants import PYTHON_BASE_IMAGE

            assert PYTHON_BASE_IMAGE == custom_image

    def test_docling_base_image_overridable_via_env(self):
        """DOCLING_BASE_IMAGE should be overridable via environment variable."""
        custom_image = "my-registry/docling:latest"
        with mock.patch.dict(os.environ, {"DOCLING_BASE_IMAGE": custom_image}):
            if "common.constants" in sys.modules:
                del sys.modules["common.constants"]
            from common.constants import DOCLING_BASE_IMAGE

            assert DOCLING_BASE_IMAGE == custom_image
