"""Tests that verify pipeline compilation artifacts exist and are valid."""

from pathlib import Path

import pytest

PIPELINES_DIR = Path(__file__).parent.parent.parent / "kubeflow-pipelines"


class TestCompiledYamlExists:
    """Verify that compiled YAML files exist for each pipeline."""

    def test_standard_compiled_yaml_exists(self):
        """Standard pipeline compiled YAML should exist."""
        compiled = PIPELINES_DIR / "docling-standard" / "standard_convert_pipeline_compiled.yaml"
        assert compiled.exists(), f"Missing compiled YAML: {compiled}"

    def test_vlm_compiled_yaml_exists(self):
        """VLM pipeline compiled YAML should exist."""
        compiled = PIPELINES_DIR / "docling-vlm" / "vlm_convert_pipeline_compiled.yaml"
        assert compiled.exists(), f"Missing compiled YAML: {compiled}"

    def test_standard_compiled_yaml_not_empty(self):
        """Standard pipeline compiled YAML should not be empty."""
        compiled = PIPELINES_DIR / "docling-standard" / "standard_convert_pipeline_compiled.yaml"
        assert compiled.stat().st_size > 0, "Compiled YAML is empty"

    def test_vlm_compiled_yaml_not_empty(self):
        """VLM pipeline compiled YAML should not be empty."""
        compiled = PIPELINES_DIR / "docling-vlm" / "vlm_convert_pipeline_compiled.yaml"
        assert compiled.stat().st_size > 0, "Compiled YAML is empty"


class TestPipelineSourceFiles:
    """Verify pipeline source files exist."""

    @pytest.mark.parametrize("pipeline,filename", [
        ("docling-standard", "standard_convert_pipeline.py"),
        ("docling-standard", "standard_components.py"),
        ("docling-vlm", "vlm_convert_pipeline.py"),
        ("docling-vlm", "vlm_components.py"),
    ])
    def test_pipeline_source_exists(self, pipeline, filename):
        """Each pipeline should have its source files."""
        source = PIPELINES_DIR / pipeline / filename
        assert source.exists(), f"Missing source: {source}"

    @pytest.mark.parametrize("filename", [
        "components.py",
        "constants.py",
        "__init__.py",
    ])
    def test_common_module_exists(self, filename):
        """Common module should have all required files."""
        source = PIPELINES_DIR / "common" / filename
        assert source.exists(), f"Missing common file: {source}"

    def test_each_pipeline_has_local_run(self):
        """Each pipeline should have a local_run.py for local testing."""
        for pipeline in ["docling-standard", "docling-vlm"]:
            local_run = PIPELINES_DIR / pipeline / "local_run.py"
            assert local_run.exists(), f"Missing local_run.py in {pipeline}"

    def test_each_pipeline_has_requirements(self):
        """Each pipeline should have its own requirements.txt."""
        for pipeline in ["docling-standard", "docling-vlm"]:
            reqs = PIPELINES_DIR / pipeline / "requirements.txt"
            assert reqs.exists(), f"Missing requirements.txt in {pipeline}"
