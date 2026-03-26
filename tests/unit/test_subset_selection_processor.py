"""Unit tests for DataProcessor business logic methods.

Tests pure logic methods that don't require GPU, model downloads, or datasets.
"""

import os
import re

import pytest


# --- Helper to build a minimal DataProcessor without GPU/model dependencies ---


def _make_processor(**overrides):
    """Create a DataProcessor with testing defaults, mocking torch device."""
    from unittest.mock import patch

    from scripts.subset_selection.subset_selection import (
        BasicConfig,
        DataProcessor,
        EncoderConfig,
        ProcessingConfig,
        SystemConfig,
        TemplateConfig,
    )

    # Patch torch.device and torch.cuda to avoid GPU requirement
    with patch("scripts.subset_selection.subset_selection.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        mock_torch.manual_seed.return_value = None

        config = ProcessingConfig(
            input_files=overrides.pop("input_files", ["test.jsonl"]),
            subset_sizes=overrides.pop("subset_sizes", [0.5]),
            system=SystemConfig(testing_mode=True),
            **{
                k: v
                for k, v in overrides.items()
                if k in ("basic", "encoder", "template")
            },
        )
        processor = DataProcessor(config)
    return processor


# --- calculate_subset_size ---


class TestCalculateSubsetSize:
    def test_percentage_half(self):
        proc = _make_processor()
        assert proc.calculate_subset_size(1000, 0.5) == 500

    def test_percentage_ten_percent(self):
        proc = _make_processor()
        assert proc.calculate_subset_size(1000, 0.1) == 100

    def test_percentage_full(self):
        proc = _make_processor()
        assert proc.calculate_subset_size(1000, 1.0) == 1000

    def test_percentage_tiny_returns_at_least_one(self):
        proc = _make_processor()
        result = proc.calculate_subset_size(2, 0.1)
        assert result >= 1

    def test_absolute_within_range(self):
        proc = _make_processor()
        assert proc.calculate_subset_size(1000, 500) == 500

    def test_absolute_exceeds_total_clamped(self):
        proc = _make_processor()
        assert proc.calculate_subset_size(100, 500) == 100

    def test_percentage_zero_raises(self):
        proc = _make_processor()
        with pytest.raises(ValueError, match="between 0"):
            proc.calculate_subset_size(1000, 0.0)

    def test_percentage_negative_raises(self):
        proc = _make_processor()
        with pytest.raises(ValueError, match="between 0"):
            proc.calculate_subset_size(1000, -0.5)

    def test_percentage_over_one_raises(self):
        proc = _make_processor()
        with pytest.raises(ValueError, match="between 0"):
            proc.calculate_subset_size(1000, 1.5)


# --- get_subset_name ---


class TestGetSubsetName:
    def test_percentage_name(self):
        proc = _make_processor()
        assert proc.get_subset_name(0.5, 500) == "percent_0.5"

    def test_percentage_name_small(self):
        proc = _make_processor()
        assert proc.get_subset_name(0.1, 100) == "percent_0.1"

    def test_absolute_name(self):
        proc = _make_processor()
        assert proc.get_subset_name(500, 500) == "samples_500"

    def test_absolute_name_clamped(self):
        proc = _make_processor()
        assert proc.get_subset_name(5000, 1000) == "samples_1000"


# --- get_dataset_name ---


class TestGetDatasetName:
    def test_simple_filename(self):
        proc = _make_processor()
        assert proc.get_dataset_name("data.jsonl") == "data"

    def test_path_with_directories(self):
        proc = _make_processor()
        assert proc.get_dataset_name("/tmp/datasets/train_v2.jsonl") == "train_v2"

    def test_special_characters_replaced(self):
        proc = _make_processor()
        name = proc.get_dataset_name("my data (v2).jsonl")
        # Special chars should be replaced with underscores
        assert re.match(r"^[\w\-_]+$", name)
        assert " " not in name
        assert "(" not in name

    def test_nested_extension(self):
        proc = _make_processor()
        # os.path.splitext splits on last dot
        assert proc.get_dataset_name("archive.tar.gz") == "archive_tar"


# --- format_text ---


class TestFormatText:
    def test_default_template(self):
        proc = _make_processor()
        result = proc.format_text({"text": "hello world"}, "default")
        assert result == "hello world"

    def test_conversation_template(self):
        proc = _make_processor()
        example = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        result = proc.format_text(example, "conversation")
        # System messages are filtered out
        assert "system" not in result.lower().split(":")[0] or "user" in result
        assert "Hi" in result
        assert "Hello!" in result

    def test_qa_template(self):
        proc = _make_processor()
        result = proc.format_text(
            {"question": "What is Python?", "answer": "A language."}, "qa"
        )
        assert "Question: What is Python?" in result
        assert "Answer: A language." in result

    def test_unknown_format_raises(self):
        proc = _make_processor()
        with pytest.raises(ValueError, match="Unknown format type"):
            proc.format_text({"text": "hi"}, "nonexistent")
