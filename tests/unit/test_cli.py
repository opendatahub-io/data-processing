"""Unit tests for subset selection CLI argument parsing."""

import sys
from unittest.mock import patch

import pytest


def _parse_with_args(args: list[str]):
    """Run parse_args with given CLI arguments."""
    from scripts.subset_selection.cli import parse_args

    with patch.object(sys, "argv", ["subset_selection"] + args):
        return parse_args()


class TestParseArgs:
    def test_required_args(self):
        args = _parse_with_args(["--input", "data.jsonl", "--subset-sizes", "0.5"])
        assert args.input == ["data.jsonl"]
        assert args.subset_sizes == "0.5"

    def test_multiple_inputs(self):
        args = _parse_with_args(
            ["--input", "a.jsonl", "b.jsonl", "--subset-sizes", "100"]
        )
        assert args.input == ["a.jsonl", "b.jsonl"]

    def test_defaults(self):
        args = _parse_with_args(["--input", "d.jsonl", "--subset-sizes", "0.1"])
        assert args.output_dir == "output"
        assert args.batch_size == 100000
        assert args.num_folds == 50
        assert args.epsilon == 160.0
        assert args.num_gpus is None
        assert args.combine_files is False
        assert args.testing_mode is False
        assert args.encoder_type == "arctic"
        assert args.template_name == "conversation"
        assert args.seed == 42

    def test_custom_values(self):
        args = _parse_with_args([
            "--input", "d.jsonl",
            "--subset-sizes", "0.1,0.5",
            "--output-dir", "/tmp/out",
            "--batch-size", "500",
            "--num-folds", "10",
            "--epsilon", "0.5",
            "--num-gpus", "2",
            "--combine-files",
            "--testing-mode",
            "--encoder-type", "custom",
            "--encoder-model", "my-model",
            "--template-name", "qa",
            "--seed", "123",
        ])
        assert args.output_dir == "/tmp/out"
        assert args.batch_size == 500
        assert args.num_folds == 10
        assert args.epsilon == 0.5
        assert args.num_gpus == 2
        assert args.combine_files is True
        assert args.testing_mode is True
        assert args.encoder_type == "custom"
        assert args.encoder_model == "my-model"
        assert args.template_name == "qa"
        assert args.seed == 123

    def test_missing_required_exits(self):
        with pytest.raises(SystemExit):
            _parse_with_args(["--input", "d.jsonl"])  # missing --subset-sizes

    def test_missing_input_exits(self):
        with pytest.raises(SystemExit):
            _parse_with_args(["--subset-sizes", "0.5"])  # missing --input


class TestSubsetSizeParsing:
    """Test the subset size string parsing logic from main()."""

    @staticmethod
    def _parse_sizes(sizes_str: str) -> list:
        """Replicate the parsing logic from cli.py main()."""
        subset_sizes = []
        for size_str in sizes_str.split(","):
            size_str = size_str.strip()
            if "." in size_str:
                subset_sizes.append(float(size_str))
            else:
                subset_sizes.append(int(size_str))
        return subset_sizes

    def test_single_percentage(self):
        assert self._parse_sizes("0.5") == [0.5]

    def test_multiple_percentages(self):
        assert self._parse_sizes("0.1,0.5") == [0.1, 0.5]

    def test_single_absolute(self):
        assert self._parse_sizes("1000") == [1000]

    def test_mixed(self):
        assert self._parse_sizes("0.1,500") == [0.1, 500]

    def test_whitespace_handling(self):
        assert self._parse_sizes("0.1, 0.5 , 1000") == [0.1, 0.5, 1000]

    def test_integer_detection(self):
        """Values without dots should be parsed as int, not float."""
        sizes = self._parse_sizes("100")
        assert isinstance(sizes[0], int)

    def test_float_detection(self):
        """Values with dots should be parsed as float."""
        sizes = self._parse_sizes("0.5")
        assert isinstance(sizes[0], float)
