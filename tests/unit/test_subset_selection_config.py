"""Unit tests for subset selection configuration dataclasses."""

import pytest


# --- BasicConfig tests ---


class TestBasicConfig:
    """Tests for BasicConfig dataclass validation."""

    def test_default_values(self):
        from scripts.subset_selection.subset_selection import BasicConfig

        config = BasicConfig()
        assert config.output_dir == "output"
        assert config.batch_size == 100000
        assert config.num_folds == 50
        assert config.combine_files is False
        assert config.epsilon == 160.0

    def test_custom_values(self):
        from scripts.subset_selection.subset_selection import BasicConfig

        config = BasicConfig(output_dir="/tmp/out", batch_size=500, num_folds=10, epsilon=0.5)
        assert config.output_dir == "/tmp/out"
        assert config.batch_size == 500
        assert config.num_folds == 10
        assert config.epsilon == 0.5

    def test_epsilon_zero_raises(self):
        from scripts.subset_selection.subset_selection import BasicConfig

        with pytest.raises(ValueError, match="epsilon must be between 0 and 160"):
            BasicConfig(epsilon=0)

    def test_epsilon_negative_raises(self):
        from scripts.subset_selection.subset_selection import BasicConfig

        with pytest.raises(ValueError, match="epsilon must be between 0 and 160"):
            BasicConfig(epsilon=-1.0)

    def test_epsilon_over_160_raises(self):
        from scripts.subset_selection.subset_selection import BasicConfig

        with pytest.raises(ValueError, match="epsilon must be between 0 and 160"):
            BasicConfig(epsilon=161.0)

    def test_epsilon_at_boundary(self):
        from scripts.subset_selection.subset_selection import BasicConfig

        config = BasicConfig(epsilon=160.0)
        assert config.epsilon == 160.0

        config = BasicConfig(epsilon=0.001)
        assert config.epsilon == 0.001

    def test_validate_epsilon_small_dataset_no_warning(self, caplog):
        """Large dataset should not trigger warnings."""
        import logging
        from scripts.subset_selection.subset_selection import BasicConfig

        config = BasicConfig(epsilon=0.5)
        with caplog.at_level(logging.WARNING):
            config.validate_epsilon_for_dataset_size(200000)
        assert "highly recommended" not in caplog.text

    def test_validate_epsilon_small_dataset_warns(self, caplog):
        """Small dataset with high epsilon should warn."""
        import logging
        from scripts.subset_selection.subset_selection import BasicConfig

        config = BasicConfig(epsilon=160.0)
        with caplog.at_level(logging.WARNING):
            config.validate_epsilon_for_dataset_size(500)
        assert "highly recommended" in caplog.text
        assert "too high" in caplog.text

    def test_validate_epsilon_small_dataset_low_epsilon_warns_once(self, caplog):
        """Small dataset with low epsilon should warn about size but not epsilon."""
        import logging
        from scripts.subset_selection.subset_selection import BasicConfig

        config = BasicConfig(epsilon=0.5)
        with caplog.at_level(logging.WARNING):
            config.validate_epsilon_for_dataset_size(500)
        assert "highly recommended" in caplog.text
        assert "too high" not in caplog.text


# --- EncoderConfig tests ---


class TestEncoderConfig:
    def test_defaults(self):
        from scripts.subset_selection.subset_selection import EncoderConfig

        config = EncoderConfig()
        assert config.encoder_type == "arctic"
        assert "snowflake-arctic" in config.encoder_model.lower()
        assert config.testing_mode is False

    def test_custom_encoder(self):
        from scripts.subset_selection.subset_selection import EncoderConfig

        config = EncoderConfig(encoder_type="custom", encoder_model="my-model")
        assert config.encoder_type == "custom"
        assert config.encoder_model == "my-model"


# --- TemplateConfig tests ---


class TestTemplateConfig:
    def test_default_templates(self):
        from scripts.subset_selection.subset_selection import TemplateConfig

        config = TemplateConfig()
        assert config.template_name == "conversation"
        assert "default" in config.templates
        assert "conversation" in config.templates
        assert "qa" in config.templates

    def test_custom_template_name(self):
        from scripts.subset_selection.subset_selection import TemplateConfig

        config = TemplateConfig(template_name="qa")
        assert config.template_name == "qa"


# --- ProcessingConfig tests ---


class TestProcessingConfig:
    def test_valid_percentage_sizes(self):
        from scripts.subset_selection.subset_selection import ProcessingConfig

        config = ProcessingConfig(
            input_files=["data.jsonl"],
            subset_sizes=[0.1, 0.5],
        )
        assert config.subset_sizes == [0.1, 0.5]

    def test_valid_absolute_sizes(self):
        from scripts.subset_selection.subset_selection import ProcessingConfig

        config = ProcessingConfig(
            input_files=["data.jsonl"],
            subset_sizes=[100, 500],
        )
        assert config.subset_sizes == [100, 500]

    def test_mixed_sizes(self):
        from scripts.subset_selection.subset_selection import ProcessingConfig

        config = ProcessingConfig(
            input_files=["data.jsonl"],
            subset_sizes=[0.1, 500],
        )
        assert config.subset_sizes == [0.1, 500]

    def test_subset_sizes_not_list_raises(self):
        from scripts.subset_selection.subset_selection import ProcessingConfig

        with pytest.raises(ValueError, match="subset_sizes must be a list"):
            ProcessingConfig(input_files=["data.jsonl"], subset_sizes=0.5)

    def test_subset_sizes_string_element_raises(self):
        from scripts.subset_selection.subset_selection import ProcessingConfig

        with pytest.raises(ValueError, match="integers or floats"):
            ProcessingConfig(input_files=["data.jsonl"], subset_sizes=["50%"])

    def test_negative_absolute_raises(self):
        from scripts.subset_selection.subset_selection import ProcessingConfig

        with pytest.raises(ValueError, match="positive"):
            ProcessingConfig(input_files=["data.jsonl"], subset_sizes=[-10])

    def test_zero_absolute_raises(self):
        from scripts.subset_selection.subset_selection import ProcessingConfig

        with pytest.raises(ValueError, match="positive"):
            ProcessingConfig(input_files=["data.jsonl"], subset_sizes=[0])

    def test_percentage_over_100_raises(self):
        from scripts.subset_selection.subset_selection import ProcessingConfig

        with pytest.raises(ValueError, match="between 0 and 100"):
            ProcessingConfig(input_files=["data.jsonl"], subset_sizes=[150.0])

    def test_percentage_zero_raises(self):
        from scripts.subset_selection.subset_selection import ProcessingConfig

        with pytest.raises(ValueError, match="between 0 and 100"):
            ProcessingConfig(input_files=["data.jsonl"], subset_sizes=[0.0])
