"""Unit tests for the encoder registry."""

import pytest

from scripts.subset_selection.encoders import ENCODER_REGISTRY, get_encoder_class
from scripts.subset_selection.encoders.arctic_encoder import ArcticEmbedEncoder


class TestEncoderRegistry:
    def test_arctic_registered(self):
        assert "arctic" in ENCODER_REGISTRY

    def test_arctic_returns_correct_class(self):
        cls = get_encoder_class("arctic")
        assert cls is ArcticEmbedEncoder

    def test_unknown_encoder_raises(self):
        with pytest.raises(ValueError, match="Unsupported encoder type"):
            get_encoder_class("nonexistent")

    def test_error_message_lists_supported(self):
        with pytest.raises(ValueError, match="arctic"):
            get_encoder_class("bad")

    def test_registry_is_not_empty(self):
        assert len(ENCODER_REGISTRY) > 0
