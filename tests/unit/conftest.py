"""Auto-apply 'unit' marker to all tests in this directory."""

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.unit)
