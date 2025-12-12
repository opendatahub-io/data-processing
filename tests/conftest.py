"""
Shared pytest configuration and fixtures for notebook testing.
"""

import glob
from pathlib import Path

import pytest


def get_notebook_files():
    """Discover all notebook files in the notebooks directory."""
    notebook_pattern = "notebooks/**/*.ipynb"
    notebook_files = glob.glob(notebook_pattern, recursive=True)
    
    # Convert to Path objects and filter out any non-existent files
    notebook_paths = [Path(f) for f in notebook_files if Path(f).exists()]
    
    return notebook_paths


@pytest.fixture
def notebook_files():
    """Fixture that provides all notebook files for testing."""
    return get_notebook_files()
