"""
Shared pytest configuration and fixtures.

Test isolation is enforced by directory structure:
  tests/unit/        — Fast tests, no external dependencies (auto-marked @pytest.mark.unit)
  tests/integration/ — Slow tests needing GPU, network, or Docker (auto-marked @pytest.mark.integration)

Each subdirectory has its own conftest.py that auto-applies the appropriate marker.
"""

import glob
from pathlib import Path

import pytest

# Notebooks to temporarily skip
SKIP_FOR_NOW = []


def get_notebook_files():
    """Discover all notebook files in the notebooks directory."""
    test_dir = Path(__file__).parent
    project_root = test_dir.parent

    notebook_pattern = str(project_root / "notebooks" / "**" / "*.ipynb")
    notebook_files = glob.glob(notebook_pattern, recursive=True)

    notebook_paths = [Path(f) for f in notebook_files if Path(f).exists()]

    filtered_notebooks = [
        nb for nb in notebook_paths
        if nb.name not in SKIP_FOR_NOW
    ]

    print(f"Found {len(notebook_paths)} total notebooks, running {len(filtered_notebooks)} (skipped {len(notebook_paths) - len(filtered_notebooks)})")

    return filtered_notebooks


@pytest.fixture
def notebook_files():
    """Fixture that provides all notebook files for testing."""
    return get_notebook_files()
