"""
Shared pytest configuration and fixtures for notebook testing.
"""

import glob
from pathlib import Path

import pytest

# Notebooks to temporarily skip due to environment/dependency issues
SKIP_FOR_NOW = [
       "dataset-preparation-for-RAG.ipynb"   # Milvus connection issues on EC2 instance
]
def get_notebook_files():
    """Discover all notebook files in the notebooks directory."""
    # Get the directory where this conftest.py file is located
    test_dir = Path(__file__).parent
    # Go up one level to the project root
    project_root = test_dir.parent
    
    # Build the notebook pattern from project root
    notebook_pattern = str(project_root / "notebooks" / "**" / "*.ipynb")
    notebook_files = glob.glob(notebook_pattern, recursive=True)
    
    # Convert to Path objects and filter out any non-existent files
    notebook_paths = [Path(f) for f in notebook_files if Path(f).exists()]
    
    # Filter out problematic notebooks for now
    filtered_notebooks = [
        nb for nb in notebook_paths 
        if not any(skip_name in nb.name for skip_name in SKIP_FOR_NOW)
    ]
    
    print(f"Found {len(notebook_paths)} total notebooks, running {len(filtered_notebooks)} (skipped {len(notebook_paths) - len(filtered_notebooks)})")
    
    return filtered_notebooks


@pytest.fixture
def notebook_files():
    """Fixture that provides all notebook files for testing."""
    return get_notebook_files()
