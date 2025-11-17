"""
Simple notebook execution tests for CI/CD integration.

This module provides basic notebook execution testing using papermill,
designed to be lightweight and integrate easily with GitHub Actions.
"""

import os
import tempfile
from pathlib import Path

import papermill as pm
import pytest

from conftest import get_notebook_files


def get_test_parameters():
    """Get default test parameters for notebook execution."""
    return {
        "files": [],  # Empty files list for basic testing
        "test_mode": True,
        "quick_run": True,
    }


def execute_single_notebook(notebook_path: Path, timeout: int = 300) -> bool:
    """
    Execute a single notebook with papermill.
    
    Args:
        notebook_path: Path to notebook to execute
        timeout: Execution timeout in seconds
        
    Returns:
        True if successful, raises exception if failed
    """
    with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as tmp_file:
        output_path = Path(tmp_file.name)
    
    try:
        pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(output_path),
            parameters=get_test_parameters(),
            timeout=timeout,
            kernel_name="python3"
        )
        return True
    except Exception as e:
        raise Exception(f"Notebook execution failed: {str(e)}") from e
    finally:
        if output_path.exists():
            output_path.unlink()


@pytest.mark.parametrize("notebook_path", get_notebook_files())
def test_notebook_executes_without_error(notebook_path):
    """Test that each notebook executes without errors."""
    
    # Skip backup/duplicate files
    if any(skip in str(notebook_path) for skip in ["copy.ipynb", ".checkpoint"]):
        pytest.skip(f"Skipping backup/checkpoint file: {notebook_path}")
    
    # Execute the notebook
    success = execute_single_notebook(notebook_path, timeout=300)
    assert success, f"Failed to execute notebook: {notebook_path}"


def test_notebooks_directory_exists():
    """Verify the notebooks directory exists and contains files."""
    notebooks = get_notebook_files()
    assert len(notebooks) > 0, "No notebook files found in notebooks directory"
    
    for notebook in notebooks:
        assert notebook.exists(), f"Notebook file does not exist: {notebook}"
        assert notebook.suffix == '.ipynb', f"File is not a notebook: {notebook}"


if __name__ == "__main__":
    # Allow running directly for testing
    import sys
    
    if len(sys.argv) > 1:
        notebook_file = Path(sys.argv[1])
        if notebook_file.exists():
            print(f"Testing {notebook_file}...")
            try:
                execute_single_notebook(notebook_file)
                print("✅ Success!")
            except Exception as e:
                print(f"❌ Failed: {e}")
                sys.exit(1)
    else:
        print("Usage: python test_notebook_execution.py <notebook_path>")
