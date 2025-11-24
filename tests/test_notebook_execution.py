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


def get_test_parameters(notebook_path: Path = None):
    """Get test parameters for notebook execution, with notebook-specific overrides."""
    
    # Default parameters
    default_params = {
        "files": [],  # specify files to use for the notebook
    }
    
    # Notebook-specific parameters
    if notebook_path:
        # Get the test directory (where this file is located)
        test_dir = Path(__file__).parent
        
        notebook_specific_params = {
            "subset-selection.ipynb": {
                "input_files": [str(test_dir / "assets" / "subset-selection" / "combined_cut_50x.jsonl")]
            },
            # Add more notebook-specific parameters here as needed
            # "document-conversion-standard.ipynb": {
            #     "files": ["https://example.com/sample.pdf"]
            # }
        }
        
        # Return notebook-specific params if available, otherwise default
        return notebook_specific_params.get(notebook_path.name, default_params)
    
    return default_params
def execute_single_notebook(notebook_path: Path) -> bool:
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
        test_params = get_test_parameters(notebook_path)  # Pass notebook_path here
        # Remove 'files' key if it's empty (let notebook use its own files)
        if 'files' in test_params and not test_params['files']:
            test_params = {k: v for k, v in test_params.items() if k != 'files'}
        pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(output_path),
            parameters=test_params
        )
        return True
    except Exception as e:
        raise Exception(f"Notebook execution failed: {str(e)}") from e
    finally:
        if output_path.exists():
            output_path.unlink()


@pytest.mark.parametrize("notebook_path", get_notebook_files(), 
                        ids=lambda path: str(path)) 
def test_notebook_executes_without_error(notebook_path):
    """Test that each notebook executes without errors."""
    
    # Skip backup/duplicate files
    if any(skip in str(notebook_path) for skip in ["copy.ipynb", ".checkpoint"]):
        pytest.skip(f"Skipping backup/checkpoint file: {notebook_path}")
    
    # Execute the notebook
    success = execute_single_notebook(notebook_path)
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
