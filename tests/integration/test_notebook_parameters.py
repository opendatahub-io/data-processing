"""
Test notebook parameters cell validation.

This module tests that all notebooks have the required parameters cell
that is needed for papermill execution.
"""

from pathlib import Path

import nbformat
import pytest

from conftest import get_notebook_files


class NotebookParametersValidator:
    """Validator for notebook parameters cells."""
    
    def validate_parameters_cell(self, notebook_path: Path) -> bool:
        """
        Validate that a notebook has at least one code cell tagged with 'parameters'.
        
        Args:
            notebook_path: Path to the notebook file
            
        Returns:
            True if notebook has parameters cell, False otherwise
            
        Raises:
            Exception: If notebook cannot be read or validated
        """
        try:
            # Read notebook with no conversion to preserve original structure
            notebook = nbformat.read(notebook_path, nbformat.NO_CONVERT)
            
            # Validate the notebook format
            nbformat.validate(notebook)
            
            # Check for parameters cell
            has_parameters_cell = False
            
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    # Check for code cells tagged with 'parameters'
                    if ('tags' in cell.metadata and 
                        'parameters' in cell.metadata.tags):
                        has_parameters_cell = True
                        break
            
            return has_parameters_cell
            
        except Exception as e:
            raise Exception(f"Failed to validate notebook {notebook_path}: {str(e)}") from e


@pytest.mark.parametrize("notebook_path", get_notebook_files())
def test_notebook_has_parameters_cell(notebook_path):
    """
    Test that each notebook has at least one code cell tagged with 'parameters'.
    
    This is required for papermill execution in the CI/CD pipeline.
    """
    validator = NotebookParametersValidator()
    
    has_parameters = validator.validate_parameters_cell(notebook_path)
    
    assert has_parameters, (
        f"Notebook '{notebook_path}' does not have any code cell tagged with 'parameters'. "
        f"Please add a code cell with metadata tag 'parameters' for papermill execution."
    )


def test_validator_itself():
    """Test the validator logic with a mock notebook structure."""
    # This tests the validator class itself to ensure it works correctly
    validator = NotebookParametersValidator()
    
    # Create a simple test notebook structure
    test_notebook = nbformat.v4.new_notebook()
    
    # Add a regular code cell
    code_cell = nbformat.v4.new_code_cell("x = 1")
    test_notebook.cells.append(code_cell)
    
    # Should fail - no parameters cell yet
    test_path = Path("test_notebook_no_params.ipynb")
    with open(test_path, 'w') as f:
        nbformat.write(test_notebook, f)
    
    try:
        assert not validator.validate_parameters_cell(test_path)
    finally:
        test_path.unlink()  # Clean up
    
    # Add a parameters cell
    params_cell = nbformat.v4.new_code_cell("# Parameters cell\nfiles = []")
    params_cell.metadata["tags"] = ["parameters"]
    test_notebook.cells.append(params_cell)
    
    # Should pass - has parameters cell
    with open(test_path, 'w') as f:
        nbformat.write(test_notebook, f)
    
    try:
        assert validator.validate_parameters_cell(test_path)
    finally:
        test_path.unlink()  # Clean up
