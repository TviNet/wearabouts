import logging
import nbformat

from enum import Enum
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from typing import Any, Dict

logger = logging.getLogger(__name__)


class CellType(Enum):
    CODE = "code"
    MARKDOWN = "markdown"

    @classmethod
    def _missing_(cls, value):
        return cls.CODE


class JupyterSandbox:
    def __init__(self, kernel_name="python3"):
        self.kernel_name = kernel_name
        self.timeout = 600  # timeout in seconds
        self._executor = ExecutePreprocessor(
            timeout=self.timeout, kernel_name=self.kernel_name
        )
        self._executor.allow_errors = True

    def create_notebook(self) -> nbformat.NotebookNode:
        """Create a new empty notebook"""
        return new_notebook()

    def add_cell(
        self,
        notebook: nbformat.NotebookNode,
        content: str,
        cell_type: CellType,
        idx: int = -1,
    ) -> nbformat.NotebookNode:
        if cell_type == CellType.CODE:
            cell = new_code_cell(content)
        elif cell_type == CellType.MARKDOWN:
            cell = new_markdown_cell(content)
        else:
            raise ValueError(f"Invalid cell type: {cell_type}")

        if idx == -1:
            notebook.cells.append(cell)
        else:
            notebook.cells.insert(idx, cell)
        return notebook

    def modify_cell(
        self, notebook: nbformat.NotebookNode, idx: int, content: str
    ) -> nbformat.NotebookNode:
        if idx < 0 or idx >= len(notebook.cells):
            raise ValueError(f"Invalid cell index: {idx}")
        notebook.cells[idx].source = content
        return notebook

    def delete_cell(
        self, notebook: nbformat.NotebookNode, idx: int
    ) -> nbformat.NotebookNode:
        if idx < 0 or idx >= len(notebook.cells):
            raise ValueError(f"Invalid cell index: {idx}")
        notebook.cells.pop(idx)
        return notebook

    def save_notebook(self, notebook: nbformat.NotebookNode, filepath: str):
        """Save the notebook to a file"""
        with open(filepath, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

    def read_notebook(self, filepath: str) -> nbformat.NotebookNode:
        """Read a notebook from a file"""
        with open(filepath, "r", encoding="utf-8") as f:
            return nbformat.read(f, as_version=4)

    def execute_notebook(self, notebook: nbformat.NotebookNode):
        """Execute all cells in the notebook"""
        ep = ExecutePreprocessor(timeout=self.timeout, kernel_name=self.kernel_name)
        try:
            # Execute the notebook
            # The input argument *nb* is modified in-place.
            executed_notebook, _ = ep.preprocess(notebook)
            return executed_notebook
        except Exception as _:
            return notebook

    def execute_cell(
        self, notebook: nbformat.NotebookNode, cell_index: int
    ) -> Dict[str, Any]:
        """Execute a single cell and return its output

        Args:
            notebook: The notebook containing the cell
            cell_index: Index of the cell to execute

        Returns:
            Dictionary containing execution results:
                success: Whether execution was successful
                output: Cell outputs if successful
                error: Error message if failed
        """
        try:
            # Execute the cell in the existing notebook
            self._executor.preprocess_cell(
                notebook.cells[cell_index], resources={}, cell_index=cell_index
            )

            return {
                "success": True,
                "output": notebook.cells[cell_index].outputs,
                "error": None,
            }
        except Exception as e:
            return {"success": False, "output": None, "error": str(e)}


# Example usage
if __name__ == "__main__":
    # Create a sandbox instance
    sandbox = JupyterSandbox()

    # Create a new notebook
    nb = sandbox.create_notebook()

    # Add some markdown and code cells
    sandbox.add_cell(
        nb,
        "# Test Notebook\nThis is a test notebook created programmatically.",
        CellType.MARKDOWN,
    )

    sandbox.add_cell(
        nb,
        """
import matplotlib.pyplot as plt
import numpy as np
""",
        CellType.CODE,
    )
    # Add a simple code cell
    sandbox.add_cell(
        nb,
        """
# Create some test data
data = np.random.randn(100)
print(f"Mean of data: {data.mean():.2f}")
""",
        CellType.CODE,
    )

    # Add another code cell
    sandbox.add_cell(
        nb,
        """
plt.hist(data, bins=20)
plt.title('Histogram of Random Data')
plt.show()
""",
        CellType.CODE,
    )
    # Execute the notebook
    executed_nb = sandbox.execute_notebook(nb)
    import json

    print(json.dumps(executed_nb, indent=4))

    # Save the executed notebook
    sandbox.save_notebook(executed_nb, "test_notebook.ipynb")

    print("Notebook created and executed successfully!")
