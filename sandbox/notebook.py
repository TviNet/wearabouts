import logging
import nbformat

from enum import Enum
from jupyter_client import KernelManager
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


class SkipCellExecutePreprocessor(ExecutePreprocessor):
    """
    https://stackoverflow.com/a/38064506
    """

    def preprocess_cell(self, cell, resources, cell_index):
        """
        Executes a single code cell. See base.py for details.
        To execute all cells see :meth:`preprocess`.

        Checks cell.metadata for 'execute' key. If set, and maps to False,
          the cell is not executed.
        """

        if not cell.metadata.get("execute", True):
            # Don't execute this cell in output
            return cell, resources

        return super().preprocess_cell(cell, resources, cell_index)


class JupyterSandbox:
    def __init__(self, kernel_name="python3"):
        self.kernel_name = kernel_name
        self.timeout = 600  # timeout in seconds

        # Initialize kernel manager
        self._kernel_manager = KernelManager(kernel_name=self.kernel_name)
        self._kernel_manager.start_kernel()
        self._kernel_client = self._kernel_manager.client()
        self._kernel_client.start_channels()

        self._executor = SkipCellExecutePreprocessor(
            timeout=self.timeout,
            kernel_name=self.kernel_name,
            kernel_manager=self._kernel_manager,
        )
        self._executor.allow_errors = False

    def shutdown(self):
        """Properly shutdown the kernel"""
        if self._kernel_client:
            self._kernel_client.stop_channels()
        if self._kernel_manager:
            self._kernel_manager.shutdown_kernel()

    def __del__(self):
        """Ensure kernel is shutdown when object is deleted"""
        self.shutdown()

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

    def skip_cell_execution(
        self, notebook: nbformat.NotebookNode, idx: int
    ) -> nbformat.NotebookNode:
        if idx < 0 or idx >= len(notebook.cells):
            raise ValueError(f"Invalid cell index: {idx}")
        notebook.cells[idx].metadata["execute"] = False
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
        try:
            # Execute the notebook
            # The input argument *nb* is modified in-place.
            executed_notebook, _ = self._executor.preprocess(
                notebook, km=self._kernel_manager
            )
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
    sandbox.add_cell(
        nb,
        """
# skipped cell
print("This should not be printed")
""",
        CellType.CODE,
    )
    sandbox.skip_cell_execution(nb, 2)
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
