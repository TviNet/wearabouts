from agent.models import ErrorOutput, StreamOutput
from sandbox.notebook import CellType, JupyterSandbox


def test_notebook_without_errors():
    sandbox = JupyterSandbox()
    nb = sandbox.create_notebook()
    sandbox.add_cell(
        nb,
        """
print("Hello, world!")
""",
        CellType.CODE,
    )
    executed_nb = sandbox.execute_notebook(nb)
    stream_output: StreamOutput = executed_nb.cells[0].outputs[0]
    assert stream_output["output_type"] == "stream"
    assert stream_output["name"] == "stdout"
    assert stream_output["text"] == "Hello, world!\n"


def test_notebook_with_errors():
    sandbox = JupyterSandbox()
    nb = sandbox.create_notebook()
    sandbox.add_cell(nb, "print(1/0)", CellType.CODE)
    executed_nb = sandbox.execute_notebook(nb)
    error_output: ErrorOutput = executed_nb.cells[0].outputs[0]
    assert error_output["output_type"] == "error"
    assert error_output["ename"] == "ZeroDivisionError"
    assert error_output["evalue"] == "division by zero"


def test_notebook_state_across_executions():
    sandbox = JupyterSandbox()
    nb = sandbox.create_notebook()
    sandbox.add_cell(nb, "x = 5", CellType.CODE)
    nb = sandbox.execute_notebook(nb)
    sandbox.skip_cell_execution(nb, 0)
    sandbox.add_cell(nb, "print(x)", CellType.CODE)
    nb = sandbox.execute_notebook(nb)
    answer_output = nb.cells[1].outputs[0]
    assert answer_output["output_type"] == "stream"
    assert answer_output["name"] == "stdout"
    assert answer_output["text"] == "5\n"
