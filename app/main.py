import logging
import nbformat
import os
import uuid

from agent.llm import get_llm_client
from agent.prompts import JupyterCodeAgentPrompt
from agent.tools import JupyterCodeActionParser, JupyterCodeParser
from constants import ARTIFACT_DIR, GARMIN_API_GUIDE_PATH, MAX_ITERATIONS
from datetime import datetime
from sandbox.notebook import CellType, JupyterSandbox

logger = logging.getLogger(__name__)


def get_prompt_factory() -> JupyterCodeAgentPrompt:
    with open(GARMIN_API_GUIDE_PATH, "r") as f:
        api_guide = f.read()

    prompt_factory = JupyterCodeAgentPrompt(
        ADDITIONAL_SYSTEM_PROMPT=f"""
The following is the API guide for the Garmin Connect API:
{api_guide}
"""
    )

    return prompt_factory


def init_notebook(sandbox: JupyterSandbox, task: str) -> nbformat.NotebookNode:
    notebook = sandbox.create_notebook()
    sandbox.add_cell(notebook, content=f"{task}", cell_type=CellType.MARKDOWN)
    sandbox.add_cell(
        notebook,
        content="""
import os
from garminconnect import Garmin
api = Garmin(os.getenv("GARMIN_EMAIL"), os.getenv("GARMIN_PASSWORD"))
api.login()
""",
        cell_type=CellType.CODE,
    )
    return notebook


def solve_with_garmin_agent(
    sandbox: JupyterSandbox, task: str
) -> nbformat.NotebookNode:
    prompt_factory = get_prompt_factory()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_id = f"garmin_agent_{current_time}"
    llm_client = get_llm_client(session_id=session_id)
    logger.info(f"[{session_id}]: Solver started")
    notebook = init_notebook(sandbox, task)

    for idx in range(MAX_ITERATIONS):
        logger.info(f"[{session_id}]: Solver iteration {idx} ...")
        notebook = sandbox.execute_notebook(notebook)
        notebook_state = JupyterCodeParser.render_notebook(notebook)
        llm_prompt = prompt_factory.forward(
            task,
            notebook_state,
        )
        actions = llm_client.get_single_answer(llm_prompt)
        updated_notebook, should_stop = JupyterCodeActionParser.response_to_actions(
            actions, sandbox=sandbox, notebook=notebook
        )
        notebook = updated_notebook
        if should_stop:
            break

    notebook = sandbox.execute_notebook(notebook)
    return notebook


def solve(task: str):
    unique_task_id = str(uuid.uuid4())
    # create a new notebook
    sandbox = JupyterSandbox()

    logger.info(f"Solving task {task} with garmin agent")
    notebook = solve_with_garmin_agent(sandbox, task)

    save_path = os.path.join(ARTIFACT_DIR, f"{unique_task_id}.ipynb")
    logger.info(f"Saving notebook to {save_path}")
    sandbox.save_notebook(notebook, save_path)


if __name__ == "__main__":
    solve("Plot my sleep times for last week")
