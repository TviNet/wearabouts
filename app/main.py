import logging
import os
import uuid

from agent.llm import get_llm_client
from agent.prompts import JupyterCodeAgentPrompt
from agent.tools import JupyterCodeActionParser, JupyterCodeParser
from constants import ARTIFACT_DIR, GARMIN_API_GUIDE_PATH, MAX_ITERATIONS
from sandbox.notebook_sandbox import JupyterSandbox

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


def solve_with_garmin_agent(sandbox: JupyterSandbox, task: str):
    notebook = sandbox.create_notebook()

    prompt_factory = get_prompt_factory()
    llm_client = get_llm_client()

    for _ in range(MAX_ITERATIONS):
        notebook_state = JupyterCodeParser.convert_notebook_to_state(notebook)
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
