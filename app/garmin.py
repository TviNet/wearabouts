import copy
import logging
import nbformat
import os
import requests

from agent.llm import LlmClient, get_llm_client
from agent.prompts import Character, JupyterCodeAgentPrompt
from agent.tools import (
    JupyterCodeActionParser,
    JupyterCodeParser,
    JupyterCritiqueActionsParser,
)
from constants import GARMIN_API_GUIDE_PATH, MAX_GOAL_ITERATIONS, MAX_ITERATIONS
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
)
from garth.exc import GarthHTTPError
from sandbox.notebook import CellType, JupyterSandbox
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init_api():
    """Initialize Garmin API with your credentials."""
    tokenstore = os.getenv("GARMINTOKENSTORE")
    tokenstore_base64 = os.getenv("GARMINTOKENSTORE_BASE64")
    if not tokenstore and not tokenstore_base64:
        raise Exception("GARMINTOKENSTORE or GARMINTOKENSTORE_BASE64 must be set")
    try:
        # Using Oauth1 and OAuth2 token files from directory
        print(
            f"Trying to login to Garmin Connect using token data from directory '{tokenstore}'...\n"
        )

        # Using Oauth1 and Oauth2 tokens from base64 encoded string
        # print(
        #     f"Trying to login to Garmin Connect using token data from file '{tokenstore_base64}'...\n"
        # )
        # dir_path = os.path.expanduser(tokenstore_base64)
        # with open(dir_path, "r") as token_file:
        #     tokenstore = token_file.read()

        garmin = Garmin()
        garmin.login(tokenstore)

    except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError):
        # Session is expired. You'll need to log in again
        print(
            "Login tokens not present, login with your Garmin Connect credentials to generate them.\n"
            f"They will be stored in '{tokenstore}' for future use.\n"
        )
        try:
            email = os.getenv("GARMIN_EMAIL")
            password = os.getenv("GARMIN_PASSWORD")
            # Ask for credentials if not set as environment variables
            if not email or not password:
                raise Exception("GARMIN_EMAIL and GARMIN_PASSWORD must be set")

            garmin = Garmin(email=email, password=password)
            garmin.login()
            # Save Oauth1 and Oauth2 token files to directory for next login
            garmin.garth.dump(tokenstore)
            print(
                f"Oauth tokens stored in '{tokenstore}' directory for future use. (first method)\n"
            )
            # Encode Oauth1 and Oauth2 tokens to base64 string and safe to file for next login (alternative way)
            token_base64 = garmin.garth.dumps()
            dir_path = os.path.expanduser(tokenstore_base64)
            with open(dir_path, "w") as token_file:
                token_file.write(token_base64)
            print(
                f"Oauth tokens encoded as base64 string and saved to '{dir_path}' file for future use. (second method)\n"
            )
        except (
            FileNotFoundError,
            GarthHTTPError,
            GarminConnectAuthenticationError,
            requests.exceptions.HTTPError,
        ) as err:
            logger.error(err)
            return None

    return


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

tokenstore = os.getenv("GARMINTOKENSTORE")
api = Garmin()
api.login(tokenstore)
""",
        cell_type=CellType.CODE,
    )
    # Execute once to login
    notebook = sandbox.execute_notebook(notebook)
    if notebook.cells[1].outputs[0]["output_type"] == "error":
        raise Exception("Login failed. Try again in a few minutes.")
    # assuming this worked
    # Skip the login cell for the next iterations
    notebook = sandbox.skip_cell_execution(notebook, 1)
    return notebook


def current_attempt_towards_goal(
    task: str,
    notebook: nbformat.NotebookNode,
    sandbox: JupyterSandbox,
    prompt_factory: JupyterCodeAgentPrompt,
    llm_client: LlmClient,
    session_id: str,
    state_trajectory: List[nbformat.NotebookNode],
    goal_idx: int,
) -> nbformat.NotebookNode:
    for idx in range(MAX_ITERATIONS):
        logger.info(
            f"[{session_id}]: Goal iteration {goal_idx} - Solver iteration {idx} ..."
        )
        notebook = sandbox.execute_notebook(notebook)
        state_trajectory.append(copy.deepcopy(notebook))
        notebook_state = JupyterCodeParser.render_notebook(notebook)
        llm_prompt = prompt_factory.forward(
            task,
            notebook_state,
            character=Character.GENERATE_CODE,
        )
        actions = llm_client.get_single_answer(llm_prompt)
        updated_notebook, should_stop = JupyterCodeActionParser.response_to_actions(
            actions, sandbox=sandbox, notebook=notebook
        )
        notebook = updated_notebook
        if should_stop:
            break
    return notebook


def solve_with_garmin_agent(
    sandbox: JupyterSandbox, task: str, unique_task_id: str
) -> nbformat.NotebookNode:
    init_api()
    prompt_factory = get_prompt_factory()
    session_id = f"garmin_agent_{unique_task_id}"
    llm_client = get_llm_client(session_id=session_id)
    logger.info(f"[{session_id}]: Solver started")
    notebook = init_notebook(sandbox, task)
    state_trajectory = []

    try:
        goal = task
        for goal_idx in range(MAX_GOAL_ITERATIONS):
            notebook = current_attempt_towards_goal(
                goal,
                notebook,
                sandbox,
                prompt_factory,
                llm_client,
                session_id,
                state_trajectory,
                goal_idx,
            )
            notebook_state = JupyterCodeParser.render_notebook(notebook)
            llm_prompt = prompt_factory.forward(
                task,
                notebook_state,
                character=Character.CRITIQUE_CODE,
            )
            actions = llm_client.get_single_answer(llm_prompt)
            feedback, should_stop = JupyterCritiqueActionsParser.response_to_actions(
                actions
            )
            if should_stop:
                break
            logger.info(
                f"[{session_id}]: Updating task with new goal - Feedback: {feedback}"
            )
            goal = task + f"\n(feedback: {feedback})\n"
    except Exception as e:
        logger.error(f"Error solving task {task} with garmin agent: {e}")
    finally:
        return notebook, state_trajectory


if __name__ == "__main__":
    init_api()
