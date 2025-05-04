import copy
import json
import logging
import nbformat
import os
import signal
import time

from abc import abstractmethod
from agent.llm import get_llm_client
from agent.prompts import Character, JupyterCodeAgentPrompt
from agent.tools import (
    JupyterCodeActionParser,
    JupyterCodeParser,
    JupyterCritiqueActionsParser,
)
from constants import (
    ARTIFACT_DIR,
    GARMIN_API_GUIDE_PATH,
    MAX_GOAL_ITERATIONS,
    MAX_ITERATIONS,
)
from dataclasses import dataclass
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
)
from garth.exc import GarthHTTPError
from sandbox.notebook import CellType, JupyterSandbox
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Solver:
    task: str
    task_id: str
    feedback: str = ""

    def _get_save_dir(self):
        return os.path.join(ARTIFACT_DIR, self.task_id)

    def _get_last_state_path(self):
        return os.path.join(self._get_save_dir(), "last.ipynb")

    def _get_metadata_path(self):
        return os.path.join(self._get_save_dir(), "metadata.json")

    def _get_state_trajectory_dir(self):
        return os.path.join(self._get_save_dir(), "state_trajectory")

    def _get_state_trajectory_path(self, idx: int):
        return os.path.join(self._get_state_trajectory_dir(), f"{idx}.ipynb")

    def init_solver(self):
        self.session_id = f"garmin_agent_{self.task_id}"
        self.save_dir = os.path.join(ARTIFACT_DIR, self.session_id)

        # Initialize components
        self.init_api()
        self.prompt_factory = self._get_prompt_factory()
        self.llm_client = get_llm_client(session_id=self.session_id)

        if os.path.exists(self.save_dir):
            self._load_state()
        else:
            # State
            self.notebook: Optional[nbformat.NotebookNode] = None
            self.state_trajectory: List[nbformat.NotebookNode] = []
            self.total_iterations = 0

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"[{self.session_id}]: Solver initialized")

    def _load_state(self):
        self.notebook = sandbox.load_notebook(self._get_last_state_path())
        with open(self._get_metadata_path(), "r") as f:
            metadata = json.load(f)
            self.total_iterations = metadata["total_iterations"]
            if metadata["total_states"] != metadata["total_iterations"]:
                raise Exception(
                    f"Total states {metadata['total_states']} != total iterations {metadata['total_iterations']}"
                )
        self.state_trajectory = [
            sandbox.load_notebook(self._get_state_trajectory_path(idx))
            for idx in range(metadata["total_states"])
        ]

    def _save_state(self):
        """Save current solver state."""
        os.makedirs(self.save_dir, exist_ok=True)
        notebook_save_path = self._get_last_state_path()

        # Save current notebook
        with open(notebook_save_path, "w", encoding="utf-8") as f:
            nbformat.write(self.notebook, f)

        # Save trajectory
        traj_save_path = self._get_state_trajectory_dir()
        os.makedirs(traj_save_path, exist_ok=True)

        # Save new states
        state_save_path = self._get_state_trajectory_path(self.total_iterations)
        with open(state_save_path, "w", encoding="utf-8") as f:
            nbformat.write(self.state_trajectory[-1], f)

        # Save metadata
        metadata = {
            "last_save": time.time(),
            "task_id": self.task_id,
            "total_states": len(self.state_trajectory),
            "total_iterations": self.total_iterations,
        }
        with open(self._get_metadata_path(), "w") as f:
            json.dump(metadata, f)

    def _signal_handler(self, signum, frame):
        """Handle interruption by saving current state."""
        logger.info(f"[{self.session_id}]: Received interrupt signal, saving state...")
        self._save_state()
        logger.info(f"[{self.session_id}]: State saved, exiting...")
        raise KeyboardInterrupt

    def _current_attempt_towards_goal(
        self,
        sandbox: JupyterSandbox,
        goal_idx: int,
    ) -> Tuple[nbformat.NotebookNode, bool]:
        should_stop_goal = False
        for idx in range(MAX_ITERATIONS):
            self.total_iterations += 1
            logger.info(
                f"[{self.session_id}]: Goal iteration {goal_idx} - Solver iteration {idx} (Total: {self.total_iterations}) ..."
            )

            self.notebook = sandbox.execute_notebook(self.notebook)
            self.state_trajectory.append(copy.deepcopy(self.notebook))

            logger.info(
                f"[{self.session_id}]: Saving state at iteration {self.total_iterations}..."
            )
            self._save_state()

            notebook_state = JupyterCodeParser.render_notebook(self.notebook)
            llm_prompt = self.prompt_factory.forward(
                self.task,
                notebook_state,
                character=Character.GENERATE_CODE,
            )
            actions = self.llm_client.get_single_answer(llm_prompt)
            self.notebook, should_stop = JupyterCodeActionParser.response_to_actions(
                actions, sandbox=sandbox, notebook=self.notebook
            )
            if should_stop:
                should_stop_goal = True
                break

        return should_stop_goal

    def solve(self, sandbox: JupyterSandbox) -> nbformat.NotebookNode:
        """Main solving loop."""
        try:
            # Initialize or load notebook
            self.notebook = self._init_notebook(sandbox)
            goal = self.task

            for goal_idx in range(MAX_GOAL_ITERATIONS):
                should_stop = self._current_attempt_towards_goal(sandbox, goal_idx)
                if should_stop:
                    break

                # Get feedback and update goal
                notebook_state = JupyterCodeParser.render_notebook(self.notebook)
                llm_prompt = self.prompt_factory.forward(
                    self.task,
                    notebook_state,
                    character=Character.CRITIQUE_CODE,
                )
                actions = self.llm_client.get_single_answer(llm_prompt)
                feedback, should_stop = (
                    JupyterCritiqueActionsParser.response_to_actions(actions)
                )
                if should_stop:
                    break

                logger.info(
                    f"[{self.session_id}]: Updating task with new goal - Feedback: {feedback}"
                )
                goal = self._combine_task_with_feedback(goal, feedback)

        except Exception as e:
            logger.error(f"Error solving task {self.task} with garmin agent: {e}")
            self._save_state()
            raise
        finally:
            self._save_state()

        return self.notebook

    def _combine_task_with_feedback(self, task: str, feedback: str) -> str:
        return f"{task}\n(feedback: {feedback})\n"

    @abstractmethod
    def _init_notebook(self, sandbox: JupyterSandbox) -> nbformat.NotebookNode:
        pass

    @abstractmethod
    def _get_prompt_factory(self) -> JupyterCodeAgentPrompt:
        pass

    @abstractmethod
    def init_api(self):
        pass


class GarminSolver(Solver):
    def _get_prompt_factory(self) -> JupyterCodeAgentPrompt:
        with open(GARMIN_API_GUIDE_PATH, "r") as f:
            api_guide = f.read()

        return JupyterCodeAgentPrompt(
            ADDITIONAL_SYSTEM_PROMPT=f"""
The following is the API guide for the Garmin Connect API:
{api_guide}
"""
        )

    def init_api(self):
        """Initialize Garmin API with credentials."""
        tokenstore = os.getenv("GARMINTOKENSTORE")
        tokenstore_base64 = os.getenv("GARMINTOKENSTORE_BASE64")
        if not tokenstore and not tokenstore_base64:
            raise Exception("GARMINTOKENSTORE or GARMINTOKENSTORE_BASE64 must be set")
        try:
            print(
                f"Trying to login to Garmin Connect using token data from directory '{tokenstore}'...\n"
            )
            garmin = Garmin()
            garmin.login(tokenstore)
        except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError):
            email = os.getenv("GARMIN_EMAIL")
            password = os.getenv("GARMIN_PASSWORD")
            if not email or not password:
                raise Exception("GARMIN_EMAIL and GARMIN_PASSWORD must be set")
            garmin = Garmin(email=email, password=password)
            garmin.login()
            garmin.garth.dump(tokenstore)

    def _init_notebook(self, sandbox: JupyterSandbox) -> nbformat.NotebookNode:
        notebook = sandbox.create_notebook()
        sandbox.add_cell(notebook, content=f"{self.task}", cell_type=CellType.MARKDOWN)
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
        # Skip the login cell for next iterations
        notebook = sandbox.skip_cell_execution(notebook, 1)
        return notebook


if __name__ == "__main__":
    with JupyterSandbox() as sandbox:
        solver = GarminSolver(task="Plot my sleep times for last week", task_id="test")
        solver.solve(sandbox)
