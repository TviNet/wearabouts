import argparse
import logging
import os

from app.garmin import solve_with_garmin_agent
from constants import ARTIFACT_DIR
from datetime import datetime
from sandbox.notebook import JupyterSandbox

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def solve(task: str):
    # timestamp
    unique_task_id = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # create a new notebook
    with JupyterSandbox() as sandbox:
        logger.info(f"Solving task {task} with garmin agent")
        answer, state_trajectory = solve_with_garmin_agent(
            sandbox, task, unique_task_id
        )

        save_dir = os.path.join(ARTIFACT_DIR, unique_task_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "answer.ipynb")
        logger.info(f"Saving notebook to {save_path}")
        sandbox.save_notebook(answer, save_path)

        traj_save_path = os.path.join(save_dir, "state_trajectory")
        os.makedirs(traj_save_path, exist_ok=True)
        for idx, state in enumerate(state_trajectory):
            sandbox.save_notebook(state, os.path.join(traj_save_path, f"{idx}.ipynb"))


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()
    solve(args.task)

    # solve("Plot my sleep times for last week")
