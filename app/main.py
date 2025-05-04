import argparse
import logging

from app.garmin import GarminSolver
from datetime import datetime
from sandbox.notebook import JupyterSandbox

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def solve(task: str, task_id: str, feedback: str):
    # timestamp
    unique_task_id = task_id or str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    with JupyterSandbox() as sandbox:
        logger.info(f"Solving task {task} with garmin agent")
        solver = GarminSolver(task=task, task_id=unique_task_id, feedback=feedback)
        solver.init_solver()
        solver.solve(sandbox)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--task-id", type=str, required=False)
    parser.add_argument("--feedback", type=str, required=False)
    args = parser.parse_args()
    solve(args.task, args.task_id, args.feedback or "")

    # solve("Plot my sleep times for last week")
