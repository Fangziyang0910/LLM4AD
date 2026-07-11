from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.tsp_construct.traceaad import run_experiment as runner
from llm4ad.task.optimization.cvrp_aco import CVRPACOEvaluation

runner.TASK = "cvrp_aco"
runner.TASK_SPLIT = "train"
runner.TASK_KWARGS = {
    "split": runner.TASK_SPLIT,
    "timeout_seconds": 120,
    "n_ants": 30,
    "n_iterations": 100,
    "aco_seed": 1234,
}
runner.TSPEvaluation = CVRPACOEvaluation
runner.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
runner.RUN_DIR = Path(__file__).resolve().parent / runner.TIMESTAMP
runner.LOG_DIR = runner.RUN_DIR / "logs"
runner.TMUX_LOG = runner.RUN_DIR / "tmux_run.log"


if __name__ == "__main__":
    runner.main()
