from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.tsp_construct.pathwise import run_experiment as runner
from llm4ad.task.optimization.op_aco import OPACOEvaluation

runner.TASK = "op_aco"
runner.TASK_SPLIT = "train"
runner.TASK_KWARGS = {
    "split": runner.TASK_SPLIT,
    "timeout_seconds": 60,
    "n_ants": 20,
    "n_iterations": 50,
    "aco_seed": 1234,
    "n_workers": int(os.environ.get("OP_EVAL_WORKERS", "5")),
}
runner.TSPEvaluation = OPACOEvaluation
runner.TIMESTAMP = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
runner.RUN_DIR = Path(__file__).resolve().parent / runner.TIMESTAMP
runner.LOG_DIR = runner.RUN_DIR / "logs"
runner.TMUX_LOG = runner.RUN_DIR / "tmux_run.log"


if __name__ == "__main__":
    runner.main()
