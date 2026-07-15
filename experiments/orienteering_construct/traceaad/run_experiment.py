from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.tsp_construct.traceaad import run_experiment as runner
from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
from llm4ad.task.optimization.orienteering_construct import OrienteeringEvaluation


runner.TASK = "orienteering_construct"
runner.TASK_SPLIT = "train"
runner.TASK_KWARGS = get_generated_task_kwargs(runner.TASK, runner.TASK_SPLIT)
runner.TSPEvaluation = OrienteeringEvaluation
runner.TIMESTAMP = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
runner.RUN_DIR = Path(__file__).resolve().parent / runner.TIMESTAMP
runner.LOG_DIR = runner.RUN_DIR / "logs"
runner.TMUX_LOG = runner.RUN_DIR / "tmux_run.log"


if __name__ == "__main__":
    runner.main()
