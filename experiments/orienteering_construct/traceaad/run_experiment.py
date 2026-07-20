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

# zhong-server
runner.BASE_URL = "http://183.36.243.124:9000/v1"
runner.API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
runner.MODEL = "Qwen3.6-27B-Q4_K_M"
runner.NO_PROXY_HOSTS = "183.36.243.124,localhost,127.0.0.1,::1"
os.environ["NO_PROXY"] = runner.NO_PROXY_HOSTS
os.environ["no_proxy"] = runner.NO_PROXY_HOSTS

# 与 version1 / 训练集 seed 对齐
runner.SEARCH_SEED = int(os.environ.get("SEARCH_SEED", "2024"))

runner.RESUME_FROM = os.environ.get("RESUME_FROM", "").strip() or None
runner.EXPERIMENT_VERSION = os.environ.get("EXPERIMENT_VERSION", "version2").strip()
if runner.RESUME_FROM:
    runner.RUN_DIR = Path(runner.RESUME_FROM).resolve()
    runner.TIMESTAMP = runner.RUN_DIR.name
else:
    runner.TIMESTAMP = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    method_root = Path(__file__).resolve().parent
    runner.RUN_DIR = (
        method_root / runner.EXPERIMENT_VERSION / runner.TIMESTAMP
        if runner.EXPERIMENT_VERSION
        else method_root / runner.TIMESTAMP
    )
runner.LOG_DIR = runner.RUN_DIR / "logs"
runner.TMUX_LOG = runner.RUN_DIR / "tmux_run.log"


if __name__ == "__main__":
    runner.main()
