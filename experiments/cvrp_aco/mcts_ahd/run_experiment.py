from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.tsp_construct.mcts_ahd import run_experiment as runner
from llm4ad.task.optimization.cvrp_aco import CVRPACOEvaluation
from llm4ad.tools.env import resolve_llm_api_key

runner.TASK = "cvrp_aco"
runner.TASK_SPLIT = "train"
runner.TASK_KWARGS = {
    "split": runner.TASK_SPLIT,
    "timeout_seconds": 120,
    "n_ants": 30,
    "n_iterations": 100,
    "aco_seed": 1234,
    # 训练评估：10 个 train 实例并行跑 ACO；可用 CVRP_EVAL_WORKERS 覆盖
    "n_workers": int(os.environ.get("CVRP_EVAL_WORKERS", "10")),
}
runner.TSPEvaluation = CVRPACOEvaluation
runner.TIMESTAMP = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
runner.RUN_DIR = Path(__file__).resolve().parent / runner.TIMESTAMP
runner.LOG_DIR = runner.RUN_DIR / "logs"
runner.TMUX_LOG = runner.RUN_DIR / "tmux_run.log"

NO_PROXY_HOSTS = os.environ.get(
    "NO_PROXY",
    "183.36.243.124,222.201.145.8,localhost,127.0.0.1,::1",
)
os.environ.setdefault("NO_PROXY", NO_PROXY_HOSTS)
os.environ.setdefault("no_proxy", NO_PROXY_HOSTS)
runner.BASE_URL = os.environ.get("LLM_BASE_URL", runner.BASE_URL)
runner.MODEL = os.environ.get("LLM_MODEL", runner.MODEL)
runner.API_KEY = os.environ.get("LLM_API_KEY") or resolve_llm_api_key(base_url=runner.BASE_URL)


if __name__ == "__main__":
    runner.main()
