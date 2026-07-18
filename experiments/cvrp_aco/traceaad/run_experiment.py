from __future__ import annotations

import os
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
    # 训练评估：10 个 train 实例并行跑 ACO；可用 CVRP_EVAL_WORKERS 覆盖
    "n_workers": int(os.environ.get("CVRP_EVAL_WORKERS", "10")),
}
runner.TSPEvaluation = CVRPACOEvaluation

# 本机 llama.cpp（qwen3.6-27b，-np 3）
runner.BASE_URL = "http://127.0.0.1:8001/v1"
runner.API_KEY = "EMPTY"
runner.MODEL = "Qwen3.6-27B"
runner.NO_PROXY_HOSTS = "127.0.0.1,localhost,::1"
os.environ["NO_PROXY"] = runner.NO_PROXY_HOSTS
os.environ["no_proxy"] = runner.NO_PROXY_HOSTS

# 与上一轮正式实验对齐：训练/ACO 相关固定 1234；搜索 seed 也固定为 1234
runner.SEARCH_SEED = int(os.environ.get("SEARCH_SEED", "1234"))

# RESUME_FROM 优先；否则新建 timestamp 目录（可被 RUN_TIMESTAMP 覆盖）
runner.RESUME_FROM = os.environ.get("RESUME_FROM", "").strip() or None
if runner.RESUME_FROM:
    runner.RUN_DIR = Path(runner.RESUME_FROM).resolve()
    runner.TIMESTAMP = runner.RUN_DIR.name
else:
    runner.TIMESTAMP = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.RUN_DIR = Path(__file__).resolve().parent / runner.TIMESTAMP
runner.LOG_DIR = runner.RUN_DIR / "logs"
runner.TMUX_LOG = runner.RUN_DIR / "tmux_run.log"


if __name__ == "__main__":
    runner.main()
