from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.method.traceaad import (
    PortfolioWeights,
    TraceAAD,
    TraceAADProfiler,
    ValueWeights,
    resume_traceaad,
)
from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI


TASK = "online_bin_packing"
METHOD = "traceaad"
RESUME_FROM = os.environ.get("RESUME_FROM", "").strip() or None
EXPERIMENT_VERSION = os.environ.get("EXPERIMENT_VERSION", "version2").strip()
if RESUME_FROM:
    RUN_DIR = Path(RESUME_FROM).resolve()
    TIMESTAMP = RUN_DIR.name
else:
    TIMESTAMP = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    method_root = Path(__file__).resolve().parent
    RUN_DIR = (method_root / EXPERIMENT_VERSION / TIMESTAMP) if EXPERIMENT_VERSION else (method_root / TIMESTAMP)
LOG_DIR = RUN_DIR / "logs"
TMUX_LOG = RUN_DIR / "tmux_run.log"

BASE_URL = os.environ.get("LLM_BASE_URL", "http://222.201.145.8:8080/v1")
API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
MODEL = os.environ.get("LLM_MODEL", "qwen3.6-27b-awq")
LLM_TIMEOUT = 600
MAX_TOKENS = 16384
LLM_TEMPERATURE = 1.0
NO_PROXY_HOSTS = os.environ.get(
    "NO_PROXY",
    "183.36.243.124,222.201.145.8,localhost,127.0.0.1,::1",
)
os.environ.setdefault("NO_PROXY", NO_PROXY_HOSTS)
os.environ.setdefault("no_proxy", NO_PROXY_HOSTS)

TASK_SPLIT = "train"
TASK_KWARGS = get_generated_task_kwargs(TASK, TASK_SPLIT)  # seed=2024

MAX_SAMPLE_NUMS = 1000
N_INIT = 4
ACTIONS_PER_ITERATION = 2
MAX_TRAJECTORY_LENGTH = 8
MAX_ACTIVE_TRAJECTORIES = 160
SAMPLING_STRATEGY = "trajectory_ucb"
TOP_K = 12
TRAJECTORY_TEMPERATURE = 0.8
N_ISLANDS = 4
MAX_PER_ISLAND = 40
NOVELTY_THRESHOLD = 0.92
MIGRATION_INTERVAL = 20
NUM_EVALUATORS = 4
MAX_CONSECUTIVE_SAMPLE_FAILURES = 20
MAX_STALLED_ITERATIONS = 20
SEARCH_SEED = int(os.environ.get("SEARCH_SEED", "2024"))
EVAL_EXECUTOR = "thread"
DEBUG = False

VALUE_WEIGHTS = ValueWeights(
    w_quality=0.42,
    w_potential=0.18,
    w_diversity=0.12,
    w_novelty=0.12,
    w_compactness=0.08,
    w_speed=0.08,
    w_sim_code=0.7,
    w_sim_trajectory=0.3,
    top_k=TOP_K,
    temperature=TRAJECTORY_TEMPERATURE,
)
PORTFOLIO_WEIGHTS = PortfolioWeights()


def write_run_config(*, resumed_from: str | None = None) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(RUN_DIR),
        "task": TASK,
        "method": METHOD,
        "timestamp": TIMESTAMP,
        "experiment_version": EXPERIMENT_VERSION or None,
        "resume_from": resumed_from,
        "llm": {
            "base_url": BASE_URL,
            "api_key": API_KEY,
            "model": MODEL,
            "timeout": LLM_TIMEOUT,
            "max_tokens": MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "enable_thinking": False,
            "no_proxy": NO_PROXY_HOSTS,
        },
        "task_eval": {"split": TASK_SPLIT, **TASK_KWARGS},
        "method_params": {
            "max_sample_nums": MAX_SAMPLE_NUMS,
            "n_init": N_INIT,
            "actions_per_iteration": ACTIONS_PER_ITERATION,
            "max_trajectory_length": MAX_TRAJECTORY_LENGTH,
            "max_active_trajectories": MAX_ACTIVE_TRAJECTORIES,
            "sampling_strategy": SAMPLING_STRATEGY,
            "top_k": TOP_K,
            "trajectory_temperature": TRAJECTORY_TEMPERATURE,
            "n_islands": N_ISLANDS,
            "max_per_island": MAX_PER_ISLAND,
            "novelty_threshold": NOVELTY_THRESHOLD,
            "migration_interval": MIGRATION_INTERVAL,
            "num_evaluators": NUM_EVALUATORS,
            "max_consecutive_sample_failures": MAX_CONSECUTIVE_SAMPLE_FAILURES,
            "max_stalled_iterations": MAX_STALLED_ITERATIONS,
            "random_seed": SEARCH_SEED,
            "eval_executor": EVAL_EXECUTOR,
            "debug": DEBUG,
            "value_weights": asdict(VALUE_WEIGHTS),
            "portfolio_weights": asdict(PORTFOLIO_WEIGHTS),
        },
    }
    (RUN_DIR / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_method() -> TraceAAD:
    llm = OpenAIAPI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        timeout=LLM_TIMEOUT,
        max_tokens=MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        enable_thinking=False,
    )
    task = OBPEvaluation(**TASK_KWARGS)
    return TraceAAD(
        llm=llm,
        evaluation=task,
        profiler=TraceAADProfiler(log_dir=str(LOG_DIR), log_style="complex", create_random_path=False),
        max_sample_nums=MAX_SAMPLE_NUMS,
        n_init=N_INIT,
        actions_per_iteration=ACTIONS_PER_ITERATION,
        max_trajectory_length=MAX_TRAJECTORY_LENGTH,
        max_active_trajectories=MAX_ACTIVE_TRAJECTORIES,
        n_islands=N_ISLANDS,
        max_per_island=MAX_PER_ISLAND,
        sampling_strategy=SAMPLING_STRATEGY,
        novelty_threshold=NOVELTY_THRESHOLD,
        migration_interval=MIGRATION_INTERVAL,
        value_weights=VALUE_WEIGHTS,
        portfolio_weights=PORTFOLIO_WEIGHTS,
        num_evaluators=NUM_EVALUATORS,
        max_consecutive_sample_failures=MAX_CONSECUTIVE_SAMPLE_FAILURES,
        max_stalled_iterations=MAX_STALLED_ITERATIONS,
        random_seed=SEARCH_SEED,
        multi_thread_or_process_eval=EVAL_EXECUTOR,
        debug_mode=DEBUG,
    )


def main() -> None:
    os.environ["NO_PROXY"] = NO_PROXY_HOSTS
    os.environ["no_proxy"] = NO_PROXY_HOSTS
    resume_from = RESUME_FROM
    if resume_from:
        if not RUN_DIR.is_dir():
            raise FileNotFoundError(f"RESUME_FROM directory does not exist: {RUN_DIR}")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    else:
        RUN_DIR.mkdir(parents=True, exist_ok=False)
    write_run_config(resumed_from=resume_from)
    print(f"run_dir={RUN_DIR}")
    with TMUX_LOG.open("a", encoding="utf-8", buffering=1) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"run_dir={RUN_DIR}", flush=True)
            print(f"log_dir={LOG_DIR}", flush=True)
            print(f"llm={MODEL} @ {BASE_URL}", flush=True)
            print(f"task_seed={TASK_KWARGS.get('seed')} search_seed={SEARCH_SEED}", flush=True)
            method = build_method()
            if resume_from:
                print(f"resume_from={resume_from}", flush=True)
                resume_traceaad(method, LOG_DIR)
            method.run()


if __name__ == "__main__":
    main()
