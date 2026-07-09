from __future__ import annotations

import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.method.traceaad import TraceAAD, TraceAADProfiler
from llm4ad.task.optimization.main.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.vllm_openai_api import VLLMOpenAIAPI


TASK = "tsp_construct"
METHOD = "traceaad"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path(__file__).resolve().parent / TIMESTAMP
LOG_DIR = RUN_DIR / "logs"
TMUX_LOG = RUN_DIR / "tmux_run.log"

BASE_URL = "http://222.201.145.8:8080/v1"
API_KEY = "EMPTY"
MODEL = "qwen3.6-27b-awq"
LLM_TIMEOUT = 600
MAX_TOKENS = 16384
LLM_TEMPERATURE = 1.0

SPLIT = "train"
TASK_TIMEOUT_SECONDS = 20
EVAL_WORKERS = 16
EVAL_BACKEND = "process"

MAX_SAMPLE_NUMS = 1000
N_INIT = 4
N_ITERATIONS = None
ACTIONS_PER_ITERATION = 2
MAX_ACTIONS_IN_PROMPT = 5
MAX_TRAJECTORY_LENGTH = 8
MAX_ACTIVE_TRAJECTORIES = 1000
SAMPLING_STRATEGY = "trajectory_ucb"
TOP_K = 5
TRAJECTORY_TEMPERATURE = 0.8
W_END = 0.45
W_PATH = 0.55
W_CONSISTENCY = 0.25
W_DOWNSIDE = 0.5
DISCOUNT = 0.8
POSITIVE_THRESHOLD = 1e-6
C0 = 0.4
NUM_EVALUATORS = 4
MAX_CONSECUTIVE_SAMPLE_FAILURES = 20
EVAL_EXECUTOR = "thread"
DEBUG = False


def write_run_config() -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(RUN_DIR),
        "task": TASK,
        "method": METHOD,
        "timestamp": TIMESTAMP,
        "llm": {
            "base_url": BASE_URL,
            "api_key": API_KEY,
            "model": MODEL,
            "timeout": LLM_TIMEOUT,
            "max_tokens": MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "enable_thinking": False,
        },
        "task_eval": {
            "split": SPLIT,
            "timeout_seconds": TASK_TIMEOUT_SECONDS,
            "eval_workers": EVAL_WORKERS,
            "eval_backend": EVAL_BACKEND,
        },
        "method_params": {
            "max_sample_nums": MAX_SAMPLE_NUMS,
            "n_init": N_INIT,
            "n_iterations": N_ITERATIONS,
            "actions_per_iteration": ACTIONS_PER_ITERATION,
            "max_actions_in_prompt": MAX_ACTIONS_IN_PROMPT,
            "max_trajectory_length": MAX_TRAJECTORY_LENGTH,
            "max_active_trajectories": MAX_ACTIVE_TRAJECTORIES,
            "sampling_strategy": SAMPLING_STRATEGY,
            "top_k": TOP_K,
            "temperature": TRAJECTORY_TEMPERATURE,
            "w_end": W_END,
            "w_path": W_PATH,
            "w_consistency": W_CONSISTENCY,
            "w_downside": W_DOWNSIDE,
            "discount": DISCOUNT,
            "positive_threshold": POSITIVE_THRESHOLD,
            "c0": C0,
            "num_evaluators": NUM_EVALUATORS,
            "max_consecutive_sample_failures": MAX_CONSECUTIVE_SAMPLE_FAILURES,
            "eval_executor": EVAL_EXECUTOR,
            "debug": DEBUG,
        },
    }
    (RUN_DIR / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_method() -> TraceAAD:
    llm = VLLMOpenAIAPI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        timeout=LLM_TIMEOUT,
        max_tokens=MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        enable_thinking=False,
    )
    task = TSPEvaluation(
        timeout_seconds=TASK_TIMEOUT_SECONDS,
        split=SPLIT,
        eval_workers=EVAL_WORKERS,
        eval_backend=EVAL_BACKEND,
    )
    return TraceAAD(
        llm=llm,
        evaluation=task,
        profiler=TraceAADProfiler(log_dir=str(LOG_DIR), log_style="complex", create_random_path=False),
        max_sample_nums=MAX_SAMPLE_NUMS,
        n_init=N_INIT,
        n_iterations=N_ITERATIONS,
        actions_per_iteration=ACTIONS_PER_ITERATION,
        max_actions_in_prompt=MAX_ACTIONS_IN_PROMPT,
        max_trajectory_length=MAX_TRAJECTORY_LENGTH,
        max_active_trajectories=MAX_ACTIVE_TRAJECTORIES,
        sampling_strategy=SAMPLING_STRATEGY,
        top_k=TOP_K,
        temperature=TRAJECTORY_TEMPERATURE,
        w_end=W_END,
        w_path=W_PATH,
        w_consistency=W_CONSISTENCY,
        w_downside=W_DOWNSIDE,
        discount=DISCOUNT,
        positive_threshold=POSITIVE_THRESHOLD,
        c0=C0,
        num_evaluators=NUM_EVALUATORS,
        max_consecutive_sample_failures=MAX_CONSECUTIVE_SAMPLE_FAILURES,
        multi_thread_or_process_eval=EVAL_EXECUTOR,
        debug_mode=DEBUG,
    )


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    write_run_config()
    print(f"run_dir={RUN_DIR}")
    with TMUX_LOG.open("a", encoding="utf-8", buffering=1) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"run_dir={RUN_DIR}", flush=True)
            print(f"log_dir={LOG_DIR}", flush=True)
            build_method().run()


if __name__ == "__main__":
    main()
