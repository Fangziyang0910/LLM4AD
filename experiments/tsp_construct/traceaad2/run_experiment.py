from __future__ import annotations

import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.method.traceaad2 import TraceAAD2, TraceAAD2Profiler, ValueWeights
from llm4ad.task.optimization.main.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.vllm_openai_api import VLLMOpenAIAPI


TASK = "tsp_construct"
METHOD = "traceaad2"
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
ACTIONS_PER_ITERATION = 2
MAX_TRAJECTORY_LENGTH = 8
MAX_ACTIVE_TRAJECTORIES = 1000
SAMPLING_STRATEGY = "trajectory_ucb"
TOP_K = 5
TRAJECTORY_TEMPERATURE = 0.8
N_ISLANDS = 4
MAX_PER_ISLAND = 40
NOVELTY_THRESHOLD = 0.92
K_DISTILL = 10
PATIENCE_REFLECT = 8
NUM_EVALUATORS = 4
MAX_CONSECUTIVE_SAMPLE_FAILURES = 20
EVAL_EXECUTOR = "thread"
DEBUG = False

W_QUALITY = 0.30
W_POTENTIAL = 0.25
W_DIVERSITY = 0.10
W_NOVELTY = 0.10
W_GENERALIZATION = 0.25
DISCOUNT = 0.8
W_CONSISTENCY = 0.25
W_DOWNSIDE = 0.5
POSITIVE_THRESHOLD = 1e-6
C0 = 0.4


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
            "actions_per_iteration": ACTIONS_PER_ITERATION,
            "max_trajectory_length": MAX_TRAJECTORY_LENGTH,
            "max_active_trajectories": MAX_ACTIVE_TRAJECTORIES,
            "sampling_strategy": SAMPLING_STRATEGY,
            "top_k": TOP_K,
            "trajectory_temperature": TRAJECTORY_TEMPERATURE,
            "n_islands": N_ISLANDS,
            "max_per_island": MAX_PER_ISLAND,
            "novelty_threshold": NOVELTY_THRESHOLD,
            "k_distill": K_DISTILL,
            "patience_reflect": PATIENCE_REFLECT,
            "num_evaluators": NUM_EVALUATORS,
            "max_consecutive_sample_failures": MAX_CONSECUTIVE_SAMPLE_FAILURES,
            "eval_executor": EVAL_EXECUTOR,
            "debug": DEBUG,
            "value_weights": {
                "w_quality": W_QUALITY,
                "w_potential": W_POTENTIAL,
                "w_diversity": W_DIVERSITY,
                "w_novelty": W_NOVELTY,
                "w_generalization": W_GENERALIZATION,
                "discount": DISCOUNT,
                "w_consistency": W_CONSISTENCY,
                "w_downside": W_DOWNSIDE,
                "positive_threshold": POSITIVE_THRESHOLD,
                "c0": C0,
            },
        },
    }
    (RUN_DIR / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_method() -> TraceAAD2:
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
    return TraceAAD2(
        llm=llm,
        evaluation=task,
        profiler=TraceAAD2Profiler(log_dir=str(LOG_DIR), log_style="complex", create_random_path=False),
        max_sample_nums=MAX_SAMPLE_NUMS,
        n_init=N_INIT,
        actions_per_iteration=ACTIONS_PER_ITERATION,
        max_trajectory_length=MAX_TRAJECTORY_LENGTH,
        max_active_trajectories=MAX_ACTIVE_TRAJECTORIES,
        n_islands=N_ISLANDS,
        max_per_island=MAX_PER_ISLAND,
        sampling_strategy=SAMPLING_STRATEGY,
        novelty_threshold=NOVELTY_THRESHOLD,
        k_distill=K_DISTILL,
        patience_reflect=PATIENCE_REFLECT,
        value_weights=ValueWeights(
            w_quality=W_QUALITY,
            w_potential=W_POTENTIAL,
            w_diversity=W_DIVERSITY,
            w_novelty=W_NOVELTY,
            w_generalization=W_GENERALIZATION,
            top_k=TOP_K,
            temperature=TRAJECTORY_TEMPERATURE,
            discount=DISCOUNT,
            w_consistency=W_CONSISTENCY,
            w_downside=W_DOWNSIDE,
            positive_threshold=POSITIVE_THRESHOLD,
            c0=C0,
        ),
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
