from __future__ import annotations

import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.method.pathwise import PathWise, PathWiseProfiler
from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
from llm4ad.task.optimization.tsp_gls_2O import TSPGLSEvaluation
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI


TASK = "tsp_gls_2O"
METHOD = "pathwise"
TIMESTAMP = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path(__file__).resolve().parent / TIMESTAMP
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

MAX_SAMPLE_NUMS = 500
POP_SIZE = 6
INIT_POP_SIZE = None
NUM_ACTIONS = 2
NUM_ROLLOUTS = 2
MAX_INNER_STEPS = 3
NUM_EVALUATORS = 4
POLICY_PERTURBATION_PROB = 0.5
POLICY_PERTURBATION_FINAL_PROB = 0.25
WORLD_MODEL_PERTURBATION_PROB = 0.5
WORLD_MODEL_PERTURBATION_FINAL_PROB = 0.25
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
            "api_key": "<from .env>",
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
            "init_pop_size": INIT_POP_SIZE,
            "pop_size": POP_SIZE,
            "num_actions": NUM_ACTIONS,
            "num_rollouts": NUM_ROLLOUTS,
            "max_inner_steps": MAX_INNER_STEPS,
            "num_evaluators": NUM_EVALUATORS,
            "policy_perturbation_prob": POLICY_PERTURBATION_PROB,
            "policy_perturbation_final_prob": POLICY_PERTURBATION_FINAL_PROB,
            "world_model_perturbation_prob": WORLD_MODEL_PERTURBATION_PROB,
            "world_model_perturbation_final_prob": WORLD_MODEL_PERTURBATION_FINAL_PROB,
            "max_consecutive_sample_failures": MAX_CONSECUTIVE_SAMPLE_FAILURES,
            "eval_executor": EVAL_EXECUTOR,
            "debug": DEBUG,
        },
    }
    (RUN_DIR / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_method() -> PathWise:
    llm = OpenAIAPI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        timeout=LLM_TIMEOUT,
        max_tokens=MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        enable_thinking=False,
    )
    task = TSPGLSEvaluation(**TASK_KWARGS)
    return PathWise(
        llm=llm,
        evaluation=task,
        profiler=PathWiseProfiler(log_dir=str(LOG_DIR), log_style="complex", create_random_path=False),
        max_sample_nums=MAX_SAMPLE_NUMS,
        pop_size=POP_SIZE,
        init_pop_size=INIT_POP_SIZE,
        num_actions=NUM_ACTIONS,
        num_rollouts=NUM_ROLLOUTS,
        max_inner_steps=MAX_INNER_STEPS,
        num_evaluators=NUM_EVALUATORS,
        policy_perturbation_prob=POLICY_PERTURBATION_PROB,
        policy_perturbation_final_prob=POLICY_PERTURBATION_FINAL_PROB,
        world_model_perturbation_prob=WORLD_MODEL_PERTURBATION_PROB,
        world_model_perturbation_final_prob=WORLD_MODEL_PERTURBATION_FINAL_PROB,
        max_consecutive_sample_failures=MAX_CONSECUTIVE_SAMPLE_FAILURES,
        multi_thread_or_process_eval=EVAL_EXECUTOR,
        debug_mode=DEBUG,
    )


def main() -> None:
    os.environ["NO_PROXY"] = NO_PROXY_HOSTS
    os.environ["no_proxy"] = NO_PROXY_HOSTS
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    write_run_config()
    print(f"run_dir={RUN_DIR}")
    with TMUX_LOG.open("a", encoding="utf-8", buffering=1) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"run_dir={RUN_DIR}", flush=True)
            print(f"log_dir={LOG_DIR}", flush=True)
            print(f"llm={MODEL} @ {BASE_URL}", flush=True)
            print(f"task_seed={TASK_KWARGS.get('seed')}", flush=True)
            build_method().run()


if __name__ == "__main__":
    main()
