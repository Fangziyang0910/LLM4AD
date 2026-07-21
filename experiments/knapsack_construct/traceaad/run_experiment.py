from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from llm4ad.method.traceaad import PortfolioWeights, TraceAAD, TraceAADProfiler, ValueWeights
from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
from llm4ad.task.optimization.knapsack_construct import KnapsackEvaluation
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI

TASK = "knapsack_construct"
TASK_KWARGS = get_generated_task_kwargs(TASK, "train")
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "http://222.201.145.8:8080/v1"
DEFAULT_MODEL = "qwen3.6-27b-awq"
DEFAULT_NO_PROXY = "183.36.243.124,222.201.145.8,localhost,127.0.0.1,::1"


def build_method(log_dir: Path, resume_from: Path | None = None) -> TraceAAD:
    no_proxy = os.environ.get("LLM_NO_PROXY", DEFAULT_NO_PROXY)
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy
    return TraceAAD(
        llm=OpenAIAPI(
            base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            timeout=600,
            max_tokens=16384,
            temperature=1.0,
            enable_thinking=False,
        ),
        evaluation=KnapsackEvaluation(**TASK_KWARGS),
        profiler=TraceAADProfiler(
            log_dir=str(log_dir), log_style="complex", create_random_path=False
        ),
        max_sample_nums=1000,
        n_init=4,
        actions_per_iteration=2,
        max_trajectory_length=8,
        max_active_trajectories=160,
        novelty_threshold=0.92,
        value_weights=ValueWeights(),
        portfolio_weights=PortfolioWeights(),
        max_consecutive_sample_failures=20,
        max_stalled_iterations=20,
        checkpoint_dir=log_dir / "checkpoints",
        checkpoint_interval=10,
        resume_from=resume_from,
    )


def main() -> None:
    resume_from = os.environ.get("RESUME_FROM", "").strip() or None
    if resume_from:
        run_dir = Path(resume_from).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
    else:
        timestamp = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        version = os.environ.get("EXPERIMENT_VERSION", "version3")
        run_dir = EXPERIMENT_ROOT / version / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_run_config(run_dir, timestamp)

    print(f"run_dir={run_dir}")
    log_dir = run_dir / "logs"
    with (run_dir / "tmux_run.log").open(
        "a", encoding="utf-8", buffering=1
    ) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"run_dir={run_dir}", flush=True)
            build_method(log_dir, None if resume_from is None else run_dir).run()


def _write_run_config(run_dir: Path, timestamp: str) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "task": TASK,
        "method": "traceaad",
        "experiment_version": os.environ.get("EXPERIMENT_VERSION", "version3"),
        "timestamp": timestamp,
        "llm": {
            "base_url": os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
            "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            "timeout": 600,
            "max_tokens": 16384,
            "temperature": 1.0,
            "enable_thinking": False,
            "api_key_configured": bool(os.environ.get("LLM_API_KEY", "EMPTY")),
            "no_proxy": os.environ.get("LLM_NO_PROXY", DEFAULT_NO_PROXY),
        },
        "task_eval": {"split": "train", **TASK_KWARGS},
        "method_params": {
            "max_sample_nums": 1000,
            "n_init": 4,
            "actions_per_iteration": 2,
            "max_trajectory_length": 8,
            "max_active_trajectories": 160,
            "novelty_threshold": 0.92,
            "max_consecutive_sample_failures": 20,
            "max_stalled_iterations": 20,
            "checkpoint_interval": 10,
            "value_weights": asdict(ValueWeights()),
            "portfolio_weights": asdict(PortfolioWeights()),
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
