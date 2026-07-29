from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from llm4ad.method.traceaad_v5 import TraceAADProfiler, TraceAADV5, ValueWeights
from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from llm4ad.tools.env import resolve_llm_api_key

TASK = "online_bin_packing"
TASK_KWARGS = get_generated_task_kwargs(TASK, "train")
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "http://222.201.145.8:8080/v1"
DEFAULT_MODEL = "qwen3.6-27b-awq"
DEFAULT_NO_PROXY = "183.36.243.124,222.201.145.8,localhost,127.0.0.1,::1"


def build_method(log_dir: Path, resume_from: Path | None = None) -> TraceAADV5:
    no_proxy = os.environ.get("LLM_NO_PROXY", DEFAULT_NO_PROXY)
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    return TraceAADV5(
        llm=OpenAIAPI(
            base_url=base_url,
            api_key=resolve_llm_api_key(base_url=base_url),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            timeout=600,
            max_tokens=int(os.environ.get("LLM_OUTPUT_TOKEN_RESERVE", "8192")),
            temperature=1.0,
            enable_thinking=False,
        ),
        evaluation=OBPEvaluation(**TASK_KWARGS),
        profiler=TraceAADProfiler(
            log_dir=str(log_dir), log_style="simple", create_random_path=False
        ),
        max_sample_nums=1000,
        n_init=30,
        actions_per_iteration=2,
        max_trajectory_length=8,
        max_active_trajectories=30,
        elite_count=3,
        softmax_temperature=0.2,
        value_weights=ValueWeights(),
        action_max_tokens=int(os.environ.get("LLM_ACTION_MAX_TOKENS", "1024")),
        max_context_tokens=(
            int(os.environ["LLM_CONTEXT_TOKENS"])
            if os.environ.get("LLM_CONTEXT_TOKENS", "").strip()
            else None
        ),
        output_token_reserve=int(os.environ.get("LLM_OUTPUT_TOKEN_RESERVE", "8192")),
        random_seed=int(os.environ.get("TRACEAAD_RANDOM_SEED", "0")),
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
        version = os.environ.get("EXPERIMENT_VERSION", "version5_4")
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
            build_method(
                log_dir,
                None
                if resume_from is None
                else run_dir / "logs" / "checkpoints" / "latest.json",
            ).run()


def _write_run_config(run_dir: Path, timestamp: str) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "task": TASK,
        "method": "traceaad_v5",
        "experiment_version": os.environ.get("EXPERIMENT_VERSION", "version5_4"),
        "timestamp": timestamp,
        "llm": {
            "base_url": os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
            "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            "timeout": 600,
            "max_tokens": int(os.environ.get("LLM_OUTPUT_TOKEN_RESERVE", "8192")),
            "temperature": 1.0,
            "enable_thinking": False,
            "api_key_configured": bool(
                resolve_llm_api_key(
                    base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
                )
                != "EMPTY"
            ),
            "no_proxy": os.environ.get("LLM_NO_PROXY", DEFAULT_NO_PROXY),
        },
        "task_eval": {"split": "train", **TASK_KWARGS},
        "method_params": {
            "max_sample_nums": 1000,
            "n_init": 30,
            "actions_per_iteration": 2,
            "max_trajectory_length": 8,
            "max_active_trajectories": 30,
            "elite_count": 3,
            "softmax_temperature": 0.2,
            "max_consecutive_sample_failures": 20,
            "max_stalled_iterations": 20,
            "checkpoint_interval": 10,
            "value_weights": asdict(ValueWeights()),
            "action_max_tokens": int(os.environ.get("LLM_ACTION_MAX_TOKENS", "1024")),
            "max_context_tokens": (
                int(os.environ["LLM_CONTEXT_TOKENS"])
                if os.environ.get("LLM_CONTEXT_TOKENS", "").strip()
                else None
            ),
            "output_token_reserve": int(
                os.environ.get("LLM_OUTPUT_TOKEN_RESERVE", "8192")
            ),
            "random_seed": int(os.environ.get("TRACEAAD_RANDOM_SEED", "0")),
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
