from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.method.mcts_ahd import MAProfiler, MCTS_AHD
from llm4ad.task.optimization.main.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.vllm_openai_api import VLLMOpenAIAPI


EXPERIMENT_DIR = Path(__file__).resolve().parent
LOG_DIR = EXPERIMENT_DIR / "logs"

llm = VLLMOpenAIAPI(
    base_url="http://222.201.145.8:8080/v1",
    api_key="EMPTY",
    model="qwen3.6-27b-awq",
    timeout=120,
    max_tokens=16384,
    temperature=1.0,
    enable_thinking=False,
)

task = TSPEvaluation(
    timeout_seconds=20,
    split="train",
    eval_workers=16,
    eval_backend="process",
)

method = MCTS_AHD(
    llm=llm,
    evaluation=task,
    profiler=MAProfiler(log_dir=str(LOG_DIR), log_style="complex", create_random_path=False),
    max_sample_nums=1000,
    init_size=4,
    pop_size=10,
    selection_num=2,
    num_samplers=4,
    num_evaluators=4,
    alpha=0.5,
    lambda_0=0.1,
    multi_thread_or_process_eval="thread",
)

method.run()
