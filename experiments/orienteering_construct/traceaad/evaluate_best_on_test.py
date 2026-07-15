from __future__ import annotations

from pathlib import Path

from experiments.orienteering_construct import evaluate_best_on_test as runner


runner.METHOD = "traceaad"
runner.RUN_DIRS = [
    Path("experiments/orienteering_construct/traceaad/20260714_141500_rep1"),
]
runner.OUTPUT_DIR = Path(__file__).resolve().parent / "eval_best_qwen36_27b_20260715_rep1"


if __name__ == "__main__":
    runner.main()
