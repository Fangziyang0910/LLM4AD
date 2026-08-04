"""Smoke test for CALM method (no LLM server required for seed/schedule)."""
from __future__ import annotations

import tempfile

from llm4ad.base import LLM
from llm4ad.method.calm import CALM, CALMProfiler
from llm4ad.method.calm.prompt import Prompt
from llm4ad.task.optimization.cvrp_aco import CVRPACOEvaluation
from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.task.optimization.op_aco import OPACOEvaluation
from llm4ad.task.optimization.tsp_construct import TSPEvaluation


class DummyLLM(LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        return '{{The idea of the algorithm is to x}}\n```python\npass\n```'


def test_detectors() -> None:
    inj = Prompt(
        'Inject one novel, meaningful component into the following algorithm '
        'while preserving its main data flow. xxx'
    )
    sim = Prompt('Please create a locally refined version of the following algorithm. Keep')
    assert inj.op == 'injection', inj.op
    assert sim.op == 'simplify', sim.op
    print('detectors ok')


def test_seeds_and_schedule() -> None:
    tasks = [
        ('online_bin_packing', OBPEvaluation(**get_generated_task_kwargs('online_bin_packing', 'train'))),
        ('tsp_construct', TSPEvaluation(**get_generated_task_kwargs('tsp_construct', 'train'))),
        (
            'op_aco',
            OPACOEvaluation(
                split='train', timeout_seconds=60, n_ants=20, n_iterations=50,
                aco_seed=1234, n_workers=2,
            ),
        ),
        (
            'cvrp_aco',
            CVRPACOEvaluation(
                split='train', timeout_seconds=120, n_ants=30, n_iterations=100,
                aco_seed=1234, n_workers=2,
            ),
        ),
    ]
    for key, evaluation in tasks:
        with tempfile.TemporaryDirectory() as td:
            method = CALM(
                llm=DummyLLM(),
                evaluation=evaluation,
                profiler=CALMProfiler(log_dir=td, create_random_path=False),
                max_sample_nums=5,
                seed=0,
                task_key=key,
                num_evaluators=1,
                numeric_refine_initial_variants=2,
                numeric_refine_variants=0,
                debug_mode=False,
            )
            method._initialize_seeds()
            assert method._algos[0].perf is not None
            print(key, 'seed', method._algos[0].perf, 'archive', len(method._algos), 'budget', method._tot_sample_nums)
            method._prepare_dataset()
            assert len(method._messages) >= 1
            print('  prompts', len(method._messages))
    print('seeds_and_schedule ok')


if __name__ == '__main__':
    test_detectors()
    test_seeds_and_schedule()
