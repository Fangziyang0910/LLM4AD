from collections import Counter

from experiments.infra.base import TASKS
from experiments.traceaad_v10_4.launch import BACKEND_MAP
from experiments.traceaad_v10_4.run import build_parser


def test_v104_formal_backend_distribution() -> None:
    assert set(BACKEND_MAP) == {
        (task, repeat)
        for task in TASKS
        for repeat in range(1, 4)
    }
    assert Counter(BACKEND_MAP.values()) == {
        "server3": 4,
        "server3b": 4,
        "server1": 4,
        "local": 3,
    }


def test_v104_run_defaults() -> None:
    args = build_parser().parse_args(["--task", "tsp_construct"])
    assert args.budget == 1000
    assert args.idea_output_tokens == 1024
    assert args.output_tokens == 16384
