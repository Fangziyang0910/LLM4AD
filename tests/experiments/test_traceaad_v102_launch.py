from collections import Counter

from experiments.infra.base import TASKS
from experiments.traceaad_v10_2.launch import BACKEND_MAP


def test_v102_formal_backend_distribution() -> None:
    assert set(BACKEND_MAP) == {
        (task, repeat)
        for task in TASKS
        for repeat in range(1, 4)
    }
    assert Counter(BACKEND_MAP.values()) == {
        "server3": 6,
        "server3b": 6,
        "local": 1,
        "server1": 2,
    }
