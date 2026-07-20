from __future__ import annotations

import pytest

from llm4ad.method.traceaad.credit import directed_delta
from llm4ad.method.traceaad.portfolio import (
    aggregate_batch_utility,
    signed_utility,
)


def test_directed_delta_uses_task_direction() -> None:
    assert directed_delta(2.0, 3.5, maximize=True) == pytest.approx(1.5)
    assert directed_delta(2.0, 1.5, maximize=False) == pytest.approx(0.5)


def test_signed_utility_is_bounded_and_keeps_sign() -> None:
    assert signed_utility(0.0) == pytest.approx(0.0)
    assert -1.0 < signed_utility(-10.0) < 0.0
    assert 0.0 < signed_utility(10.0) < 1.0


def test_batch_reward_is_mean_candidate_utility() -> None:
    assert aggregate_batch_utility([0.2, -0.4, 0.8]) == pytest.approx(0.2)
    assert aggregate_batch_utility([]) == pytest.approx(-1.0)
