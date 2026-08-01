from __future__ import annotations

import pytest

from llm4ad.method.traceaad_v4.operators import directed_delta


def test_directed_delta_uses_task_direction() -> None:
    assert directed_delta(2.0, 3.5, maximize=True) == pytest.approx(1.5)
    assert directed_delta(2.0, 1.5, maximize=False) == pytest.approx(0.5)
