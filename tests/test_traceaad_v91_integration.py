from __future__ import annotations

import json
from pathlib import Path

from experiments.runners.traceaad import run
from llm4ad.method.traceaad_v9 import TraceAADV9
from llm4ad.method.traceaad_v9_1 import (
    CHECKPOINT_VERSION,
    PROTOCOL_ID,
    TraceAADV91,
)


def test_v91_runner_is_independent_from_v9(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_1",
        budget=17,
        seed=4,
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")
    assert isinstance(method, TraceAADV91)
    assert not isinstance(method, TraceAADV9)
    assert spec.method_name == "traceaad_v9_1"
    assert spec.n_init == 4
    assert method._offspring_per_iteration == 1
    assert method._alpha == 0.5
    assert method.search_configuration()["protocol_id"] == PROTOCOL_ID
    assert method.search_configuration()["checkpoint_schema_version"] == CHECKPOINT_VERSION


def test_v91_run_config_records_mcts_protocol(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="online_bin_packing",
        version="v9_1",
        budget=17,
        seed=3,
        repeat=2,
        run_name="v9_1_obp_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    params = payload["method_params"]

    assert not resumed
    assert payload["method"] == "traceaad_v9_1"
    assert params["protocol_id"] == PROTOCOL_ID
    assert params["checkpoint_schema_version"] == CHECKPOINT_VERSION
    assert params["n_init"] == 4
    assert params["offspring_per_iteration"] == 1
    assert params["alpha"] == 0.5
    assert params["expansion_policy"] == "progressive_widening_uct"
    assert params["expansion_reward"] == "child_continuation_value"
    assert params["root_expansion"] is True
    assert "expansion_prior_weight" not in params
    assert "failed_expansion_reward" not in params


def test_v91_resume_rejects_v9_run_config(tmp_path: Path) -> None:
    v9 = run.make_run_spec(
        task="tsp_construct",
        version="v9",
        budget=17,
        experiments_root=tmp_path,
    )
    run_dir, run_name, _ = run.resolve_run_dir(v9)
    run.write_run_config(v9, run_dir, run_name)
    v91 = run.make_run_spec(
        task="tsp_construct",
        version="v9_1",
        budget=17,
        resume_from=run_dir,
        experiments_root=tmp_path,
    )
    try:
        run.resolve_run_dir(v91)
    except ValueError as exc:
        assert "resume config mismatch" in str(exc)
    else:
        raise AssertionError("V9.1 must not resume a V9 run")
