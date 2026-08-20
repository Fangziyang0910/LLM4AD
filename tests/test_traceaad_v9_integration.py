from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.method.traceaad_v9 import TraceAADV9


def test_v9_is_a_runner_version_with_core_defaults(tmp_path: Path) -> None:
    assert "v9" in run.VERSIONS
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9",
        budget=17,
        seed=4,
        experiments_root=tmp_path,
    )
    assert spec.n_init == 10
    assert spec.llm_output_tokens == 8192
    method = run.build_method(spec, tmp_path / "run")

    assert isinstance(method, TraceAADV9)
    assert method._n_init == 10
    assert method._offspring_per_iteration == 2
    assert method._ancestor_history_limit == 8
    assert method._direct_child_limit == 8
    assert method._direct_child_top_count == 4
    assert method._reference_temperature == 0.2
    assert method._exploration_constant == 0.1
    assert method._expansion_prior_weight == 1.0
    assert method.search_configuration()["history_protocol"] == "matched_history"


def test_v9_run_config_has_core_parameters(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="online_bin_packing",
        version="v9",
        budget=17,
        seed=3,
        repeat=2,
        run_name="v9_core_obp_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    params = payload["method_params"]

    assert not resumed
    assert payload["method"] == "traceaad_v9"
    assert params["history_protocol"] == "matched_history"
    assert params["n_init"] == 10
    assert params["offspring_per_iteration"] == 2
    assert params["expansion_policy"] == "adaptive_new_child_uct"
    assert params["root_expansion"] is False
    assert params["operators"] == [
        "trace_ideate",
        "trace_refine",
        "trace_synthesize",
        "trace_transfer",
    ]
    assert "action_max_tokens" not in params
    assert "value_weights" not in params


def test_v9_resume_rejects_changed_core_configuration(tmp_path: Path) -> None:
    run_dir = tmp_path / "v9"
    run_dir.mkdir()
    original = run.make_run_spec(
        task="tsp_construct",
        version="v9",
        budget=100,
        seed=3,
        experiments_root=tmp_path,
    )
    run.write_run_config(original, run_dir, run_dir.name)
    changed = run.make_run_spec(
        task="tsp_construct",
        version="v9",
        budget=101,
        seed=3,
        resume_from=run_dir,
        experiments_root=tmp_path,
    )
    with pytest.raises(ValueError, match="resume config mismatch"):
        run.resolve_run_dir(changed)
