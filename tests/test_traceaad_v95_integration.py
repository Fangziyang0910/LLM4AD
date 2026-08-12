from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.method.traceaad_v9_5 import (
    CHECKPOINT_VERSION,
    PROTOCOL_ID,
    TraceAADV95,
)


def test_v95_runner_builds_complete_frozen_method(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_5",
        budget=1000,
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")

    assert isinstance(method, TraceAADV95)
    assert spec.method_name == "traceaad_v9_5"
    assert spec.n_init == 8
    assert spec.context_token_limit == 32768
    assert spec.llm_output_tokens == 8192
    assert method.search_configuration() == run._v95_method_params(spec)
    assert method.search_configuration()["protocol_id"] == PROTOCOL_ID
    assert method.search_configuration()["checkpoint_schema_version"] == (
        CHECKPOINT_VERSION
    )
    assert "operator" not in method.search_configuration()
    assert "credit" not in " ".join(method.search_configuration())
    method._llm.close()


def test_v95_run_config_records_logical_generator_without_service_source(
    tmp_path: Path,
) -> None:
    spec = run.make_run_spec(
        task="online_bin_packing",
        version="v9_5",
        backend="server3",
        budget=1000,
        repeat=2,
        run_name="v9_5_obp_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    assert payload["method"] == "traceaad_v9_5"
    assert payload["method_params"] == run._v95_method_params(spec)
    assert payload["generator_environment"]["logical_model_name"] == "Qwen3.6-27B"
    assert payload["generator_environment"]["max_total_context"] == 32768
    assert payload["generator_environment"]["max_new_tokens"] == 8192
    assert "backend" not in payload
    assert "llm" not in payload
    assert "base_url" not in json.dumps(payload)
    assert "quant" not in json.dumps(payload).lower()


def test_v95_resume_accepts_only_matching_protocol(tmp_path: Path) -> None:
    original = run.make_run_spec(
        task="tsp_construct",
        version="v9_5",
        budget=1000,
        run_name="matching_v95",
        experiments_root=tmp_path,
    )
    run_dir, run_name, _ = run.resolve_run_dir(original)
    run.write_run_config(original, run_dir, run_name)
    resumed_spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_5",
        budget=1000,
        resume_from=run_dir,
        experiments_root=tmp_path,
    )

    resolved, _, resumed = run.resolve_run_dir(resumed_spec)

    assert resumed
    assert resolved == run_dir


def test_v95_official_runner_fixes_root_count_to_eight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly eight"):
        run.make_run_spec(
            task="tsp_construct",
            version="v9_5",
            n_init=10,
            experiments_root=tmp_path,
        )
