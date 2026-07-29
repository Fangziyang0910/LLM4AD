from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from llm4ad.method.traceaad_v5 import TraceAADV5


ENTRYPOINTS = (
    "experiments.tsp_construct.traceaad_v5.run_experiment",
    "experiments.cvrp_aco.traceaad_v5.run_experiment",
    "experiments.op_aco.traceaad_v5.run_experiment",
    "experiments.online_bin_packing.traceaad_v5.run_experiment",
)


def test_each_supported_task_builds_the_independent_v5_method(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EXPERIMENT_VERSION", raising=False)
    for module_name in ENTRYPOINTS:
        module = import_module(module_name)
        method = module.build_method(log_dir=tmp_path / module_name)

        assert isinstance(method, TraceAADV5)
        assert method._llm.max_tokens == 8192
        assert method._output_token_reserve == 8192
        assert method._action_max_tokens == 1024
        assert method._max_context_tokens is None
        assert not hasattr(method, "_global_experience")
        assert not hasattr(method, "_global_reflection_code_batch")
        assert callable(module.main)

        run_dir = tmp_path / f"{module_name}.run"
        run_dir.mkdir()
        module._write_run_config(run_dir, "test_timestamp")
        config = json.loads(
            (run_dir / "run_config.json").read_text(encoding="utf-8")
        )
        assert config["experiment_version"] == "version5_4"
