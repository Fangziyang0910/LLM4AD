"""Launch wrapper for the 2026-09-12 targeted baseline rerun (batch rerun2).

Runs EoH / MCTS-AHD with the current runner code but suppresses the
client-side top_p/top_k that today's infra defaults would inject: the July
original batches and the 20260824_rerun batch sent temperature only, and
this batch must stay comparable to them. Server-side sampling defaults
apply exactly as they did historically.

Usage:
    uv run python -m experiments.ops_baseline_rerun2 <method> [-- runner args]
"""

from __future__ import annotations

import sys

from experiments.infra import base

_original_client = base.build_llm_client


def _client_without_topk(**kwargs):
    kwargs["top_p"] = None
    kwargs["top_k"] = None
    return _original_client(**kwargs)


base.build_llm_client = _client_without_topk

_original_payload = base.llm_payload


def _payload_without_topk(**kwargs):
    kwargs["top_p"] = None
    kwargs["top_k"] = None
    return _original_payload(**kwargs)


base.llm_payload = _payload_without_topk


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    method = sys.argv[1]
    sys.argv = [f"experiments.{method}.run", *sys.argv[2:]]
    if method == "eoh":
        from experiments.eoh.run import main as run_main
    elif method == "mcts_ahd":
        from experiments.mcts_ahd.run import main as run_main
    else:
        raise SystemExit(f"unknown method: {method}")
    run_main()


if __name__ == "__main__":
    main()
