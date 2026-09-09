"""Independent constructors; all arms share the production group scheduler."""

from pathlib import Path

from llm4ad.method.traceaad_v10_8 import TraceAADV108
from llm4ad.method.traceaad_v10_8.trajectory import digest


def build_history_ablation(arm, **kwargs):
    limits = {'code_only': 0, 'single_edge': 1, 'multi_edge': 8}
    runner = TraceAADV108(traj_gens=limits[arm], **kwargs)
    runner.mechanism['experiment_arm'] = arm
    runner.mechanism['source_hashes'][str(Path(__file__).resolve())] = digest(Path(__file__).read_text())
    return runner
