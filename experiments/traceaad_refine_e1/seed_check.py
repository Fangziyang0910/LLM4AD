"""Small prespecified seed-noise check; no labels used to choose the anchors."""
import json
from pathlib import Path
import numpy as np
from . import profile_core as core
from .profile import inputs,candidate
from .prepare import DEFAULT,dump


def main():
    rows=[];seen=set()
    for run,nodes in inputs(DEFAULT):
        task=run['task']
        if task in seen:continue
        seen.add(task)
        for node in nodes[:2]:
            core._init_worker(task,'A',core.DEFAULT_TRAJECTORY_POINTS[task],core.DEFAULT_TIMEOUT_SECONDS[task])
            base_seed=core.PROGRAM_RANDOM_SEED
            base=core._profile_candidate(candidate(node))
            try:
                core.PROGRAM_RANDOM_SEED=base_seed+10000
                if 'evaluator' in core._GLOBAL_DATA and hasattr(core._GLOBAL_DATA['evaluator'],'aco_seed'):
                    core._GLOBAL_DATA['evaluator'].aco_seed+=10000
                alt=core._profile_candidate(candidate(node))
            finally:core.PROGRAM_RANDOM_SEED=base_seed
            row=dict(task=task,node_id=node['id'],ok=base['ok'] and alt['ok'],
                original=base,alternate=alt,self_distance=core.profile_distance(base,alt) if base['ok'] and alt['ok'] else None)
            rows.append(row);dump(DEFAULT/'seed_sensitivity.json',rows)
            print(task,node['id'],row['self_distance'],flush=True)

if __name__=='__main__':main()
