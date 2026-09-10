"""Synthetic branch checks for reviewed snapshot programs; no task evaluation.

Uses installed NumPy. Only the three final TSP programs (n > DP thresholds)
and the listed OBP programs are executed. Does not establish held-out quality.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
snapshot = json.loads((HERE / '终局与过程快照.json').read_text())
result = {'numpy_version': np.__version__, 'tsp': [], 'obp': []}
for node in snapshot['representative_nodes']:
    if node['version'] != 'v107':
        continue
    tsp = node['task'] == 'tsp_construct' and (node['repeat'], node['id']) in [(1, 567), (2, 904), (3, 825)]
    obp = node['task'] == 'online_bin_packing' and (
        (node['repeat'] == 1 and node['id'] in [119, 130, 177, 195, 197])
        or (node['repeat'], node['id']) == (2, 61))
    if not (tsp or obp):
        continue
    assert hashlib.sha256(node['code'].encode()).hexdigest() == node['code_sha256']
    namespace = {}
    exec(compile(node['code'], f"reviewed_node_{node['id']}", 'exec'), namespace)
    row = {'repeat': node['repeat'], 'node_id': node['id'], 'code_sha256': node['code_sha256']}
    if tsp:
        disagreements = 0
        for seed in range(8):
            xy = np.random.default_rng(seed).random((50, 2))
            dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
            for count in (16, 20, 35, 48):
                candidates = np.arange(2, count + 2)
                candidates = candidates[np.argsort(dist[1, candidates])]
                with np.errstate(invalid='ignore', divide='ignore'):
                    choice = namespace['select_next_node'](1, 0, candidates, dist)
                assert choice in candidates
                disagreements += choice != candidates[0]
        row.update(cases=32, candidate_counts=[16, 20, 35, 48], seeds=list(range(8)),
                   non_nearest_choices=int(disagreements))
        assert disagreements == 0
        result['tsp'].append(row)
    else:
        bins = np.array([11.0, 20.0])
        scores = namespace['priority'](10.0, bins)
        assert np.isfinite(scores).all()
        row.update(item=10, bins=bins.tolist(), scores=scores.tolist(),
                   selected_capacity=float(bins[np.argmax(scores)]))
        assert row['selected_capacity'] == (20 if node['id'] in [195, 197] else 11)
        result['obp'].append(row)
assert len(result['tsp']) == 3 and len(result['obp']) == 6
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
