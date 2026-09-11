"""Plot the saved snapshot; no live run reads or evaluations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
snapshot = json.loads((OUT / 'summary.json').read_text())
tasks = [('tsp_construct', 'TSP'), ('cvrp_aco', 'CVRP'), ('op_aco', 'OP'),
         ('online_bin_packing', 'OBP'), ('vrptw_construct', 'VRPTW')]
colors = ['#2878B5', '#D95F02', '#239B56']
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, (task, label) in zip(axes.flat, tasks):
    runs = [r for r in snapshot['runs'] if r['task'] == task]
    limit = max(r['budget'] for r in runs if r['version'] == '10_10')
    for run in runs:
        edges = [e for e in run['frontier'] if e['evaluation_id'] <= limit]
        xs = [e['evaluation_id'] for e in edges] + [min(limit, run['budget'])]
        ys = [(1 if task == 'op_aco' else -1) * e['fitness'] for e in edges]
        ys.append(ys[-1])
        current = run['version'] == '10_10'
        ax.step(xs, ys, where='post', color=colors[run['repeat']-1],
                linestyle='-' if current else '--', alpha=1 if current else .45,
                linewidth=1.6)
    ax.set_title(f'{label} ({"higher" if task == "op_aco" else "lower"} is better)')
    ax.set_xlabel('Evaluator calls')
    ax.set_xlim(1, limit)
    ax.grid(alpha=.18)
    values = [abs(value) for r in runs for value in (r['curve']['100'], r['best']['fitness'])]
    low, high = min(values), max(values)
    padding = (high-low)*.07
    ax.set_ylim(low-padding, high+padding)
axes.flat[-1].axis('off')
when = snapshot['snapshot_at'][:16].replace('T', ' ')
axes.flat[-1].text(.05, .8,
    'Solid: V10.10\nDashed: V10.9\n\nBlue / orange / green: runs 1 / 2 / 3\n\n'
    f'Snapshot: {when}\nAxes focus on post-initialization quality.',
    fontsize=12, linespacing=1.8)
fig.tight_layout()
fig.savefig(OUT / 'budget_curves.png', dpi=170)
plt.close(fig)
