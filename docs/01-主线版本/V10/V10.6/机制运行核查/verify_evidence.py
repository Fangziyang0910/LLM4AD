"""Verify the frozen handoff without model calls or candidate execution."""
import gzip
import hashlib
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
EVIDENCE = BASE / 'evidence'


def read_json(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def main():
    manifest = read_json(EVIDENCE / 'manifest.json')
    expected = {item['path'] for item in manifest['files']}
    actual = {str(path.relative_to(EVIDENCE)) for path in EVIDENCE.rglob('*')
              if path.is_file() and path.name != 'manifest.json'}
    assert actual == expected, 'evidence inventory mismatch'
    for item in manifest['files']:
        path = EVIDENCE / item['path']
        raw = path.read_bytes()
        assert len(raw) == item['bytes'], path
        assert hashlib.sha256(raw).hexdigest() == item['sha256'], path
        if path.suffix in {'.json', '.gz'}:
            read_json(path)
    snapshots = read_json(EVIDENCE / 'health_20260908_0942.json')['runs']
    runs = read_json(EVIDENCE / 'formal_summary.json')['runs']
    assert len(runs) == len(snapshots) == 30
    key = lambda r: (r['version'], r['task'], r['repeat'])
    assert len({key(r) for r in runs}) == 30
    for run in runs:
        snap = next(s for s in snapshots if key(s) == key(run))
        assert all(run[k] == snap[k] for k in ('budget_used', 'valid_nodes', 'best', 'best_root'))
        frontier = run['frontier']
        assert frontier and frontier[-1]['fitness'] == snap['best']
        assert all(n['evaluation_id'] <= snap['budget_used'] for n in frontier)
        assert all(a['evaluation_id'] < b['evaluation_id'] and a['fitness'] < b['fitness']
                   for a, b in zip(frontier, frontier[1:]))
        assert sum(run['executed_operator_counts'].values()) == snap['completed_attempts']
    pilot = EVIDENCE / 'attribution/v107_prompt_counterfactual'
    plan = read_json(pilot / 'plan.json')
    results = read_json(pilot / 'summary.json')['results']
    assert len(results) == len(plan['jobs']) == 24
    assert {r['id'] for r in results} == {r['id'] for r in plan['jobs']}
    for anchor in plan['anchors']:
        for prompt in anchor['prompts'].values():
            raw = (pilot / prompt['prompt_file']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == prompt['prompt_hash']
    for result in results:
        raw = (pilot / (result['id'] + '.py')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == result['code_hash']
    candidates = EVIDENCE / 'attribution/v107_nan_candidates'
    plan = read_json(candidates / 'plan.json')
    assert len(plan) == 9
    for item in plan:
        raw = (candidates / Path(item['path']).name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
    for name in ['README.md', '归因论证.md']:
        path = BASE / name
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent / target.split('#')[0]).exists(), (name, target)
    print(f'PASS: {len(expected)} files, 30 progress/frontier summaries, 12 prompts, 33 candidate hashes and handoff links')


if __name__ == '__main__':
    main()
