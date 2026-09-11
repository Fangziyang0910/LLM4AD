"""Snapshot live journals; diagnose actual operators and formation paths offline."""
import ast
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics as st

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SNAP = REPO / 'experiments/traceaad_v10_10/results/mechanism_snapshot_20260910'

def signature(code, numeric=False):
    class Clean(ast.NodeTransformer):
        def visit_Expr(self, node):
            return None if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) else self.generic_visit(node)
        def visit_Constant(self, node):
            if numeric and type(node.value) in (int, float, complex):
                return ast.Constant(value=0)
            return node
    return ast.dump(Clean().visit(ast.parse(code)), include_attributes=False)

def journal(path):
    rows=[]
    if path.exists():
        for line in path.read_bytes().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: break
    return rows

def main():
    SNAP.mkdir(parents=True, exist_ok=True)
    output={'snapshot_at':datetime.now().astimezone().isoformat(), 'runs':[]}
    for version,batch in [('10_10','20260910_v1010_formal'),('10_9','20260909_v109_initaware')]:
        root=REPO/f'experiments/traceaad_v{version}/results'
        plan=json.loads((root/f'batch_{batch}.json').read_text())['plan']
        for row in plan:
            run=root/row['task']/row['run_name']; snap=SNAP/row['run_name'];snap.mkdir(exist_ok=True)
            state_file=snap/'tree_state.json'
            if not state_file.exists():
                state_file.write_bytes((run/'tree_state.json').read_bytes())
                state=json.loads(state_file.read_text())
                events=[e for e in journal(run/'events.jsonl') if e['candidate_id']<=state['completed_attempts']]
                (snap/'events.json').write_text(json.dumps(events))
            state=json.loads(state_file.read_text());events=json.loads((snap/'events.json').read_text());nodes={n['id']:n for n in state['nodes']}
            best=max(nodes.values(),key=lambda n:n['fitness']);chain=[];cursor=best
            while cursor:
                chain.append({k:cursor.get(k) for k in ['id','parent_id','donor_id','evaluation_id','fitness','operator','idea']})
                cursor=nodes.get(cursor['parent_id'])
            chain.reverse()
            sig={i:signature(n['code']) for i,n in nodes.items()}
            num={i:signature(n['code'],True) for i,n in nodes.items()}
            groups=Counter(sig.values());ops={};parent_counts=Counter()
            for op in ['Init','Refine','Tune','Pivot','Fuse']:
                es=[e for e in events if e['operator']==op];valid=[e for e in es if e['status']=='ok']; normal=[e for e in es if not e.get('repair_of')]
                opnodes=[nodes[e['node_id']] for e in valid]
                ops[op]={'requests':len(normal),'repairs':len(es)-len(normal),'evaluations':sum(e.get('evaluation_id') is not None for e in es),'valid':len(valid),
                    'parent_wins':sum(e.get('parent_fitness') is not None and e['fitness']>e['parent_fitness']+1e-10 for e in valid),
                    'both_wins':sum(e.get('donor_fitness') is not None and e['fitness']>max(e['parent_fitness'],e['donor_fitness'])+1e-10 for e in valid),
                    'frontier_wins':sum(e.get('best_before') is not None and e['fitness']>e['best_before']+1e-10 for e in valid),
                    'unchanged_ast':sum(n['parent_id'] is not None and sig[n['id']]==sig[n['parent_id']] for n in opnodes),
                    'only_numeric':sum(n['parent_id'] is not None and num[n['id']]==num[n['parent_id']] and sig[n['id']]!=sig[n['parent_id']] for n in opnodes),
                    'median_prompt':st.median([e['prompt_tokens'] for e in normal]) if normal else None}
                parent_counts.update(e['parent_id'] for e in normal if e['parent_id'] is not None)
            frontier=[];f=float('-inf')
            for e in events:
                if e['status']=='ok' and e['fitness']>f+1e-10:
                    f=e['fitness'];frontier.append({k:e.get(k) for k in ['candidate_id','node_id','evaluation_id','operator','parent_id','donor_id','parent_fitness','donor_fitness','fitness','repair_of']})
            repairs=[e for e in events if e.get('repair_of')]
            info=dict(version=version,task=row['task'],repeat=row['repeat'],source=str(run.relative_to(REPO)),snapshot=str(snap.relative_to(REPO)),
                budget=state['budget_used'],completed=state['completed_attempts'],best={k:best[k] for k in ['id','fitness','evaluation_id','operator']},
                init_best=max(n['fitness'] for n in nodes.values() if n['parent_id'] is None),unique_asts=len(groups),nodes=len(nodes),largest_ast_group=max(groups.values()),
                statuses=dict(Counter(e['status'] for e in events)),failure_reasons=dict(Counter(e.get('reason') for e in events if e['status']!='ok')),
                ops=ops,parent_top=parent_counts.most_common(8),chain=chain,frontier=frontier,
                repairs=dict(total=len(repairs),valid=sum(e['status']=='ok' for e in repairs),evals=sum(e.get('evaluation_id') is not None for e in repairs)),
                curve={str(k):max((n['fitness'] for n in nodes.values() if n['evaluation_id']<=k),default=None) for k in [50,100,200,300,400,500,700,1000] if k<=state['budget_used']})
            output['runs'].append(info)
            if version=='10_10':
                (snap/'best_clean.py').write_text(ast.unparse(ast.parse(best['code']))+'\n')
    (OUT/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    for r in output['runs']:
        if r['version']=='10_10':
            print(r['task'],r['repeat'],r['budget'],round(r['best']['fitness'],5),'ops', {o:(s['valid'],s['parent_wins'],s['frontier_wins']) for o,s in r['ops'].items()},'repairs',r['repairs'])

if __name__=='__main__': main()
