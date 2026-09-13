"""Freeze content-independent paired samples; rerun with --output to preserve original snapshot."""
import argparse, hashlib, json, random, statistics, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from llm4ad.method.traceaad_v10_6.traceaad import CODE_RE, SUMMARY_RE
from llm4ad.method.traceaad_v10_3.traceaad import _strip_thinking
from llm4ad.method.traceaad_v10_3.schema import normalize_code

def lines(p):
    with p.open() as f:
        for line in f:
            try: yield json.loads(line)
            except json.JSONDecodeError: continue # active JSONL tail may be incomplete

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=Path(__file__).with_name('配对样本.json')); args=parser.parse_args()
    if args.output.exists(): raise SystemExit('Snapshot exists: select a new --output path.')
    base=ROOT/'experiments/traceaad_v10_6/results'; batch='20260906_215231_revised'
    manifest=json.loads((base/f'batch_{batch}.json').read_text()); rng=random.Random(20260907)
    out={'batch':batch,'observed_at':datetime.now().astimezone().isoformat(),'seed':20260907,'sampling':'Sorted task/repeat; one uniformly sampled valid non-root child per evaluation stratum [1,200] and [201,400]; no filtering by operator, score or summary status.','runs':[],'samples':[]}
    for r in sorted(manifest['plan'],key=lambda r:(r['task'],r['repeat'])):
        d=base/r['task']/r['run_name']; state=json.loads((d/'tree_state.json').read_text()); nodes={n['id']:n for n in state['nodes']}
        events=[e for e in lines(d/'events.jsonl') if e.get('evaluation_id') is not None and e.get('budget_used',0)<=state['budget_used']]
        em={e['node_id']:e for e in events if e.get('node_id') in nodes}
        chosen=[]
        for low,high in [(1,200),(201,400)]:
            eligible=sorted([n for n in nodes.values() if n['parent_id'] is not None and low<=n['evaluation_id']<=high],key=lambda n:n['evaluation_id'])
            n=rng.choice(eligible); e=em[n['id']]
            sample={'sample_id':f"{r['task']}_r{r['repeat']}_e{n['evaluation_id']}",'task':r['task'],'repeat':r['repeat'],'backend':r['backend'],'run_path':str(d.relative_to(ROOT)),'stratum':[low,high],'eligible_count':len(eligible),'node':n,'event':e,'parent_code':nodes[n['parent_id']]['code'],'design_idea':e['design_idea'],'calibrated_idea':n['idea'],'code_sha256':hashlib.sha256(n['code'].encode()).hexdigest()}
            chosen.append(sample)
        by_cid={s['event']['candidate_id']:s for s in chosen}; stage_counts=Counter(); seconds=Counter(); tokens=Counter()
        for c in lines(d/'llm_calls.jsonl'):
            stage=c.get('stage','missing'); stage_counts[stage]+=1; seconds[stage]+=c.get('seconds',0)
            u=c.get('usage') or {}; tokens[stage]+=u.get('total_tokens',0)
            if c.get('candidate_id') in by_cid:
                by_cid[c['candidate_id']].setdefault('calls',[]).append(c)
        for s in chosen:
            g=[c for c in s['calls'] if c.get('stage')=='generation' and c.get('response')][-1]
            gm=CODE_RE.fullmatch(_strip_thinking(g['response'])); a=[c for c in s['calls'] if c.get('stage')=='thought_alignment' and c.get('response')]
            am=SUMMARY_RE.fullmatch(_strip_thinking(a[-1]['response']).strip()) if a else None
            s['provenance_checks']={'generated_code_equals_archive':bool(gm and normalize_code(gm.group(2))==s['node']['code']),'design_equals_generation':bool(gm and gm.group(1).strip()==s['design_idea']),'calibrated_equals_second_call':bool(am and am.group(1).strip()==s['calibrated_idea']),'code_in_second_prompt':bool(a and '# Final Code\n```python\n'+s['node']['code']+'\n```' in a[-1]['prompt'])}
        out['runs'].append({**r,'budget_used':state['budget_used'],'valid_nodes':len(nodes),'evaluated_events':len(events),'summary_status_all_evaluated':dict(Counter(e.get('summary_status','absent') for e in events)),'summary_status_valid':dict(Counter(e.get('summary_status','absent') for e in em.values())),'empty_node_ideas':sum(not n['idea'] for n in nodes.values()),'valid_summary_tokens':[e.get('summary_tokens',0) for e in em.values()],'call_counts':dict(stage_counts),'call_seconds':dict(seconds),'call_total_tokens':dict(tokens),'mechanism':state['mechanism']})
        out['samples'].extend(chosen)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(args.output)
    for s in out['samples']: print(s['sample_id'],s['node']['operator'],s['event'].get('summary_status'),s['provenance_checks'])
    print('budgets',sum(r['budget_used'] for r in out['runs']),'valid',sum(r['valid_nodes'] for r in out['runs']))
if __name__=='__main__': main()
