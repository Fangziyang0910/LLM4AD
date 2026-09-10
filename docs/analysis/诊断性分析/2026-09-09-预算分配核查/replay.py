"""Read-only selection audit. No LLM, tokenizer service, or evaluator calls.
Run from repo root with .venv/bin/python; snapshots bound live runs to checkpoint IDs.
"""
import json, math, hashlib, statistics
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from llm4ad.method.traceaad_v10_3.traceaad import calibrate_beta

OUT=Path(__file__).resolve().parent
ROOT=Path.cwd()
def mean(x): return statistics.mean(x) if x else None
def median(x): return statistics.median(x) if x else None
def digest(b): return hashlib.sha256(b).hexdigest()
def ess(p): return 1/sum(x*x for x in p)
def distribution(scores, counts, beta, decay=True):
    mx=max(scores); w=[math.exp(beta*(q-mx))/(math.sqrt(1+c) if decay else 1) for q,c in zip(scores,counts)]
    z=sum(w); return [x/z for x in w]
def tournament(scores,k=20):
    hist=Counter(scores); n=len(scores); better=0; probs={}
    for q,m in sorted(hist.items(),reverse=True):
        probs[q]=(((n-better)/n)**k-((n-better-m)/n)**k)/m; better+=m
    return [probs[q] for q in scores]
def metrics(p,scores,counts):
    mx=max(scores); n=len(p)
    return {'ess':ess(p),'max_probability':max(p),'best_tie_mass':sum(x for x,q in zip(p,scores) if q==mx),
            'untried_mass':sum(x for x,c in zip(p,counts) if c==0),
            'bottom_half_mass':sum(x for x,q in zip(p,scores) if sum(z>q for z in scores)>=n/2),
            'zero_probability_nodes':sum(x==0 for x in p)}

def compact(data):
    for version in ('v107r','v108_old'):
        for run in data[version]:
            records=run.pop('records')
            bins={}
            for name,check in [('first',lambda r:r['count_before']==0),('second_to_fourth',lambda r:1<=r['count_before']<=3),('fifth_plus',lambda r:r['count_before']>=4),('pivot_bottom_half',lambda r:r['operator']=='Pivot' and r['better_fraction']>=.5)]:
                rr=[r for r in records if check(r)]
                bins[name]={'attempts':len(rr),'parent_improved':sum(r['parent_improved'] for r in rr),'frontier_improved':sum(r['frontier_improved'] for r in rr),'valid':sum(r['status']=='ok' for r in rr)}
            run['response_bins']=bins
            for label,stage in run['stages'].items():
                lo,hi={'first180':(0,180),'early':(0,200),'middle':(200,500),'late':(500,1000),'all':(0,1000)}[label]
                rr=[r for r in records if lo<=r['budget_before']<hi]
                stage['first_use']=sum(r['count_before']==0 for r in rr)
                counts=stage.pop('parent_uses'); total=sum(counts.values())
                stage['distinct_parents']=len(counts)
                stage['top10_parent_share']=sum(sorted(counts.values(),reverse=True)[:10])/total if total else None
                stage['empirical_parent_ess']=total**2/sum(c*c for c in counts.values()) if total else None
                stage['statuses']=dict(Counter(r['status'] for r in rr))
            for snap in run['snapshots']:
                alt=snap['state_only_alternatives']
                alt['tournament20_no_count']=alt.pop('tournament20')
    return data


def audit(version,batch):
    root=ROOT/'experiments'/('traceaad_'+version)/'results'; manifest=json.loads((root/f'batch_{batch}.json').read_text()); runs=[]
    for row in manifest['plan']:
        path=root/row['task']/row['run_name']; raw=(path/'tree_state.json').read_bytes(); state=json.loads(raw)
        eventraw=(path/'events.jsonl').read_bytes(); events=[json.loads(line) for line in eventraw.splitlines() if line.strip()]
        events=[e for e in events if e['candidate_id']<=state['completed_attempts']]
        nodes={n['id']:n for n in state['nodes']}; assert len({e['candidate_id'] for e in events})==len(events)
        assert [e['evaluation_id'] for e in events if e.get('evaluation_id') is not None]==list(range(1,state['budget_used']+1))
        archive={}; counts=Counter(); groups={}; records=[]; snapshots=[]; births={}; selected=defaultdict(list); budget=0
        for e in events:
            parent=e.get('parent_id'); sel=e.get('selection',{}); op=e['operator']; req=e['requested_operator']
            if parent is not None:
                assert parent in archive and sel['parent_count_before']==counts[parent]
                pn=archive[parent]; values=[n['fitness'] for n in archive.values()]; n=len(values)
                if 'eligible_groups' in sel:
                    units=list(groups.values()); scores=[ns[0]['fitness'] for ns in units]; cs=[sum(counts[a['id']] for a in ns) for ns in units]
                    expected=sel['eligible_groups']
                else:
                    units=[[a] for a in archive.values()]; scores=[a[0]['fitness'] for a in units]; cs=[counts[a[0]['id']] for a in units]; expected=sel['eligible_nodes']
                correct=expected==len(units)
                max_score=max(scores)
                if correct:
                    p0=distribution(scores,cs,sel['beta']); assert math.isclose(ess(p0),sel['corrected_ess'],rel_tol=1e-7)
                    qi=next(i for i,ns in enumerate(units) if any(a['id']==parent for a in ns))
                    actual=([.925*x+.075/len(units) for x in p0] if version=='v10_7' else ([.5*x+.5/len(units) for x in p0] if req=='Pivot' else p0))
                    prob=actual[qi]/len(units[qi]); assert math.isclose(prob,sel['parent_probability'],rel_tol=1e-7,abs_tol=1e-12)
                r={'candidate_id':e['candidate_id'],'budget_before':budget,'operator':op,'requested':req,'parent_id':parent,
                   'fitness':e.get('fitness'),'parent_fitness':pn['fitness'],'count_before':counts[parent],
                   'parent_age':budget-pn['evaluation_id'],'better_fraction':sum(q>pn['fitness'] for q in values)/n,
                   'parent_best':pn['fitness']==max(values),'parent_improved':e.get('parent_improved',False),
                   'frontier_improved':e.get('frontier_improved',False),'both_improved':e.get('both_improved',False),
                   'status':e['status'],'parent_delta':e.get('parent_delta'),'frontier_delta':e.get('frontier_delta'),
                   'quality_ess':sel['quality_ess'],'corrected_ess':sel['corrected_ess'],
                   'nominal_ess':min(expected,max(.1*expected,2)),'eligible':expected,'all_archive_eligible':correct,
                   'best_ties':sum(q==max_score for q in scores),'beta':sel['beta']}
                records.append(r); selected[parent].append(e)
                if correct and budget in (49,99,179,249,499,749,949) and (not snapshots or snapshots[-1]['budget_before']!=budget):
                    node_scores=values; node_counts=[counts[a['id']] for a in archive.values()]
                    beta,_,_=calibrate_beta(node_scores,.1,2)
                    cb,_,_=calibrate_beta(node_scores,min(.1,32/len(node_scores)),2)
                    ps={'current_node':distribution(node_scores,node_counts,beta),
                        'no_count':distribution(node_scores,node_counts,beta,False),
                        'ess_cap32':distribution(node_scores,node_counts,cb),'tournament20':tournament(node_scores)}
                    base=ps['current_node']
                    snapshots.append({'budget_before':budget,'nodes':len(node_scores),'best_ties':sum(q==max(node_scores) for q in node_scores),
                        'state_only_alternatives':{name:{**metrics(p,node_scores,node_counts),'tv_from_current':.5*sum(abs(a-b) for a,b in zip(p,base))} for name,p in ps.items()},
                        'actual_unit_distribution':metrics(p0,scores,cs)})
                counts[parent]+=1
            if e.get('evaluation_id') is not None: budget=e['evaluation_id']
            if e.get('node_id') is not None:
                node=nodes[e['node_id']]; assert node['fitness']==e['fitness'] and node['parent_id']==parent
                assert digest(node['code'].encode())==e['code_hash']
                archive[node['id']]=node; groups.setdefault(node['code'],[]).append(node); births[node['id']]=e
        assert len(archive)==len(nodes)
        assert dict(counts)=={int(k):v for k,v in state['parent_selection_counts'].items()}
        cohorts={}
        for name,subset in [('early', [n for n in nodes.values() if n['evaluation_id']<=min(800,budget-100)]),('pivot_regressed',[n for n in nodes.values() if n['evaluation_id']<=min(800,budget-100) and n['operator']=='Pivot' and births[n['id']].get('parent_delta',0)<0])]:
            followed=[n for n in subset if any(e.get('evaluation_id') is not None and e['evaluation_id']<=n['evaluation_id']+100 for e in selected[n['id']])]
            rf=[n for n in subset if any(e['operator'] in ('Refine','Fuse') and e.get('evaluation_id') is not None and e['evaluation_id']<=n['evaluation_id']+100 for e in selected[n['id']])]
            cohorts[name]={'nodes':len(subset),'parent_eval_within100':len(followed),'rf_eval_within100':len(rf)}
        stages={}
        for label,lo,hi in [('first180',0,180),('early',0,200),('middle',200,500),('late',500,1000),('all',0,1000)]:
            rr=[r for r in records if lo<=r['budget_before']<hi]
            ops={}
            for op in ('Refine','Pivot','Fuse'):
                oo=[r for r in rr if r['operator']==op]
                ops[op]={'attempts':len(oo),'valid':sum(r['status']=='ok' for r in oo),'parent_improved':sum(r['parent_improved'] for r in oo),'frontier_improved':sum(r['frontier_improved'] for r in oo),'both_improved':sum(r['both_improved'] for r in oo),'mean_parent_better_fraction':mean([r['better_fraction'] for r in oo]),'mean_count_before':mean([r['count_before'] for r in oo]),'first_use':sum(r['count_before']==0 for r in oo),'bottom_half':sum(r['better_fraction']>=.5 for r in oo)}
            stages[label]={'attempts':len(rr),'operators':ops,'quality_ess_median':median([r['quality_ess'] for r in rr]),'corrected_ess_median':median([r['corrected_ess'] for r in rr]),'tie_floor_active':sum(r['best_ties']>r['nominal_ess'] for r in rr),'parent_best':sum(r['parent_best'] for r in rr),'parent_uses':dict(Counter(r['parent_id'] for r in rr))}
        summary={'task':row['task'],'repeat':row['repeat'],'backend':row['backend'],'path':str(path.relative_to(ROOT)),
                 'budget':budget,'nodes':len(nodes),'state_version':state['version'],'state_sha256':digest(raw),'events_sha256':digest(eventraw),
                 'completed_attempts':state['completed_attempts'],'last_event_ts':events[-1]['ts'],'mechanism':{k:state['mechanism'].get(k) for k in ['ess_fraction','ess_minimum','context_policy','group_policy','selection_policy','dedup_policy']},
                 'unique_code':len(groups),'best_fitness':max(ns[0]['fitness'] for ns in groups.values()) if version=='v10_8' else max(n['fitness'] for n in nodes.values()),
                 'replay_all_archive_eligible':sum(r['all_archive_eligible'] for r in records),'parent_events':len(records),'stages':stages,'cohorts':cohorts,'snapshots':snapshots,
                 'most_selected':[{'node_id':nid,'count':c,'fitness':nodes[nid]['fitness'],'birth_eval':nodes[nid]['evaluation_id'],'parent_improvements':sum(e.get('parent_improved',False) for e in selected[nid]),'frontier_improvements':sum(e.get('frontier_improved',False) for e in selected[nid])} for nid,c in counts.most_common(5)],
                 'final_best_node':max(nodes.values(),key=lambda n:n['fitness'])['id']}
        # Record concrete outcome examples and fixed-prefix records; no raw code duplication.
        summary['records']=records
        runs.append(summary)
    return runs
if __name__=='__main__':
    data={'captured_at':datetime.now(timezone.utc).isoformat(),'definitions':'All archived nodes reconstructed in candidate order. Distribution verified only when logged eligible count equals archive units. Alternatives are frozen-state probabilities, not counterfactual search outcomes. Live V108 bounded by checkpoint completed_attempts.',
          'v107r':audit('v10_7','20260908_v107r_formal'),'v108_old':audit('v10_8','20260909_v108_formal')}
    data=compact(data)
    (OUT/'snapshot.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('saved',OUT/'snapshot.json')
