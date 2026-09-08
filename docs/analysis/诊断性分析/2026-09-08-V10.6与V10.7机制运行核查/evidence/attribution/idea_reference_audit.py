import pathlib,json,re,collections
ROOT=pathlib.Path('/home/fang/code/LLM4AD/LLM4AD/experiments/traceaad_v10_7/results')
cuts={r['run']:r['snapshot_attempts'] for r in json.loads(pathlib.Path('/tmp/v107_mechanism_audit.json').read_text())}
pattern=re.compile(r'\bAlgorithm\s+(\d+)\b',re.I)
blockpat=re.compile(r'Algorithm (\d+)\nFitness: ([^\n]+)\nIdea: (.*?)\nCode:\n```python\n(.*?)\n```',re.S)
reports=[];examples=[]
for row in json.loads((ROOT/'batch_20260907_bounded_formal.json').read_text())['plan']:
 p=ROOT/row['task']/row['run_name'];cut=cuts[row['run_name']];ns={n['id']:n for n in json.loads((p/'tree_state.json').read_text())['nodes']};es={}
 with (p/'events.jsonl').open() as f:
  for l in f:
   try:e=json.loads(l)
   except:break
   if e['candidate_id']<=cut:es[e['candidate_id']]=e
 birth={e['node_id']:e for e in es.values() if e.get('node_id') is not None};ct={h:collections.Counter() for h in ['250','full']}
 for nid,e in birth.items():
  refs=pattern.findall(ns[nid]['idea'])
  for h in ct:
   if h=='250' and e['budget_used']>250:continue
   ct[h]['archived_nodes']+=1;ct[h]['nodes_with_Algorithm_N']+=bool(refs)
 with (p/'llm_calls.jsonl').open() as f:
  for l in f:
   try:c=json.loads(l)
   except:break
   e=es.get(c['candidate_id'])
   if not e:continue
   blocks=blockpat.findall(c['prompt'])
   for block,nid in zip(blocks,e['context_node_ids']):
    idea=block[2];refs=set(int(v) for v in pattern.findall(idea));b=birth[nid];oldids=b['context_node_ids'];newids=e['context_node_ids'];changed=[];missing=[];unknown=[]
    for x in refs:
     if not 1<=x<=len(oldids):unknown.append(x)
     elif not 1<=x<=len(newids):missing.append(x)
     elif oldids[x-1]!=newids[x-1]:changed.append(x)
    changed_code=[x for x in changed if ns[oldids[x-1]]['code']!=ns[newids[x-1]]['code']]
    for h in ct:
     if h=='250' and e['budget_used']>250:continue
     s=ct[h];s['all_blocks']+=1;s['shown_nonempty_ideas']+=bool(idea);s['blocks_with_Algorithm_N']+=bool(refs);s['blocks_wrong_or_missing']+=bool(changed or missing);s['blocks_changed_referent']+=bool(changed);s['blocks_changed_code']+=bool(changed_code);s['blocks_missing_current']+=bool(missing);s['blocks_unresolved_original']+=bool(unknown);s['resolved_mentions']+=len(refs)-len(unknown);s['changed_mentions']+=len(changed);s['missing_mentions']+=len(missing)
    if (changed or missing) and len([a for a in examples if a['task']==row['task']])<3:
     examples.append({'task':row['task'],'run':row['run_name'],'node':nid,'birth_candidate':b['candidate_id'],'birth_eval':b['evaluation_id'],'birth_context':oldids,'reuse_candidate':e['candidate_id'],'reuse_eval':e['evaluation_id'],'reuse_context':newids,'changed_indices':changed,'missing_indices':missing,'idea':idea})
 for h,s in ct.items():reports.append({'run':row['run_name'],'task':row['task'],'horizon':h,'counts':dict(s)})
result={'regex':pattern.pattern,'cuts':cuts,'reports':reports,'examples':examples}
pathlib.Path('/tmp/v107_idea_reference_audit.json').write_text(json.dumps(result,indent=2))
for h in ['250','full']:
 for task in ['tsp_construct','vrptw_construct','online_bin_packing','cvrp_aco','op_aco','ALL']:
  s=collections.Counter()
  for r in reports:
   if r['horizon']==h and (task=='ALL' or r['task']==task):s.update(r['counts'])
  print(h,task,dict(s))
