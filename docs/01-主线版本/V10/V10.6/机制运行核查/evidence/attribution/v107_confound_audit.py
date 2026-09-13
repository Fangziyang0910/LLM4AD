"""Read-only comparison of response/code structure and selection into evaluations."""
import ast, io, tokenize, re, json, math, statistics
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

ROOT=Path('/home/fang/code/LLM4AD/LLM4AD')
history=json.loads(Path('/tmp/v107_matched_history.json').read_text())
TASKS=['tsp_construct','cvrp_aco','op_aco','online_bin_packing','vrptw_construct']
PREFIXES={5:'20260905_',6:'20260906_215231_revised_',7:'20260907_bounded_formal_'}
BUDGETS=[100,250]

def rows(path):
 for line in path.open():
  try:yield json.loads(line)
  except json.JSONDecodeError:break

def structure(code):
 tree=ast.parse(code);ns=list(ast.walk(tree));comments=[]
 try:comments=[t.string for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type==tokenize.COMMENT]
 except tokenize.TokenError:pass
 cwords=sum(len(s.split()) for s in comments)
 docstrings=[ast.get_docstring(n,clean=False) or '' for n in ns if isinstance(n,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
 return dict(code_chars=len(code),code_lines=len(code.splitlines()),comment_lines=len(comments),comment_words=cwords,comment_chars=sum(len(s) for s in comments),comment_char_fraction=sum(len(s) for s in comments)/max(1,len(code)),docstring_words=sum(len(s.split()) for s in docstrings),ast_nodes=len(ns),statements=sum(isinstance(n,ast.stmt) for n in ns),loops=sum(isinstance(n,(ast.For,ast.While,ast.AsyncFor)) for n in ns),branches=sum(isinstance(n,(ast.If,ast.IfExp)) for n in ns),functions=sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in ns),calls=sum(isinstance(n,ast.Call) for n in ns))

def stats(xs):
 xs=sorted(xs)
 if not xs:return {'n':0}
 return dict(n=len(xs),mean=statistics.mean(xs),median=statistics.median(xs),p90=xs[min(len(xs)-1,math.ceil(.9*len(xs))-1)],max=max(xs),sum=sum(xs))

def summarize(gs,es,ns):
 genmetrics=['completion_tokens','prompt_tokens','response_chars','idea_words','raw_code_chars','seconds']
 nodemetrics=['code_chars','code_lines','comment_lines','comment_words','comment_chars','comment_char_fraction','docstring_words','ast_nodes','statements','loops','branches','functions','calls','generation_idea_words','node_idea_words']
 out={'generation_attempts':len(gs),'generation_status':dict(Counter(g['status'] for g in gs)),'event_status':dict(Counter(e['status'] for e in es)),'generation_finish_reason':dict(Counter(g['finish_reason'] for g in gs)),'reported_models':dict(Counter(g['reported_model'] for g in gs)),'accepted_nodes':len(ns),'generation':{m:stats([g[m] for g in gs if isinstance(g.get(m),(int,float))]) for m in genmetrics},'accepted_code':{m:stats([n[m] for n in ns if isinstance(n.get(m),(int,float))]) for m in nodemetrics},'generation_by_status':{s:{m:stats([g[m] for g in gs if g['status']==s and isinstance(g.get(m),(int,float))]) for m in genmetrics} for s in set(g['status'] for g in gs)},'generation_ideas_over100':sum(g['idea_words']>100 for g in gs),'generations_with_nonzero_reported_reasoning_tokens':sum((g.get('reasoning_tokens') or 0)>0 for g in gs),'eval_failed_reasons':dict(Counter(e.get('reason') for e in es if e['status']=='eval_failed'))}
 return out

report={'snapshot_started':datetime.now().isoformat(),'note':'First 100/250 actual evaluations, including initialization and failed evaluations. All generation attempts through the completing candidate are included. AST structure and length describe output form, not quality, semantic novelty, executed complexity or reasoning effort.','runs':[],'aggregate':{},'source_hash_variants':{},'same_endpoint_pairs':{},'formal_progress':[]}
buffers=defaultdict(lambda:defaultdict(lambda:[[],[],[]]))
for v in [5,6,7]:
 for cp in sorted((ROOT/f'experiments/traceaad_v10_{v}/results').glob('*/*/run_config.json')):
  if not cp.parent.name.startswith(PREFIXES[v]) or '_rep' not in cp.parent.name:continue
  c=json.loads(cp.read_text());state=json.loads((cp.parent/'tree_state.json').read_text());completed=state['completed_attempts']
  report['formal_progress'].append(dict(version=f'10.{v}',task=c['task'],rep=c['repeat'],budget=state['budget_used'],nodes=len(state['nodes']),backend=c['backend']))
  if c['task'] not in TASKS:continue
  es=[]
  for e in rows(cp.parent/'events.jsonl'):
   if e['candidate_id']>completed:break
   es.append(e)
   if e.get('evaluation_id')==250:break
  cutoffs={b:next(e['candidate_id'] for e in es if e.get('evaluation_id')==b) for b in BUDGETS}
  eventmap={e['candidate_id']:e for e in es};nodeevents={e['node_id']:e for e in es if e.get('node_id') is not None}
  gens=[];genmap={};summary_stats=Counter();seen=set()
  for g in rows(cp.parent/'llm_calls.jsonl'):
   if g['candidate_id']>cutoffs[250]:break
   if g.get('stage','generation')!='generation':
    summary_stats['calls']+=1;summary_stats['reported_completion_tokens']+=(g.get('usage') or {}).get('completion_tokens',0) or 0
    continue
   if g['call_id'] in seen:continue
   seen.add(g['call_id']);response=g.get('response') or '';e=eventmap.get(g['candidate_id'])
   if e is None:continue
   body=re.sub(r'<think>.*?</think>','',response,flags=re.S).strip();prefix=body.split('```',1)[0]
   idea=re.sub(r'^\s*(?:Design |Latest Design )?Idea:\s*','',prefix).strip()
   block=re.search(r'```(?:python)?[^\n]*\n(.*?)(?:\n\s*```|\Z)',body,re.S)
   usage=g.get('usage') or {};details=usage.get('completion_tokens_details') or {}
   rec=dict(candidate_id=g['candidate_id'],operator=e['operator'],parent_id=e['parent_id'],status=e['status'],finish_reason=g.get('finish_reason'),completion_tokens=usage.get('completion_tokens'),prompt_tokens=g.get('prompt_tokens'),response_chars=len(response),idea_words=len(idea.split()),raw_code_chars=len(block.group(1)) if block else 0,seconds=g.get('seconds'),reported_model=g.get('model'),reasoning_tokens=details.get('reasoning_tokens'))
   gens.append(rec);genmap[g['candidate_id']]=rec
  ns=[]
  for n in state['nodes']:
   if n['evaluation_id']>250:continue
   e=nodeevents[n['id']];g=genmap[e['candidate_id']]
   ns.append(dict(node_id=n['id'],evaluation_id=n['evaluation_id'],candidate_id=e['candidate_id'],operator=e['operator'],generation_idea_words=g['idea_words'],node_idea_words=len(n['idea'].split()),fitness=n['fitness'],**structure(n['code'])))
  run=dict(version=f'10.{v}',task=c['task'],rep=c['repeat'],backend=c['backend'],endpoint=c['llm']['base_url'],model=c['llm']['model'],run_dir=str(cp.parent),checkpoint_budget=state['budget_used'],cutoffs=cutoffs,summary=summary_stats,by_budget={},generation_records=gens,node_records=ns)
  for b,cut in cutoffs.items():
   gs=[g for g in gens if g['candidate_id']<=cut];bes=[e for e in es if e['candidate_id']<=cut];bns=[n for n in ns if n['evaluation_id']<=b]
   run['by_budget'][str(b)]=summarize(gs,bes,bns)
   dest=buffers[(c['task'],f'10.{v}')][b]
   for target,items in zip(dest,[gs,bes,bns]):target.extend(items)
  report['runs'].append(run)
  hashes=state['mechanism']['source_hashes']
  for p,h in hashes.items():
   if '/task/' in p or p.endswith('/base/evaluate.py'):
    report['source_hash_variants'].setdefault(p,{}).setdefault(h,[]).append(cp.parent.name)
  print(v,c['task'],c['repeat'],len(gens),len(ns),flush=True)
for (task,v),budgets in buffers.items():
 report['aggregate'].setdefault(task,{})[v]={str(b):summarize(*buf) for b,buf in budgets.items()}
for task in TASKS:
 report['same_endpoint_pairs'][task]={}
 for old in ['10.5','10.6']:
  oldruns={r['rep']:r for r in report['runs'] if r['task']==task and r['version']==old}
  newruns={r['rep']:r for r in report['runs'] if r['task']==task and r['version']=='10.7'}
  common=[rep for rep in [1,2,3] if oldruns[rep]['endpoint']==newruns[rep]['endpoint'] and oldruns[rep]['model']==newruns[rep]['model']]
  hlookup={(r['version'],r['rep']):r for r in history['runs'] if r['task']==task}
  report['same_endpoint_pairs'][task][old]={'rep_ids':common,'runs':[{v:dict(endpoint=(oldruns if v==old else newruns)[rep]['endpoint'],root_best=hlookup[(v,rep)]['root_best'],best_at_budget=hlookup[(v,rep)]['best_at_budget']) for v in [old,'10.7']} for rep in common]}
report['snapshot_finished']=datetime.now().isoformat()
Path('/tmp/v107_confound_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
