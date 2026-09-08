import json, re, ast, math, statistics
from pathlib import Path
from collections import Counter, defaultdict

path=Path('/tmp/v107_confound_audit.json');report=json.loads(path.read_text())
pattern=re.compile(r'\A\s*(?:Design )?Idea:\s*(\S.*?)\s*```(?:python)?[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*\s*\Z',re.DOTALL|re.MULTILINE)
result={}; examples=[]
def interface(fn):
 a=fn.args
 return ([x.arg for x in a.posonlyargs],[x.arg for x in a.args],[x.arg for x in a.kwonlyargs],a.vararg.arg if a.vararg else None,a.kwarg.arg if a.kwarg else None)
for run in report['runs']:
 if run['task'] not in ['tsp_construct','cvrp_aco','op_aco']:continue
 invalid={g['candidate_id'] for g in run['generation_records'] if g['status']=='invalid_output'}
 if not invalid:continue
 nodes=json.loads((Path(run['run_dir'])/'tree_state.json').read_text())['nodes']
 fns=[n for n in ast.parse(nodes[0]['code']).body if isinstance(n,ast.FunctionDef)]
 target=next((n for n in fns if n.name in ['select_next_node','heuristics']),fns[-1]);expected=interface(target)
 group=result.setdefault(run['task'],{}).setdefault(run['version'],{'invalid_count':0,'conditions':Counter(),'primary_reason':Counter(),'recovered_complete_interface_count':0,'recovered_ast_nodes':[]})
 for line in (Path(run['run_dir'])/'llm_calls.jsonl').open():
  g=json.loads(line)
  if g['candidate_id']>int(run['cutoffs']['250']):break
  if g['candidate_id'] not in invalid or g.get('stage','generation')!='generation':continue
  text=re.sub(r'<think>.*?</think>','',g.get('response') or '',flags=re.S).strip();conditions=[]
  if not re.match(r'^(?:Design )?Idea:\s*\S',text):conditions.append('noncanonical_idea_prefix')
  opened=re.search(r'```(?:python)?[ \t]*\r?\n',text)
  code=None
  if not opened:conditions.append('no_python_fence')
  else:
   suffix=text[opened.end():];end=re.search(r'^[ \t]*```[ \t]*$',suffix,re.M)
   if end:code=suffix[:end.start()]
   else:code=suffix;conditions.append('missing_closing_fence')
  complete=False
  if code is not None:
   try:
    tree=ast.parse(code);compile(tree,'<offline-check>','exec');funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==target.name]
    if len(funcs)!=1:conditions.append('missing_or_multiple_target_function')
    elif interface(funcs[0])!=expected:conditions.append('wrong_parameter_names_or_signature')
    else:complete=True
   except (SyntaxError,ValueError):conditions.append('unparsable_python')
  if not conditions:conditions.append('other_protocol_rejection')
  group['invalid_count']+=1;group['conditions'].update(conditions);group['primary_reason'][conditions[0]]+=1
  if complete:
   group['recovered_complete_interface_count']+=1;group['recovered_ast_nodes'].append(sum(1 for n in ast.walk(tree)))
  if len(examples)<40:examples.append(dict(task=run['task'],version=run['version'],rep=run['rep'],candidate_id=g['candidate_id'],conditions=conditions,complete_python_with_expected_interface=complete,prefix=text[:180],suffix=text[-180:]))
for task,vs in result.items():
 for v,s in vs.items():
  xs=s.pop('recovered_ast_nodes');s['recovered_ast_nodes_median']=statistics.median(xs) if xs else None
report['format_rejection_audit']={'by_task_version':result,'examples':examples,'note':'Recovery is only offline AST compilation and target function signature validation; it does not establish runtime validity or quality. First opening fenced block used; truncated prose and multiple code blocks may remain unrecoverable.'}
path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
