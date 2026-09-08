"""Fixed 4-anchor x 3-arm x 2-seed pilot; independent from formal runs."""
import concurrent.futures
import dataclasses
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import re
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path('/home/fang/code/LLM4AD/LLM4AD')
sys.path.insert(0, str(ROOT))
OUT = Path('/tmp/v107_prompt_counterfactual')
SEEDS = [20260908, 20260909]
ANCHORS = [('tsp_pivot', 'tsp_construct', 2, 97),
           ('obp_pivot', 'online_bin_packing', 3, 104),
           ('vrptw_fuse', 'vrptw_construct', 1, 197),
           ('tsp_fuse', 'tsp_construct', 3, 399)]

def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()

def client():
    from experiments.infra.base import BACKENDS, build_llm_client
    llm = build_llm_client(**dataclasses.asdict(BACKENDS['server3b']), max_tokens=16384)
    llm._client = llm._client.with_options(max_retries=0)
    return llm

def prepare():
    from experiments.infra.base import build_task, BACKENDS
    from llm4ad.method.traceaad_v10_3.schema import Node
    from llm4ad.method.traceaad_v10_7.prompts import PromptBuilder, TrajectoryBuilder, build_task_contract
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'plan.json').exists(), 'existing immutable plan; run it rather than replace'
    llm = client()
    anchors = []
    for label, task, repeat, cid in ANCHORS:
        run = next((ROOT / f'experiments/traceaad_v10_7/results/{task}').glob(f'*rep{repeat}'))
        events = [json.loads(l) for l in (run/'events.jsonl').open()]
        event = next(e for e in events if e['candidate_id'] == cid)
        call = None
        with (run/'llm_calls.jsonl').open() as f:
            for line in f:
                c = json.loads(line)
                if c['candidate_id'] == cid:
                    assert call is None
                    call = c
        before_ids = {e['node_id'] for e in events if e['candidate_id'] < cid and e['node_id'] is not None}
        nodes = {n['id']:Node(**n) for n in json.loads((run/'tree_state.json').read_text())['nodes'] if n['id'] in before_ids}
        parent = nodes[event['parent_id']]
        refs = [nodes[i] for i in event['context_node_ids'] if i != parent.id]
        evaluation, eval_config = build_task(task, None)
        contract = build_task_contract(evaluation)
        builder = TrajectoryBuilder(llm, contract, max_tokens=16128, history_tokens=8192, max_events=8, lookup=nodes.get)
        rebuilt, _, _, _, _ = builder.trajectory(parent, refs, event['operator'])
        assert rebuilt == call['prompt'], f'A reconstruction differs: {label}'
        assert sha(rebuilt) == call['prompt_hash']
        a = call['prompt']
        b, cleared = re.subn(r'(?m)^Idea: [\s\S]*?\n(?=Code:\n```python\n)', 'Idea: \n', a)
        assert cleared == len(event['context_node_ids'])
        for n in [parent, *refs]:
            assert a.count(n.code) == b.count(n.code), 'B changed code'
        def chain(node_id):
            found = []
            while nodes[node_id].parent_id is not None:
                node_id = nodes[node_id].parent_id
                found.append(node_id)
            return found
        c_meta = {}
        if event['operator'] == 'Pivot':
            ancestors = [nodes[i] for i in chain(parent.id)]
            legacy = PromptBuilder(llm, contract, max_tokens=16096, history_tokens=8192, max_events=8, lookup=nodes.get)
            built = legacy.build(parent, ancestors, 'Pivot')
            c_prompt = built.text + '\nKeep the Idea within 100 words.'
            c_meta = {'kind':'ancestor_context_instruction_comment_bundle', 'history_ids':built.history_ids, 'omissions':built.omissions}
        else:
            old_donor = nodes[event['donor_id']]
            auxiliary = next(n for n in refs if n.id != old_donor.id)
            eligible = [n for n in nodes.values() if n.id != parent.id and n.id not in chain(parent.id) and parent.id not in chain(n.id) and n.code not in [parent.code, auxiliary.code]]
            eligible.sort(key=lambda n:(-n.fitness,n.id))
            chosen = None
            for candidate in eligible:
                test, programs, donor, executed, blocks = builder.trajectory(parent, [auxiliary,candidate], 'Fuse')
                if llm.count_prompt_tokens(test) <= 16128:
                    assert donor.id == candidate.id
                    chosen = candidate
                    c_prompt = test
                    break
            assert chosen is not None
            c_meta = {'kind':'best_fitting_cross_lineage_donor_upper_bound', 'old_donor':dataclasses.asdict(old_donor), 'new_donor':dataclasses.asdict(chosen), 'auxiliary_id':auxiliary.id, 'context_node_ids':[n.id for n in programs]}
        prompts = {}
        for arm, text in [('A',a),('B',b),('C',c_prompt)]:
            tokens = llm.count_prompt_tokens(text)
            assert tokens <= 16128
            prompts[arm] = {'prompt':text,'prompt_hash':sha(text),'prompt_tokens':tokens}
            (OUT/f'{label}_{arm}.prompt.txt').write_text(text)
        anchors.append({'label':label,'task':task,'repeat':repeat,'run_dir':str(run),'candidate_id':cid,'original_event':event,'parent':dataclasses.asdict(parent),'context_nodes':[dataclasses.asdict(nodes[i]) for i in event['context_node_ids']],'evaluation_config':eval_config,'A_reconstructed_identical':True,'C':c_meta,'prompts':prompts})
        print('PREPARED',label,{arm:p['prompt_tokens'] for arm,p in prompts.items()},'C donor',c_meta.get('new_donor',{}).get('id'),flush=True)
    jobs = [{'anchor':a['label'],'arm':arm,'seed':seed,'id':f"{a['label']}_{arm}_{seed}"} for a in anchors for arm in ['A','B','C'] for seed in SEEDS]
    random.Random(20260908).shuffle(jobs)
    plan={'pilot':True,'anchors':anchors,'jobs':jobs,'backend':dataclasses.asdict(BACKENDS['server3b']),'seeds':SEEDS,'max_tokens':16384,'max_concurrency':3,'generation_retry':False,'created_at':time.time()}
    (OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    print('PLAN SAVED: 24 jobs; no generation yet.',flush=True)

def worker(job, anchor):
    from experiments.infra.base import build_task, set_random_seed
    from llm4ad.base import TextFunctionProgramConverter
    from llm4ad.base.evaluate import SecureEvaluator
    from llm4ad.method.traceaad_v10_6.traceaad import TraceAADV106
    path = OUT/f"{job['id']}.json"
    assert not path.exists(), 'never duplicate a completed request'
    row={**job,'task':anchor['task'],'parent_fitness':anchor['parent']['fitness'],'anchor_fitness':anchor['original_event']['fitness'],'started_at':time.time(),'prompt_hash':anchor['prompts'][job['arm']]['prompt_hash']}
    path.write_text(json.dumps({**row,'status':'reserved'},indent=2))
    started=time.monotonic()
    try:
        llm=client()
        response=llm.draw_sample_with_details(anchor['prompts'][job['arm']]['prompt'],max_tokens=16384,seed=job['seed'])
        row.update(response=response, generation_seconds=time.monotonic()-started, status='responded')
        path.write_text(json.dumps(row,ensure_ascii=False,indent=2))
        evaluator,config=build_task(anchor['task'],None)
        assert config == anchor['evaluation_config']
        holder=SimpleNamespace(_template_func=TextFunctionProgramConverter.text_to_function(evaluator.template_program))
        parsed=TraceAADV106.parse_response(holder,response['content'],response['finish_reason'] or 'unknown')
        if parsed is None:
            row.update(status='invalid_output',fitness=None,reason='strict_parse_failed')
        else:
            code=parsed[1]
            (OUT/f"{job['id']}.py").write_text(code)
            row.update(idea=parsed[0],code_hash=sha(code),status='evaluating')
            path.write_text(json.dumps(row,ensure_ascii=False,indent=2))
            set_random_seed(job['seed'])
            start_eval=time.monotonic()
            result=SecureEvaluator(evaluator).evaluate_program_with_details(code)
            row.update(status='ok' if result.result is not None else 'eval_failed',fitness=float(result.result) if result.result is not None else None,reason=result.failure_kind,evaluation_error=result.error,evaluation_seconds=time.monotonic()-start_eval)
    except Exception:
        row.update(status='error',error=traceback.format_exc(),fitness=None)
    row['finished_at']=time.time()
    path.write_text(json.dumps(row,ensure_ascii=False,indent=2))
    return row

def run():
    plan=json.loads((OUT/'plan.json').read_text());anchors={a['label']:a for a in plan['anchors']};results=[]
    assert not any((OUT/f"{j['id']}.json").exists() for j in plan['jobs']), 'existing reserved/completed jobs; do not regenerate'
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker,j,anchors[j['anchor']]):j for j in plan['jobs']}
        for future in concurrent.futures.as_completed(futures):
            row=future.result();results.append(row)
            summary={'pilot':True,'planned':24,'completed':len(results),'results':[{k:v for k,v in r.items() if k not in ['response','idea']} for r in results]}
            (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
            print('RESULT',row['id'],row['status'],row.get('fitness'),'seconds',round(row.get('generation_seconds',0),1),flush=True)

def summarize():
    from collections import Counter
    plan=json.loads((OUT/'plan.json').read_text())
    results=[json.loads((OUT/f"{job['id']}.json").read_text()) for job in plan['jobs']]
    assert len(results)==24 and all(r['status'] not in ['reserved','responded','evaluating'] for r in results)
    tables=[]
    for anchor in plan['anchors']:
        arms={arm:[next(r for r in results if r['anchor']==anchor['label'] and r['arm']==arm and r['seed']==seed)['fitness'] for seed in SEEDS] for arm in ['A','B','C']}
        tables.append({'anchor':anchor['label'],'parent_fitness':anchor['parent']['fitness'],'fitness_by_seed':arms})
    usage=Counter()
    for r in results:
        usage.update({k:v for k,v in r.get('response',{}).get('usage',{}).items() if isinstance(v,(int,float))})
    summary={'pilot':True,'planned':24,'completed':24,'generation_calls':24,'evaluation_calls':sum('evaluation_seconds' in r for r in results),'statuses':dict(Counter(r['status'] for r in results)),'finish_reasons':dict(Counter(r.get('response',{}).get('finish_reason') for r in results)),'seeds':SEEDS,'tables':tables,'usage':dict(usage),'generation_seconds_sum':sum(r.get('generation_seconds',0) for r in results),'evaluation_seconds_sum':sum(r.get('evaluation_seconds',0) for r in results),'results':[{k:v for k,v in r.items() if k not in ['response','idea']} for r in results],'limitations':['Two samples per arm per anchor; pilot only.','B removes displayed Idea while retaining full code comments.','Pivot C changes ancestry context, operator wording and code-comment display together.','Fuse C uses best fitting cross-lineage donor, not uniform top-5 reproduction.','Shared server3b and seed do not identify long-run search effect or guarantee deterministic sampling.']}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    if '--summarize' in sys.argv:summarize()
    elif '--run' in sys.argv:run()
    else:prepare()
