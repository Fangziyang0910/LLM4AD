"""One inspected OBP state; illustrates a decision change, not a task evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
snapshot=json.loads((HERE/'snapshot.json').read_text())
run=next(r for r in snapshot['v107r'] if r['task']=='online_bin_packing' and r['repeat']==2)
state=json.loads((Path(run['path'])/'tree_state.json').read_text())
nodes={n['id']:n for n in state['nodes']}
result={'task':'online_bin_packing','repeat':2,'purpose':'Purposive one-state mechanism illustration; no estimate of behavioral frequency or task fitness.', 'item':50.,'bins':[70.,100.],'nodes':[]}
for node_id in (412,433,602):
    node=nodes[node_id]
    scope={}
    # These three archived modules were inspected: numpy import and priority only.
    exec(compile(node['code'],f'archived-node-{node_id}','exec'),scope)
    scores=scope['priority'](50.,np.array([70.,100.]))
    result['nodes'].append({'node_id':node_id,'evaluation_id':node['evaluation_id'],'fitness':node['fitness'],'code_sha256':hashlib.sha256(node['code'].encode()).hexdigest(),'scores':scores.tolist(),'selected_capacity':float(result['bins'][int(np.argmax(scores))])})
assert [n['selected_capacity'] for n in result['nodes']]==[100.,70.,100.]
(HERE/'probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
