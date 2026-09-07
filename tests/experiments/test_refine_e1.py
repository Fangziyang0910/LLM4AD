"""Checks against self-leakage, future labels, and false zero-distance missingness."""
import copy
import numpy as np

from experiments.traceaad_refine_e1.replay import neighborhood,predict,MODELS
from experiments.traceaad_refine_e1.profile_core import compute_distance_matrix,profile_distance


def test_neighborhood_excludes_self_and_unobserved_nodes():
    nodes={i:{'fitness':float(i),'code':str(i)} for i in range(4)}
    distance=np.array([[0,.1,.2,.3],[.1,0,.3,.2],[.2,.3,0,.1],[.3,.2,.1,0]])
    edges=[dict(parent_id=0,parent_fitness=0.,parent_improved=True,fitness=1.,parent_delta=1.),
           dict(parent_id=1,parent_fitness=1.,parent_improved=False,fitness=1.,parent_delta=0.),
           dict(parent_id=3,parent_fitness=3.,parent_improved=True,fitness=4.,parent_delta=1.)]
    idx={i:i for i in nodes}
    a=neighborhood(0,{0,1},edges,nodes,distance,idx,.1)
    b=neighborhood(0,{0,1},edges[1:2],nodes,distance,idx,.1)
    assert a==b  # neither current-parent success nor unobserved-parent success may enter
    assert a[0]==.1 and a[4]<.1
    missing=neighborhood(9,{0,1},edges,nodes,distance,idx,.1)
    assert np.isnan(missing[0]) and missing[3]==0


def test_prequential_prediction_does_not_use_current_or_future_labels():
    rows=[]
    for i in range(55):
        rows.append(dict(y=int(i%7==0),prior=.1,features={m:[i/55.,float(i%3)]+[.1]*10 for m in MODELS if m!='M0'}))
    a=list(predict(rows))
    changed=copy.deepcopy(rows)
    for row in changed[40:]:row['y']=1-row['y']
    b=list(predict(changed))
    assert [r['predictions'] for r in a[:41]]==[r['predictions'] for r in b[:41]]
    assert all(0<=p<=1 for r in a for p in r['predictions'].values())


def test_optimized_dtw_matches_literal_for_prefix_and_nonprefix():
    for prefix,trajs in [(True,[[[0],[0,1],[0,1,2]],[[0],[0,2],[0,2,1]]]),
                         (False,[[[0,1,2],[0,2,1]],[[0,2,3],[0,3,2]]])]:
        profiles=[dict(trajectories=[t]) for t in trajs]
        m=compute_distance_matrix(profiles,prefix_mode=prefix)
        assert np.allclose(m,m.T) and np.allclose(np.diag(m),0)
        assert np.isclose(m[0,1],profile_distance(*profiles),rtol=1e-6)
