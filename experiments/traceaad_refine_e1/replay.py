"""Prequential E1-A: predict before updating, with no cross-run future feedback."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .prepare import DEFAULT, dump

KERNEL_MODELS={'B_kernel':'M3','E_kernel':'M2','Q_kernel':'M_quality'}
MODELS=['M0','M1','M2','M3_static','M3','M3_samecode_excluded','M3_self_history','M_quality',*KERNEL_MODELS]
K=10
SHRINK=10.0
WARMUP=50
REFIT=10


def neighborhood(pid,visible,past,nodes,distance,index,prior,exclude_code=False,allow_self=False):
    """Only earlier-selected parents and earlier completed Refine outcomes enter."""
    missing=[np.nan,np.nan,np.nan,0.0,prior,0.0,0.0,prior,0.0]
    if pid not in index:return missing
    pool=[j for j in visible if j!=pid and j in index and (not exclude_code or nodes[j]['code']!=nodes[pid]['code'])]
    if not pool:return missing
    ds=np.array([distance[index[pid],index[j]] for j in pool])
    order=np.argsort(ds,kind='stable')[:K];neighbors=[pool[k] for k in order];near=ds[order]
    dispersion=np.mean([abs(nodes[pid]['fitness']-nodes[j]['fitness']) for j in neighbors])
    static=[float(near[0]),float(np.log1p(1/(1e-6+near.mean()))),float(dispersion),1.0]
    selected=set(neighbors)
    if allow_self:selected.add(pid)
    edges=[e for e in past if e['parent_id'] in selected]
    if not edges:return static+[prior,0.0,0.0,prior,0.0]
    width=max(float(near[-1]),1e-6)
    w=np.array([math.exp(-0.5*(distance[index[pid],index[e['parent_id']]]/width)**2) for e in edges])
    y=np.array([float(e.get('parent_improved',False)) for e in edges])
    valid=np.array([e.get('fitness') is not None for e in edges],dtype=float)
    gain=np.array([max(e.get('parent_delta',0.0),0.0)/max(abs(e['parent_fitness']),1e-8) for e in edges])
    catastrophe=np.array([e.get('fitness') is not None and e.get('parent_delta',0)<-0.1*max(abs(e['parent_fitness']),1e-8) for e in edges],dtype=float)
    mass=float(w.sum());p=float((SHRINK*prior+w@y)/(SHRINK+mass))
    return static+[p,float(np.log1p(mass)),float(w@gain/(SHRINK+mass)),float((SHRINK*prior+w@valid)/(SHRINK+mass)),float(w@catastrophe/(SHRINK+mass))]


def feature_rows(out,run,require_behavior=True):
    root=out/'snapshot'/run['run_name'];folder=out/'distances'/run['run_name']
    nodes={n['id']:n for n in json.loads((root/'tree_state.json').read_text())['nodes']}
    events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    bids=json.loads((folder/'ids.json').read_text()) if (folder/'ids.json').exists() else []
    if require_behavior:assert (folder/'ids.json').exists(), 'Behavior matrix not ready'
    eids=json.loads((folder/'embedding_ids.json').read_text())
    bidx={n:i for i,n in enumerate(bids)};eidx={n:i for i,n in enumerate(eids)}
    bd=np.load(folder/'behavior.npy') if bids else np.zeros((0,0));ed=np.load(folder/'embedding.npy')
    q=np.array([nodes[n]['fitness'] for n in eids]);qd=abs(q[:,None]-q[None,:])
    past=[];visible=set();budget_before=0;archive=set();rows=[]
    for e in events:
        if e['requested_operator']==e['operator']=='Refine':
            pid=e['parent_id'];assert pid in archive
            prior=(1+sum(bool(x.get('parent_improved')) for x in past))/(10+len(past))
            baseline=[e['parent_fitness'],np.log1p(e['selection']['parent_count_before']),budget_before/1000]
            bx=neighborhood(pid,visible,past,nodes,bd,bidx,prior)
            ex=neighborhood(pid,visible,past,nodes,ed,eidx,prior)
            qx=neighborhood(pid,visible,past,nodes,qd,eidx,prior)
            features={'M1':baseline,'M2':baseline+ex,'M3_static':baseline+bx[:4], 'M3':baseline+bx,
                'M3_samecode_excluded':baseline+neighborhood(pid,visible,past,nodes,bd,bidx,prior,exclude_code=True),
                'M3_self_history':baseline+neighborhood(pid,visible,past,nodes,bd,bidx,prior,allow_self=True),
                'M_quality':baseline+qx}
            rows.append(dict(run=run['run_name'],task=run['task'],backend=run['backend'],candidate_id=e['candidate_id'],
                parent_id=pid,parent_fitness=e['parent_fitness'],budget_before=budget_before,
                history_count=len(past),first_parent_use=pid not in visible,eligible_history_parents=len(visible-{pid}),behavior_history_available=bool(bx[3]),
                behavior_parent_available=pid in bidx,prior=prior,features=features,
                y=int(bool(e.get('parent_improved'))),evaluated=e.get('evaluation_id') is not None,valid=e.get('fitness') is not None,
                gain=max(e.get('parent_delta',0.0),0.0),delta=e.get('parent_delta'),frontier=int(bool(e.get('frontier_improved'))),
                catastrophe=int(e.get('fitness') is not None and e.get('parent_delta',0)<-0.1*max(abs(e['parent_fitness']),1e-8))))
            visible.add(pid);past.append(e)
        if e.get('node_id') is not None:archive.add(e['node_id'])
        budget_before=e['budget_used']
    return rows


def predict(rows,C=1.0,models=MODELS):
    fits={};x={m:[] for m in models if m!='M0' and m not in KERNEL_MODELS};ys=[]
    for i,row in enumerate(rows):
        pred={'M0':row['prior']}
        pred.update({m:float(row['features'][source][7]) for m,source in KERNEL_MODELS.items() if m in models})
        for m in x:
            if len(ys)>=30 and len(set(ys))==2 and (i%REFIT==0 or m not in fits):
                fit=make_pipeline(SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True),StandardScaler(),
                                  LogisticRegression(C=C,max_iter=500,solver='lbfgs'))
                fit.fit(x[m],ys);fits[m]=fit
            pred[m]=float(fits[m].predict_proba([row['features'][m]])[0,1]) if m in fits else row['prior']
            x[m].append(row['features'][m])
        yield {k:v for k,v in row.items() if k!='features'}|{'predictions':pred,'primary':i>=WARMUP}
        ys.append(row['y'])


def metrics(rows,m):
    y=np.array([r['y'] for r in rows]);p=np.clip([r['predictions'][m] for r in rows],1e-6,1-1e-6)
    top=np.argsort(-p,kind='stable')[:max(1,math.ceil(.2*len(y)))]
    bins=[]
    for lo,hi in zip([0,.05,.1,.2,.4],[.05,.1,.2,.4,1.00001]):
        mask=(p>=lo)&(p<hi)
        if mask.any():bins.append(dict(lo=lo,hi=hi,n=int(mask.sum()),predicted=float(p[mask].mean()),actual=float(y[mask].mean())))
    return dict(n=len(y),positive=int(y.sum()),brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),
        pr_auc=float(average_precision_score(y,p)) if y.any() else None,
        lift=float(y[top].mean()/y.mean()) if y.any() else None,
        top_gain=float(np.mean([rows[i]['gain'] for i in top])),
        top_frontier_rate=float(np.mean([rows[i]['frontier'] for i in top])),
        top_catastrophe_rate=float(np.mean([rows[i]['catastrophe'] for i in top])),calibration=bins)


def summarize(predictions):
    rows=[r for r in predictions if r['primary']]
    groups={}
    for level in ['run','task','backend']:
        groups[level]={key:{m:metrics([r for r in rows if r[level]==key],m) for m in MODELS} for key in sorted({r[level] for r in rows})}
    deltas={}
    rng=np.random.default_rng(20260907)
    for m in MODELS:
        if m=='M1':continue
        d=np.array([v[m]['brier']-v['M1']['brier'] for v in groups['run'].values()])
        bs=np.mean(rng.choice(d,size=(10000,len(d)),replace=True),axis=1)
        deltas[m]=dict(mean=float(d.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),runs_better=int((d<0).sum()),runs=len(d))
    return dict(overall={m:metrics(rows,m) for m in MODELS},by=groups,
        run_macro_brier_delta_vs_M1=deltas,
        coverage=dict(primary=len(rows),evaluated=sum(r.get('evaluated',r['valid']) for r in rows),finite_children=sum(r['valid'] for r in rows),improved=sum(r['y'] for r in rows),behavior_available=sum(r['behavior_parent_available'] for r in rows),
                      behavior_history_available=sum(r['behavior_history_available'] for r in rows)))


def main(out):
    runs=json.loads((out/'snapshot.json').read_text())['runs'];predictions=[]
    for run in runs:
        rows=feature_rows(out,run);result=list(predict(rows));predictions+=result
        print(run['run_name'],'predicted',len(result),flush=True)
    (out/'replay_predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in predictions))
    dump(out/'summary.json',summarize(predictions))
    # Same comparisons on successful behavior coverage only, so missingness cannot explain gains.
    dump(out/'summary_complete_profiles.json',summarize([r for r in predictions if r['behavior_parent_available'] and r['behavior_history_available']]))
    dump(out/'summary_new_parents.json',summarize([r for r in predictions if r['first_parent_use']]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);args=p.parse_args();main(args.out)
