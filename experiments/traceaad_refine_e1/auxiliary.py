"""Prespecified auxiliary targets: relative positive gain, failure tail, frontier."""
import json

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .prepare import DEFAULT,dump
from .replay import feature_rows,predict,metrics,WARMUP,REFIT

MODELS=['M0','M1','M2','M3']


def main():
    manifest=json.loads((DEFAULT/'snapshot.json').read_text());summaries=[];predictions=[]
    for run in manifest['runs']:
        rows=feature_rows(DEFAULT,run)
        for target in ['catastrophe','frontier']:
            relabeled=[];past=[]
            for r in rows:
                relabeled.append(r|{'y':r[target],'prior':(1+sum(past))/(10+len(past))})
                past.append(r[target])
            ps=list(predict(relabeled,models=MODELS))
            for p in ps:predictions.append(p|{'target':target})
            summaries.append(dict(run=run['run_name'],task=run['task'],target=target,
                metrics={m:metrics([p for p in ps if p['primary']],m) for m in MODELS}))
        xs={m:[] for m in MODELS if m!='M0'};fits={};ys=[];errors={m:[] for m in MODELS}
        for i,r in enumerate(rows):
            target=r['gain']/max(abs(r['parent_fitness']),1e-8)
            pred={'M0':float(np.mean(ys)) if ys else 0.0}
            for m in xs:
                if i>=30 and (i%REFIT==0 or m not in fits):
                    fit=make_pipeline(SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True),StandardScaler(),Ridge(alpha=1.0))
                    fit.fit(xs[m],ys);fits[m]=fit
                pred[m]=max(0.0,float(fits[m].predict([r['features'][m]])[0])) if m in fits else pred['M0']
                xs[m].append(r['features'][m])
            if i>=WARMUP:
                for m in MODELS:errors[m].append((pred[m]-target)**2)
            predictions.append({k:v for k,v in r.items() if k!='features'}|{'target':'relative_positive_gain','target_value':target,'predictions':pred,'primary':i>=WARMUP})
            ys.append(target)
        summaries.append(dict(run=run['run_name'],task=run['task'],target='relative_positive_gain',metrics={m:{'mse':float(np.mean(errors[m])),'n':len(errors[m])} for m in MODELS}))
        print('auxiliary',run['run_name'],flush=True)
    dump(DEFAULT/'auxiliary_summary.json',summaries)
    (DEFAULT/'auxiliary_predictions.jsonl').write_text(''.join(json.dumps(p,allow_nan=False)+'\n' for p in predictions))

if __name__=='__main__':main()
