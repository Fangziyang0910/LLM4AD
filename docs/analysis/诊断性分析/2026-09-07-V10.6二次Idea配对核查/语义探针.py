"""Small deterministic checks on frozen, manually inspected NumPy programs. No fitness evaluation."""
import ast, json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
S=json.loads((HERE/'配对样本.json').read_text())['samples']

def fn(i,name,parent=False):
    ns={}; exec(compile(S[i]['parent_code'] if parent else S[i]['node']['code'],f'<sample-{i}>','exec'),ns)
    return ns[name]

def trace_return(f,*args):
    captured={}
    def trace(frame,event,arg):
        if frame.f_code is f.__code__ and event=='return': captured.update(frame.f_locals)
        return trace
    old=sys.gettrace();sys.settrace(trace)
    try:
        with np.errstate(all='ignore'): result=f(*args)
    finally:sys.settrace(old)
    return result,captured

def stripped_ast(code):
    t=ast.parse(code)
    for n in ast.walk(t):
        if isinstance(n,(ast.FunctionDef,ast.Module)) and n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str):n.body.pop(0)
    return ast.dump(t,include_attributes=False)

rng=np.random.default_rng(20260907); out={}
for i in (6,8,10):
    f=fn(i,'priority'); count=0
    for _ in range(1000):
        item=float(rng.integers(1,100)); bins=item+rng.integers(0,1000,30).astype(float)
        count+=int(np.argmax(f(item,bins))==np.argmin(bins))
    assert count==1000
    out[f'{i}_best_fit']={'sample':S[i]['sample_id'],'same_argmax_as_best_fit':count,'states':1000}
# Identical executable AST despite comment and docstring differences.
eq=stripped_ast(S[11]['node']['code'])==stripped_ast(S[11]['parent_code']);assert eq
out['11_parent_equivalence']={'ast_equal_ignoring_docstrings':eq}
# One altered OP clipping constant, but ratio / max(mean, .5*max) <= 2.
f,p=fn(15,'heuristics'),fn(15,'heuristics',True);same=0
with np.errstate(all='ignore'):
    for _ in range(100):
        xy=rng.random((25,2));dm=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
        prize=rng.random(25);prize[0]=0
        a=f(prize,dm,4.0);b=p(prize,dm,4.0);same+=int(np.array_equal(a,b))
assert same==100
out['15_inactive_clip']={'identical_parent_child_matrices':same,'instances':100}
# Vectorized assignment reads the old RHS, not a sequential running sum.
values=np.array([[2.,3.,5.,7.]]);cum=np.zeros_like(values);cum[:,0]=values[:,0];cum[:,1:]=values[:,1:]+cum[:,:-1]
out['17_cumulative_assignment']={'implemented':cum.tolist(),'true_cumsum':np.cumsum(values,axis=1).tolist()}
assert cum.tolist()==[[2.,5.,5.,7.]]
# Trace exact final TSP function on a finite Euclidean state, then shuffle candidates.
xy=rng.random((30,2));dm=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1);U=np.argsort(dm[0])[1:]
f=fn(21,'select_next_node');answer,loc=trace_return(f,0,0,U,dm)
V=U[::-1];answer2,loc2=trace_return(f,0,0,V,dm)
assert np.all(np.isnan(loc['scores'])) and answer==U[0] and answer2==V[0]
out['21_nan_scores']={'candidates':len(U),'all_scores_nan':bool(np.all(np.isnan(loc['scores']))),'all_structural_inf':bool(np.all(np.isinf(loc['structural']))),'chosen':int(answer),'first_candidate':int(U[0]),'chosen_after_reversal':int(answer2),'first_reversed':int(V[0])}
# Actual cluster accumulation in final OP function.
prize=np.arange(30,dtype=float);f=fn(17,'heuristics');answer,loc=trace_return(f,prize,dm,4.0)
out['17_actual_cumsum']={'equals_true_cumsum':bool(np.array_equal(loc['cum_prizes'],np.cumsum(loc['sorted_prizes'],axis=1))),'first_row_implemented':loc['cum_prizes'][0,:6].tolist(),'first_row_expected':np.cumsum(loc['sorted_prizes'],axis=1)[0,:6].tolist()}
assert not out['17_actual_cumsum']['equals_true_cumsum']
# The alleged identical VRPTW child actually modifies weight conditions.
eq=stripped_ast(S[25]['node']['code'])==stripped_ast(S[25]['parent_code']);assert not eq
out['25_false_identical']={'ast_equal_ignoring_docstrings':eq,'slack_norm_example':0.4,'parent_W_CONT':0.3,'child_W_CONT':0.39,'slack_norm_second_example':0.1,'parent_W_URG':0.2,'child_W_URG':0.15}
# Check sign of urgency and phase-dependent load terms (other score terms held fixed).
slack_ratio=np.array([0.1,0.4]); urgency=(2*np.maximum(.5-slack_ratio,0))**1.5
norm=lambda a:(a-a.min())/(a.max()-a.min())
load=np.array([.2,.6]); high=.2*norm((2*.75-1)*load);low=.2*norm((2*.25-1)*load)
out['26_preference_signs']={'slack_ratios':slack_ratio.tolist(),'urgency_cost':(.3*norm(urgency)).tolist(),'load_ratios':load.tolist(),'high_p_load_cost':high.tolist(),'low_p_load_cost':low.tolist()}
# A shared row scale cancels before floors/roundoff; maxlen multiplication breaks distance-scale invariance.
h=np.array([1.,3.,7.]);tau=np.array([.5,2.,1.]);prob=lambda h:(tau*h)/(tau*h).sum()
assert np.allclose(prob(h),prob(h/h.sum()))
out['0_row_scaling']={'raw_probabilities':prob(h).tolist(),'row_normalized_probabilities':prob(h/h.sum()).tolist(),'scope':'multiplicative ACO weights before additive epsilon, floors and floating point effects'}
out['12_distance_scaling']={'parent_ratio_original':.2/2,'parent_ratio_scaled_x10':2/20,'child_ratio_original':.2*4/2,'child_ratio_scaled_x10':2*40/20}
# Actual final Gaussian width differs from both descriptions.
f=fn(7,'priority');_,loc=trace_return(f,10.,np.array([10.,15.,20.,30.]))
assert loc['sigma']==10.
out['7_sigma']={'item':10,'implemented_sigma':float(loc['sigma']),'claimed_sigma':5}
(HERE/'语义探针结果.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
