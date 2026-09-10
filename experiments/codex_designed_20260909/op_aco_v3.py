import numpy as np


def heuristics(prize, distance, maxlen):
    """Prize-efficient route insertion and 2-opt before ACO sampling."""
    d = np.asarray(distance, dtype=float).copy()
    np.fill_diagonal(d,0.0)
    prize = np.asarray(prize,dtype=float)
    best, best_reward, best_cost = None,-1.0,float('inf')
    for exponent in (0.7,1.0,1.4,2.0,3.0):
        route = [0,0]
        remaining = list(range(1,len(d)))
        length = 0.0
        while remaining:
            nodes = np.asarray(remaining)
            a,b = np.asarray(route[:-1]),np.asarray(route[1:])
            extra = d[a[None,:],nodes[:,None]]+d[nodes[:,None],b[None,:]]-d[a,b][None,:]
            positions = np.argmin(extra,axis=1)
            delta = extra[np.arange(len(nodes)),positions]
            feasible = delta+length <= maxlen-1e-10
            if not feasible.any():
                break
            merit = np.where(feasible,prize[nodes]/np.maximum(delta,1e-8)**exponent,-np.inf)
            pick = int(np.argmax(merit))
            route.insert(int(positions[pick])+1,int(nodes[pick]))
            remaining.remove(int(nodes[pick]))
            for _ in range(20):
                gain,move = 0.0,None
                for i in range(1,len(route)-2):
                    for j in range(i+1,len(route)-1):
                        change = d[route[i-1],route[j]]+d[route[i],route[j+1]]-d[route[i-1],route[i]]-d[route[j],route[j+1]]
                        if change < gain-1e-12:
                            gain,move = change,(i,j)
                if move is None:
                    break
                i,j=move
                route[i:j+1]=route[i:j+1][::-1]
            length = sum(d[a,b] for a,b in zip(route,route[1:]))
        reward = float(prize[route[1:-1]].sum())
        if reward > best_reward or (reward == best_reward and length < best_cost):
            best,best_reward,best_cost = route,reward,length
    prior = 1e-4*np.maximum(prize[None,:],1e-6)/np.maximum(d,1e-6)
    for a,b in zip(best,best[1:]):
        prior[a,b] = prior[b,a] = 1e3/max(d[a,b],1e-6)
    np.fill_diagonal(prior, 1e-9)
    return np.maximum(prior, 1e-9)
