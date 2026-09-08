import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Selects the next node using a 2-step lookahead with progress-based return penalties.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
        
    n_remaining = len(unvisited_nodes)
    
    if n_remaining == 1:
        return int(unvisited_nodes[0])
        
    if n_remaining == 2:
        # Calculate exact cost for both possible choices
        i1 = unvisited_nodes[0]
        i2 = unvisited_nodes[1]
        
        # Option 1: Go to i1, then i2, then dest
        cost1 = distance_matrix[current_node, i1] + distance_matrix[i1, i2] + distance_matrix[i2, destination_node]
        # Option 2: Go to i2, then i1, then dest
        cost2 = distance_matrix[current_node, i2] + distance_matrix[i2, i1] + distance_matrix[i1, destination_node]
        
        return int(i1 if cost1 < cost2 else i2)

    n_total = distance_matrix.shape[0]
    if n_total <= 1:
        return destination_node

    # Extract distances
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Submatrix of distances between unvisited nodes
    sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
    np.fill_diagonal(sub, np.inf)
    
    # 1. Lookahead Cost: min_{j != i} (d(i, j) + d(j, dest))
    # This estimates the cost of visiting i, then the best next node j, then returning to dest.
    # This is a lower bound on the actual remaining tour cost from node i.
    lookahead = np.min(sub + d_dest[np.newaxis, :], axis=1)
    
    # 2. Isolation/Compactness: Average distance to other unvisited nodes
    # Helps keep the tour tight and avoid scattering.
    avg_iso = np.sum(sub, axis=1) / (n_remaining - 1)
    
    # 3. Progress-based weights
    progress = (n_total - n_remaining) / (n_total - 1)
    
    # We want to heavily penalize returning to destination too early.
    # The term (d_curr + d_dest - d_current_dest) is a lower bound on the extra cost 
    # if we visited 'i' and went straight to destination (skipping other unvisited nodes).
    # Actually, a better proxy for "is this a good time to go home?" is the potential
    # savings lost. But a simple heuristic is to weight the return distance heavily 
    # as progress increases, but also consider the lookahead.
    
    # Let's use a composite score:
    # Score = d_curr + lookahead + w_return * d_dest + w_iso * avg_iso
    
    # w_return should be small initially (don't go home yet) and large later.
    # However, simply adding d_dest might not be enough. 
    # The lookahead already includes d(j, dest). 
    # If we pick i, the remaining cost is roughly d_curr + min_j(d(i,j)+d(j,dest)).
    # This naturally handles the return constraint.
    
    # The main risk is picking a node that is close to current but far from the rest,
    # leading to a long jump later. The lookahead term min_j(d(i,j)+d(j,dest))
    # captures this: if i is isolated, d(i,j) is large, increasing the lookahead cost.
    
    # Weights:
    w_iso = 0.1 * (1.0 - progress)  # Fade out compactness preference as tour ends
    
    # Base score
    scores = d_curr + lookahead + w_iso * avg_iso
    
    # Tie-breaking / Stability: Add a tiny noise or prefer higher degree?
    # Let's just take the argmin.
    
    idx = int(np.argmin(scores))
    return int(unvisited_nodes[idx])