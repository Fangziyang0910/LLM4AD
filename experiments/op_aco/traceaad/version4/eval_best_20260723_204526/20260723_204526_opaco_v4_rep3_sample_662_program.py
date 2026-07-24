import numpy as np

def heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    """Return edge desirability values for OP ant colony optimization.

    Args:
        prize: Node prizes with shape (n,). Node 0 is the depot.
        distance: Pairwise Euclidean distances with shape (n, n).
            Diagonal entries are large sentinels so self-loops are unused.
        maxlen: Maximum allowed tour length (return-to-depot constrained).

    Returns:
        An (n, n) edge-prior matrix. Larger values make an edge more likely
        to be sampled. Values at or below zero are treated as 1e-9.
    """
    n = len(prize)
    
    # Prize array with depot prize set to 0 (since depot gives no prize)
    prize_vals = prize.copy()
    prize_vals[0] = 0.0
    
    # Distance from any node i to depot (node 0)
    dist_to_depot = distance[:, 0]  # shape (n,)
    
    # Distance from depot (node 0) to any node j
    dist_from_depot = distance[0, :]  # shape (n,)
    
    # For edge (i, j):
    # cost of edge i->j
    dist_ij = distance  # shape (n, n)
    
    # Prize at destination j
    prize_j = prize_vals  # shape (n,) broadcasted
    
    # Avoid division by zero; replace zeros in dist_ij with a large number
    small_dist = 1e-9
    safe_dist = np.where(dist_ij < small_dist, small_dist, dist_ij)
    
    # Compute base efficiency: prize_j / dist_ij
    efficiency = prize_j / safe_dist  # shape (n, n)
    
    # Feasibility check:
    # Estimate minimum total tour length if we take edge i->j:
    # min_total = dist_from_depot[i] + dist_ij[i,j] + dist_to_depot[j]
    # This is a lower bound on the total tour length including this edge.
    min_total_tour = dist_from_depot[:, None] + distance + dist_to_depot[None, :]
    
    # Slack: how much budget remains after this lower-bound tour
    slack = maxlen - min_total_tour
    
    # Identify infeasible moves where even the shortest possible tour exceeds maxlen
    infeasible = slack <= 0
    
    # Dynamic budget-ratio penalty:
    # Calculate the ratio of the current edge cost to the remaining slack.
    # When slack is large, this ratio is small, and the penalty is mild.
    # When slack is small, this ratio is large, and the penalty is severe.
    epsilon = 1e-9
    safe_slack = np.maximum(slack, epsilon)
    
    # The penalty factor is inversely proportional to (1 + ratio).
    # Alternatively, use an exponential decay based on the ratio for sharper differentiation.
    # Let's use: decay = exp(-alpha * (dist_ij / safe_slack))
    # This ensures that as the edge cost becomes a larger fraction of the slack,
    # the heuristic value drops significantly.
    alpha = 2.0 # Increased alpha for sharper differentiation
    budget_ratio = dist_ij / safe_slack
    decay_factor = np.exp(-alpha * budget_ratio)
    
    # Compute a slack factor for feasible moves to reward flexibility
    # For infeasible moves, we will set a very low value later, so the decay factor 
    # there doesn't matter as much, but we keep it consistent.
    # Using log(1+slack) for feasible parts to give diminishing returns on extra slack
    slack_factor = np.where(
        infeasible,
        1.0, # Neutral factor, will be overridden by feasibility penalty
        1.0 + np.log1p(np.maximum(slack, 0.0))
    )
    
    # Combine efficiency, slack factor, and distance-based decay
    heuristic = efficiency * slack_factor * decay_factor
    
    # Refine prize-efficiency synergy term with non-linear scaling
    # Calculate dynamic threshold: median efficiency of feasible neighbors for each node i
    # We need to mask infeasible edges when calculating the threshold
    
    # Vectorized threshold calculation using np.nanmedian
    # Create a masked array where infeasible entries are set to NaN
    masked_efficiency = np.where(infeasible, np.nan, efficiency)
    
    # Compute median along axis 1 (columns), ignoring NaNs
    # np.nanmedian returns the median of non-NaN values
    threshold_row = np.nanmedian(masked_efficiency, axis=1)  # shape (n,)
    
    # Handle rows with all infeasible moves (NaN result)
    # If all moves are infeasible, nanmedian returns NaN. We handle this by filling with 0.
    threshold_row = np.where(np.isnan(threshold_row), 0.0, threshold_row)
    
    # Broadcast threshold to (n, n)
    threshold = threshold_row[:, None]  # shape (n, 1) broadcasted to (n, n)
            
    # Non-linear scaling function:
    # Scale factor = 1.0 if efficiency <= threshold
    # Scale factor = 1.0 + beta * ((efficiency - threshold) / threshold)^gamma if efficiency > threshold
    # This amplifies reward only for edges that are significantly more efficient than average
    
    beta = 0.5
    gamma = 1.5 # Non-linear exponent
    
    # Avoid division by zero in threshold
    safe_threshold = np.where(threshold < small_dist, small_dist, threshold)
    
    # Calculate ratio of efficiency to threshold
    eff_ratio = efficiency / safe_threshold
    
    # Apply non-linear scaling only when efficiency > threshold
    synergy_term = np.where(
        efficiency > threshold,
        1.0 + beta * np.power(np.maximum(eff_ratio - 1.0, 0.0), gamma),
        1.0
    )
    
    # Clamp synergy term to reasonable range [1, 10] to prevent domination
    synergy_term = np.clip(synergy_term, 1.0, 10.0)
    
    # MODIFIED IDEA: Dynamic budget-urgency scaling factor
    # Instead of a static 20% threshold, we calculate a dynamic critical threshold
    # based on the local density of high-prize nodes.
    
    # 1. Calculate average distance between nodes to estimate spatial density
    avg_dist = np.mean(distance[distance > small_dist])
    
    # 2. Estimate a "density radius" based on inverse density
    # Higher density -> smaller radius
    density_radius = avg_dist * 0.5 
    
    # 3. For each node, calculate the "potential prize density" within its local neighborhood
    # We define local neighborhood as nodes within density_radius * sqrt(n) or similar
    # Simpler approach: Use the median distance to nearest neighbors as a proxy for local cluster tightness
    
    # Calculate for each node j, the sum of prizes of nodes within 'density_radius'
    # This is an O(N^2) precomputation.
    
    # Mask for local neighborhood
    local_mask = distance < density_radius
    local_mask = np.logical_and(local_mask, np.eye(n) == 0) # Exclude self
    
    # Sum of prizes in local neighborhood for each node
    # prize_vals has shape (n,)
    local_prize_density = np.sum(prize_vals[None, :] * local_mask, axis=1)
    
    # Normalize by the number of neighbors or use raw sum if consistent
    # Let's use the raw sum scaled by a factor to match magnitude of maxlen
    
    # Dynamic critical threshold:
    # Base threshold is some fraction of maxlen.
    # We scale this fraction by the local prize density.
    # High density -> Lower critical threshold (wait longer to trigger urgency, as we can collect more)
    # Low density -> Higher critical threshold (trigger urgency earlier to ensure we grab what's available)
    
    # Actually, if density is high, we might want to be more greedy sooner? 
    # Or if density is low, we need to be careful not to waste budget?
    
    # Let's try: 
    # critical_threshold_factor = 0.2 + (max_prize - local_prize) * coeff
    # If local prize is low (sparse), factor increases -> urgency triggers earlier.
    # If local prize is high (dense), factor decreases -> urgency triggers later.
    
    max_local_prize = np.max(local_prize_density)
    if max_local_prize > 0:
        # Normalize local prize to [0, 1]
        norm_local_prize = local_prize_density / max_local_prize
    else:
        norm_local_prize = np.zeros(n)
        
    # Invert: Low prize -> High urgency factor
    # Factor ranges from 0.1 (high prize) to 0.5 (low prize)
    dynamic_factor = 0.1 + 0.4 * (1.0 - norm_local_prize)
    
    # Apply this factor to each node's critical threshold
    # critical_threshold[i, j] = dynamic_factor[j] * maxlen
    # We broadcast dynamic_factor to (n, n)
    dynamic_thresholds = dynamic_factor[None, :] * maxlen
    
    # Calculate urgency using these dynamic thresholds
    # raw_urgency = 1.0 - (safe_slack / dynamic_thresholds)
    # But safe_slack is (n, n) and dynamic_thresholds is (n, n)
    
    # Avoid division by zero
    safe_dyn_thresh = np.maximum(dynamic_thresholds, epsilon)
    
    raw_urgency = 1.0 - (safe_slack / safe_dyn_thresh)
    
    # MODIFIED: Use hyperbolic tangent transition for smooth urgency activation
    # tanh(x) ranges from -1 to 1. We want a factor that ranges from 1.0 to (1.0 + max_amplification).
    # We shift and scale: factor = 1.0 + max_amp/2 * (1 + tanh(k * (slack / thresh - 1)))
    # When slack >> thresh, arg is large positive, tanh -> 1, factor -> 1.0 + max_amp (No, wait. We want urgency when slack is LOW)
    # Let's define x = (slack / thresh).
    # We want urgency when x < 1.
    # Let's use: urgency_factor = 1.0 + max_amp * (1 - tanh(k * (slack / thresh - 1))) / 2 ??
    # If x >> 1 (plenty of slack), tanh(large) = 1. Factor = 1.0 + max_amp * (0) = 1.0.
    # If x << 1 (tight slack), tanh(negative large) = -1. Factor = 1.0 + max_amp * (1 - (-1))/2 = 1.0 + max_amp.
    
    # Refinement: Replace static k_steepness with node-dependent value
    # k_steepness is inversely proportional to local prize density.
    # High density -> Low k (smooth transition)
    # Low density -> High k (sharp transition)
    
    # Normalize local_prize_density for k calculation
    # Avoid division by zero
    safe_local_prize = np.maximum(local_prize_density, epsilon)
    
    # Base k value
    base_k = 5.0
    # Scale factor. If local_prize is 0, k is large. If local_prize is max, k is base_k.
    # Let's define k_j = base_k * (1 + alpha_k * (1 - norm_local_prize))
    alpha_k = 2.0
    
    k_steepness_per_node = base_k * (1.0 + alpha_k * (1.0 - norm_local_prize))
    
    # Broadcast k to (n, n) based on destination node j
    k_matrix = k_steepness_per_node[None, :] # shape (n, n)
    
    urgency_arg = k_matrix * (safe_slack / safe_dyn_thresh - 1.0)
    
    # tanh maps to [-1, 1]
    # We want urgency factor to map to [1.0, 1.0 + max_amplification]
    # When slack is high (arg > 0), tanh -> 1. We want factor -> 1.0.
    # When slack is low (arg < 0), tanh -> -1. We want factor -> 1.0 + max_amp.
    
    max_amplification = 2.0
    
    tanh_val = np.tanh(urgency_arg)
    
    # Map tanh_val from [-1, 1] to [1.0, 1.0 + max_amplification]
    # y = m * x + c
    # x = 1 -> y = 1
    # x = -1 -> y = 1 + max_amp
    # 1 = m + c
    # 1 + max_amp = -m + c
    # Subtract: max_amp = 2m => m = max_amp / 2
    # c = 1 - m = 1 - max_amp / 2
    
    m = max_amplification / 2.0
    c = 1.0 - m
    
    urgency_amplifier = m * tanh_val + c

    # Apply urgency amplifier to the synergy term specifically
    synergy_term_urgent = synergy_term * urgency_amplifier

    # Use the urgent synergy term for the final heuristic calculation
    heuristic *= synergy_term_urgent

    # Depot-return feasibility bonus
    # Rewards edges (i, j) where destination j is close to depot relative to slack
    # Ratio: dist_to_depot[j] / safe_slack
    # If j is far from depot or slack is tight, this ratio is high -> penalty (decay)
    # If j is close to depot or slack is loose, this ratio is low -> bonus (less decay)
    # We use a similar exponential decay form.
    dist_return_ratio = dist_to_depot[None, :] / safe_slack
    # Use a smaller alpha for this factor to not overpower other terms, but still penalize
    alpha_return = 1.0
    depot_bonus = np.exp(-alpha_return * dist_return_ratio)
    
    heuristic *= depot_bonus
    
    # Modified Angular Diversification with Neighbor Accessibility Heuristic
    # We want to encourage moves that do not double back towards the depot immediately
    # AND reward nodes that have high "branching factor" (many feasible next steps).

    # 1. Angular Term
    # Vectors:
    # V_depot = Depot - Current Node i
    # V_j     = Node j - Current Node i
    # Using Law of Cosines with distances from depot:
    # a = dist(i, depot), b = dist(j, depot), c = dist(i, j)
    # cos(theta) = (a^2 + c^2 - b^2) / (2*a*c)
    
    a_sq = dist_to_depot[:, None]**2  # a^2 for all j
    c_sq = dist_ij**2                 # c^2 for all i, j
    b_sq = dist_to_depot[None, :]**2 # b^2 for all i
    
    # Denominator: 2 * a * c
    denom = 2.0 * dist_to_depot[:, None] * dist_ij
    
    # Avoid division by zero in denominator
    safe_denom = np.where(denom < small_dist, small_dist, denom)
    
    # Numerator: a^2 + c^2 - b^2
    numer = a_sq + c_sq - b_sq
    
    # Cosine of the angle
    cos_theta = numer / safe_denom
    
    # Clip cosine to [-1, 1] to handle numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 2. Neighbor Accessibility Score (Optimized Fully Vectorized)
    # Estimate the number of feasible next-step nodes from j.
    # Remaining budget after taking edge i->j and arriving at j:
    # Budget at j (after i->j) = maxlen - (dist_depot_i + dist_ij)
    budget_at_j = maxlen - (dist_from_depot[:, None] + dist_ij) # shape (n, n)
    
    # To visit a next node k from j and return to depot, we need:
    # dist(j, k) + dist(k, 0) <= budget_at_j[i, j]
    # We want to count how many nodes k satisfy this.
    
    # Precompute cost for each node j and next node k: cost_j_k_return[j, k] = dist(j, k) + dist(k, 0)
    cost_j_k_return = distance + dist_to_depot[None, :] # shape (n, n)
    
    # Vectorized count using broadcasting:
    # cost_j_k_return has shape (n, n). Element [j, k] is the cost to go j->k->depot.
    # budget_at_j has shape (n, n). Element [i, j] is the budget available at j after coming from i.
    # We want to count, for each pair (i, j), how many k satisfy cost_j_k_return[j, k] <= budget_at_j[i, j].
    
    # Expand dimensions to align:
    # cost_j_k_return: shape (1, n, n) -> broadcasts over i
    # budget_at_j:    shape (n, n, 1) -> broadcasts over k
    # Comparison:
    # feasible_mask[i, j, k] = cost_j_k_return[0, j, k] <= budget_at_j[i, j, 0]
    
    feasible_mask = cost_j_k_return[np.newaxis, :, :] <= budget_at_j[:, :, np.newaxis]
    
    # Sum over k (axis 2) to get the count of feasible neighbors for each (i, j)
    neighbor_count = np.sum(feasible_mask, axis=2) # shape (n, n)
    
    # Normalize neighbor_count to [1, 2] scale to use as a multiplicative factor
    max_count = np.max(neighbor_count)
    if max_count > 0:
        norm_neighbor_score = 1.0 + (neighbor_count / max_count)
    else:
        norm_neighbor_score = 1.0
    
    # Penalize nodes with low branching factor (fewer future options)
    
    # 3. Angular Diversification Term
    # Base angular penalty for moving towards depot (cos_theta > 0)
    cos_pos = np.maximum(cos_theta, 0.0)
    
    alpha_div = 1.5
    
    # Scale factor for angular part based on budget progress
    budget_ratio_global = safe_slack / maxlen
    k = 0.5
    scale_factor = 1.0 + k * (2.0 * budget_ratio_global - 1.0)
    scale_factor = np.maximum(scale_factor, 0.1)
    
    angular_part = np.exp(-alpha_div * scale_factor * cos_pos)
    
    # Combine angular diversification and neighbor accessibility
    diversification_factor = angular_part * norm_neighbor_score
    
    heuristic *= diversification_factor

    # Apply heavy penalty for infeasible edges
    heuristic = np.where(infeasible, 1e-10, heuristic)
    
    # Ensure no NaN or Inf
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.maximum(heuristic, 1e-12)
    
    # Return the heuristic matrix
    return heuristic
