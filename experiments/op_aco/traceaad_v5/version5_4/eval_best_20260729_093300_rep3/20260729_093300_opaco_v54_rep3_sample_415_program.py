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
    n = prize.shape[0]
    
    # Precompute return distances to depot (column 0) and from depot (row 0)
    dist_to_depot = distance[:, 0]  # shape (n,)
    dist_from_depot = distance[0]   # shape (n,)
    
    # Compute remaining budget for direct return after visiting j from i:
    # remaining[i,j] = maxlen - distance[i,j] - dist_to_depot[j]
    remaining = maxlen - distance - dist_to_depot[np.newaxis, :]  # shape (n, n)
    
    # Clip remaining budget for computation
    remaining_clipped = np.clip(remaining, 0, None)  # shape (n, n)
    
    # Compute distance squared for normalization
    dist_sq = distance ** 2  # shape (n, n)
    
    # Avoid division by zero; set a small epsilon for zero distances
    dist_sq_safe = np.where(dist_sq < 1e-12, 1e-12, dist_sq)
    
    # Prize at destination j
    prize_dest = prize[np.newaxis, :]  # shape (1, n)
    
    # Prize at current node i
    prize_src = prize[:, np.newaxis]  # shape (n, 1)
    
    # --- Dynamic Slack-Aware Margin ---
    
    # Dynamic margin: Reflects actual budget residual after taking edge i->j.
    # Simplified as remaining * (1 - distance/maxlen) to maintain O(N^2) complexity.
    # This replaces the static margin: maxlen - dist_from_depot[i] - dist_to_depot[j]
    budget_ratio = distance / maxlen
    ratio_clipped = np.clip(budget_ratio, 0, 1)
    
    # The "slack-aware" margin combines the remaining budget after the hop
    # with a penalty factor that reduces value as the edge consumes more of the total budget.
    # This effectively creates a term proportional to remaining * (1 - dist/maxlen).
    dynamic_margin = remaining_clipped * (1 - ratio_clipped)
    
    # Prize Density Term:
    # Ratio of prize[j] / sqrt(dist_sq_safe[i,j]) to reward short hops to high-value nodes.
    prize_density = prize_dest / np.sqrt(dist_sq_safe)
    
    # Hybrid score: Combine dynamic margin, remaining, and prize density
    # Note: The static margin was previously multiplied by remaining_clipped.
    # Now dynamic_margin already incorporates remaining_clipped and the ratio penalty.
    # To maintain structural consistency, we treat dynamic_margin as the new margin component.
    # However, the original hybrid_score was: prize * margin * remaining / dist_sq * prize_density.
    # If we replace margin with dynamic_margin, we might double-count remaining if not careful.
    # Let's look at the structure:
    # Original: prize_dest * margin_clipped * remaining_clipped / dist_sq_safe * prize_density
    # New:      prize_dest * dynamic_margin * remaining_clipped / dist_sq_safe * prize_density
    # This seems redundant because dynamic_margin includes remaining_clipped.
    # Let's refine: The requested change is to replace margin_clipped with a dynamic slack term.
    # The suggestion is `remaining * (1 - distance/maxlen)`.
    # Let's use this term directly in place of margin_clipped * remaining_clipped effectively,
    # or just replace margin_clipped with a term that scales with remaining.
    # Let's stick to the prompt's simplification: "simplified to `remaining * (1 - distance/maxlen)`".
    # This term itself represents the effective "weighted remaining budget".
    # So, Hybrid Score = prize_dest * [dynamic_slack_term] / dist_sq_safe * prize_density
    # where dynamic_slack_term = remaining_clipped * (1 - ratio_clipped)
    
    hybrid_score = prize_dest * dynamic_margin / dist_sq_safe * prize_density
    
    # Apply multiplicative budget penalty factor from reference
    # Quadratic penalty (1 - ratio)^2 to aggressively penalize long edges
    # Note: We already used (1 - ratio) in dynamic_margin.
    # The reference used a separate penalty_factor.
    # To avoid over-penalizing (power 3 or 4 on the ratio term), 
    # we should likely remove the separate penalty_factor or adjust it.
    # However, the prompt asks to replace margin in hybrid_score.
    # The existing code has a separate `penalty_factor` applied multiplicatively later.
    # Let's keep the structure but ensure the dynamic margin doesn't conflict.
    # The dynamic_margin has (1-ratio)^1. The penalty_factor has (1-ratio)^2.
    # Total effect on ratio: (1-ratio)^3.
    # Original: margin (static) * remaining * (1-ratio)^2.
    # New: remaining * (1-ratio) * remaining * (1-ratio)^2 = remaining^2 * (1-ratio)^3.
    # This changes the scaling significantly. 
    # Let's assume the "hybrid_score" in the previous code included the margin and remaining.
    # If we replace margin_clipped with `remaining * (1-ratio)`, we should probably 
    # NOT multiply by `remaining_clipped` again in hybrid_score to avoid squaring remaining.
    
    # Revised Hybrid Score construction:
    # Old: prize * margin_clipped * remaining_clipped / dist_sq * prize_density
    # New: prize * dynamic_margin / dist_sq * prize_density
    # where dynamic_margin = remaining_clipped * (1 - ratio_clipped)
    
    # We still apply the quadratic penalty factor separately as per existing structure?
    # If so, the total power of (1-ratio) becomes 3.
    # Let's check the coherence and other terms.
    
    # Coherence term (Modified):
    # Bias towards high-value clusters relative to current node's depot distance.
    # Combined prize of current node i and destination node j.
    combined_prize = prize_src + prize_dest  # shape (n, n)
    
    # Normalize by total path length from depot to destination (dist_from_depot[i] + distance[i,j])
    # This prevents overestimation of edges that are locally short but globally expensive relative to the depot.
    eps = 1e-10
    # Total path length from depot to j via i: dist_from_depot[i] + distance[i,j]
    total_path_len = dist_from_depot[:, np.newaxis] + distance  # shape (n, n)
    total_path_len_safe = np.maximum(total_path_len, eps)
    coherence = combined_prize / total_path_len_safe  # shape (n, n)
    
    # Apply multiplicative budget penalty factor from reference
    # Quadratic penalty (1 - ratio)^2 to aggressively penalize long edges
    # Keeping this as is, resulting in total (1-ratio)^3 interaction if dynamic_margin uses (1-ratio)^1.
    penalty_factor = (1 - ratio_clipped) ** 2
    
    # Combine hybrid score, coherence, and penalty factor multiplicatively
    dynamic_component = hybrid_score * coherence * penalty_factor
    
    # Lookahead efficiency term (from Primary/Reference):
    # Estimated potential of destination node j = prize[j] / (distance[j,0] + eps)
    # Multiply by current remaining budget margin.
    eps_lookahead = 1e-6
    return_efficiency = prize_dest / (dist_to_depot[np.newaxis, :] + eps_lookahead)  # shape (1, n)
    lookahead_term = return_efficiency * remaining_clipped  # shape (n, n)
    
    # Combine dynamic component (multiplicative structure) with lookahead term (additive bonus)
    heuristic = dynamic_component + lookahead_term
    
    # Set infeasible edges (where remaining < 0) to a very small value
    # These edges cannot be part of a feasible tour if we return directly after visiting j
    heuristic = np.where(remaining < 0, 1e-9, heuristic)
    
    # Ensure all values are positive (larger = more desirable)
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Set diagonal to 0 to avoid self-loops
    np.fill_diagonal(heuristic, 0.0)
    
    return heuristic
