
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    bins_float = bins.astype(np.float64)
    item_float = float(item)
    
    # Handle edge case where item is 0
    if item_float == 0:
        return bins_float * 100.0
        
    # Calculate remaining capacity after placing the item
    remaining_after = bins_float - item_float
    
    # 1. Rational Exact-Fit Bonus (Inspired by No.2)
    # Creates a sharp peak at remaining_after = 0.
    eps_exact = 1e-4
    exact_bonus = 100.0 / (remaining_after**2 + eps_exact)
    
    # 2. Harmonic Resonance Term (Inspired by No.2)
    # Rewards bins where remaining space is an integer multiple of item size.
    ratio = remaining_after / item_float
    rounded_ratio = np.round(ratio)
    dist_to_int = np.abs(ratio - rounded_ratio)
    
    # Sharper peak for integer multiples
    resonance_score = 1.0 / (1.0 + (dist_to_int ** 2) * 30.0)
    
    # 3. Fragment Penalty (Inspired by No.1)
    # Penalize small remainders using a smooth quadratic-exponential decay.
    # This mimics No.1's approach: P(r) = A * r^2 * exp(-B * r)
    # This penalty is 0 at r=0 (protected by exact_bonus) and grows for small r.
    A_frag = 10.0
    B_frag = 0.3
    fragment_penalty = A_frag * (remaining_after ** 2) * np.exp(-B_frag * remaining_after)
    
    # 4. Waste Penalty (Inspired by No.2)
    # Penalize large remainders quadratically to prefer tighter fits.
    norm_rem = remaining_after / item_float
    waste_penalty = 0.5 * (norm_rem ** 2)
    
    # 5. Base Score (Inspired by No.1)
    # Best Fit logic: prefer smaller remainders.
    # We subtract a scaled version of remaining to bias towards tighter fits.
    base_score = -0.1 * remaining_after
    
    # Combine scores
    # High priority for exact fits and resonant bins, low for fragments and waste.
    priority_score = exact_bonus + resonance_score + base_score - fragment_penalty - waste_penalty
    
    # Ensure finiteness
    assert np.all(np.isfinite(priority_score)), "Priority scores must be finite"
    
    return priority_score
