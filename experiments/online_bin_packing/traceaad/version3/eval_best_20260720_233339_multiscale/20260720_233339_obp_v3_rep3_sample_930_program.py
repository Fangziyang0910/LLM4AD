import numpy as np

# Global state to keep track of recent item sizes for look-ahead estimation
class LookAheadStats:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.history = []
        self.mean = 1.0
        self.std = 0.1
        
    def update(self, item_size):
        self.history.append(item_size)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        if len(self.history) > 1:
            h = np.array(self.history)
            self.mean = np.mean(h)
            self.std = np.std(h)
        else:
            self.mean = self.history[-1] if self.history else 1.0
            self.std = max(0.1, self.mean * 0.1) if self.history else 0.1

look_ahead_stats = LookAheadStats(window_size=20)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Update look-ahead stats with the current item
    look_ahead_stats.update(item)
    
    # Calculate remainders if item is placed in each bin
    remainders = bins - item
    
    # Filter out bins where item doesn't fit (remainder < 0)
    # We'll set their priority to -infinity so they are never chosen
    feasible = remainders >= 0
    
    # Initialize priorities with -inf
    priorities = np.full_like(remainders, -np.inf, dtype=float)
    
    if not np.any(feasible):
        return priorities
    
    # Focus only on feasible bins
    rem = remainders[feasible]
    
    eps = 1e-8
    tau = 0.1  # Threshold multiplier for small remainders
    
    # 1. Smooth Piecewise Base Score:
    # Blend between inverse-squared (for small remainders) and inverse (for large remainders)
    # using a smooth transition function.
    threshold = tau * item
    
    # Calculate the two potential scores
    # Score 1: Strong penalty for tiny fragments (inverse squared)
    score_tight = 1.0 / (rem**2 + eps)
    
    # Score 2: Standard penalty (inverse)
    score_loose = 1.0 / (rem + eps)
    
    # Smooth transition function (sigmoid-like)
    # We want weight 1 for tight (rem < threshold) and weight 0 for loose (rem > threshold)
    
    if threshold == 0:
        threshold = eps
        
    # Scale factor for steepness
    alpha = 5.0
    
    # Calculate exponent argument
    try:
        exp_arg = alpha * (rem - threshold) / threshold
        # Clamp exp_arg to reasonable range to avoid overflow/underflow in exp
        exp_arg_clamped = np.clip(exp_arg, -10, 10)
        weight_tight = 1.0 / (1.0 + np.exp(exp_arg_clamped))
    except Exception:
        weight_tight = np.ones_like(rem)
        
    weight_loose = 1.0 - weight_tight
    
    # Blended base score
    base_score = weight_tight * score_tight + weight_loose * score_loose
    
    # 2. Symmetry Score:
    # Penalize remainders that are close to half the max available gap.
    max_rem = np.max(rem) if np.any(rem > 0) else 1.0
    norm_rem = rem / (max_rem + eps)
    
    # Sine wave penalty term
    symmetry_score = -0.5 * np.sin(2 * np.pi * norm_rem)
    
    # 3. Dynamic Hyperbolic Penalty:
    # Penalty = k / (remainder + delta)
    
    # Calculate k: scaled by item size and minimum remainder threshold
    min_remainder_threshold = 0.01 * item
    k = max(0.01 * item, min_remainder_threshold)
    
    # Delta parameter to control penalty softness
    delta = 0.01 * max_rem + eps
    
    # Calculate the base hyperbolic penalty
    hyperbolic_penalty_base = k / (rem + delta)
    
    # 4. Harmonic Resonance Factor:
    # Transplant a multiplicative decay factor for the harmonic resonance term 
    # 1 / (1 + gamma * abs(remainder - item))
    # Applied only when epsilon_item < remainder < item
    
    epsilon_item = 0.01 * item
    gamma = 10.0
    
    # Initialize resonance factor to 1.0 (no change) for all bins
    resonance_factor = np.ones_like(rem)
    
    # Mask for bins where epsilon_item < remainder < item
    resonance_mask = (rem > epsilon_item) & (rem < item)
    
    if np.any(resonance_mask):
        # Calculate harmonic resonance term for the mask
        abs_diff = np.abs(rem - item)
        resonance_term = 1.0 / (1.0 + gamma * abs_diff)
        
        # Apply the mask: only update resonance_factor where the condition holds
        resonance_factor[resonance_mask] = resonance_term[resonance_mask]
        
    # Apply resonance factor to the hyperbolic penalty
    final_hyperbolic_penalty = hyperbolic_penalty_base * resonance_factor
    
    # 5. Look-Ahead Compatibility Term (Modified):
    # Use a multi-modal density estimator (Kernel Density Estimation) based on history.
    
    history = look_ahead_stats.history
    
    # If no history, fallback to a small constant or zero bonus
    if not history:
        compat_bonus = 0.0
    else:
        history_arr = np.array(history)
        # Bandwidth for the KDE.
        if len(history_arr) > 1 and look_ahead_stats.std > 0:
            h_kde = look_ahead_stats.std * 1.06 * (len(history_arr) ** (-0.2))
            h_kde = max(h_kde, eps)
        else:
            h_kde = max(0.1 * item, eps)
            
        # Calculate KDE value for each remainder
        try:
            diff = rem[:, None] - history_arr[None, :]
            z = diff / h_kde
            pdf_terms = np.exp(-0.5 * z**2)
            sum_pdf = np.sum(pdf_terms, axis=1)
            
            n = len(history_arr)
            normalizer = n * h_kde * np.sqrt(2 * np.pi)
            kde_values = sum_pdf / normalizer
            
            look_ahead_weight = 100.0
            compat_bonus = look_ahead_weight * kde_values
            
        except Exception:
            compat_bonus = np.zeros_like(rem)

    # 6. Fragment Danger Penalty:
    # Penalize remainders that are larger than min_history but smaller than second_min_history.
    # These gaps are "dangerous" because they can only fit the rarest smallest items.
    
    fragment_penalty = np.zeros_like(rem)
    
    if len(history) >= 2:
        sorted_hist = np.sort(history)
        min_item = sorted_hist[0]
        second_min_item = sorted_hist[1]
        
        # Danger zone: min_item < rem < second_min_item
        danger_mask = (rem > min_item) & (rem < second_min_item)
        
        if np.any(danger_mask):
            # Penalty proportional to 1 / (remainder - min_item + eps)
            # This creates a strong penalty when remainder is just above min_item
            danger_diff = rem[danger_mask] - min_item
            # Scale factor to make penalty significant
            scale_factor = 50.0 
            penalty_val = scale_factor / (danger_diff + eps)
            fragment_penalty[danger_mask] = penalty_val

    # Combine scores
    # Priority = Base_Score + Symmetry_Score - Hyperbolic_Penalty + LookAhead_Compatibility - Fragment_Danger_Penalty
    priority_score = base_score + symmetry_score - final_hyperbolic_penalty + compat_bonus - fragment_penalty
    
    priorities[feasible] = priority_score
    
    return priorities
