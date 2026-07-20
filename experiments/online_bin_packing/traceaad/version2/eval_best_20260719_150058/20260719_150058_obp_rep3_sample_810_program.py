import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Create a copy of bins to avoid modifying input and ensure float operations
    bin_capacities = np.array(bins, dtype=float)
    
    # Calculate remaining capacity after inserting item
    remaining = bin_capacities - item
    
    # Create a mask for bins where the item fits
    fits_mask = remaining >= 0
    
    # Initialize priorities with -infinity
    priorities = np.full_like(bin_capacities, -np.inf, dtype=float)
    
    # For bins where item fits, calculate priority using adaptive geometric target slack
    if np.any(fits_mask):
        bin_caps_fits = bin_capacities[fits_mask]
        remaining_fits = remaining[fits_mask]
        
        # Calculate target slack: item * (capacity / item - 1) ** 0.5
        # Note: capacity / item >= 1 since item fits, so the term inside sqrt is >= 0
        ratio = bin_caps_fits / item
        target_slack = item * (ratio - 1) ** 0.5
        
        # Priority is negative squared relative difference from target slack
        # Higher priority (less negative) is better
        # Normalize by target_slack to ensure consistent penalty magnitude across scales
        relative_error = (remaining_fits - target_slack) / np.maximum(target_slack, 1e-6)
        primary_priority = -relative_error ** 2
        
        # Secondary sorting term: favor bins with lower absolute remaining capacity
        # when relative error is within tolerance.
        # This breaks ties by preferring tighter fits once the geometric target slack constraint is satisfied.
        tolerance = 0.1
        tight_mask = np.abs(relative_error) < tolerance
        
        # Initialize secondary term to 0
        secondary_term = np.zeros_like(remaining_fits)
        
        # For bins within tolerance, subtract a scaled remaining capacity
        # This makes bins with smaller remaining capacity have higher priority (less subtraction)
        # Scale factor chosen to be small enough not to overwhelm primary priority but large enough to break ties
        secondary_scale = 1e-4
        secondary_term[tight_mask] = -secondary_scale * remaining_fits[tight_mask]
        
        # Simplified bonus: rewards bins with small remaining capacity (tight fits)
        # Bonus is inversely proportional to remaining capacity
        bonus = 1.0 / np.maximum(remaining_fits, 1e-6)
        
        # Scale bonus to have comparable magnitude to primary_priority
        # Primary priority ranges from -infinity to 0 (typically small negative values)
        # Scale bonus down to avoid overwhelming the primary metric
        bonus_scale = 0.1
        
        # Combine all terms
        final_priority = primary_priority + bonus_scale * bonus + secondary_term
        
        priorities[fits_mask] = final_priority
        
    return priorities
