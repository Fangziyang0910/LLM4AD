template_program = '''
import numpy as np
from typing import List, Optional, Tuple

def determine_next_assignment(remaining_items: List[int], remaining_capacities: List[int]) -> Tuple[int, Optional[int]]:
    """
    Determine the next item and bin to pack based on a greedy heuristic.

    Args:
        remaining_items: A list of remaining item weights.
        remaining_capacities: Remaining capacity of every available bin. The
            selected item must fit in the selected bin.

    Returns:
        A tuple containing:
        - The selected item to pack.
        - The selected bin to pack the item into (or None if no feasible bin is found).
    """
    # Iterate through items in their original order
    for item in remaining_items:
        # Iterate through bins to find the first feasible one
        for bin_id, capacity in enumerate(remaining_capacities):
            if item <= capacity:
                return item, bin_id  # Return the selected item and bin
    return remaining_items[0], None  # If no feasible bin is found, return the first item and no bin
'''

task_description = '''
Given item weights and the remaining capacity of every available bin, iteratively
choose one remaining item and a bin that can hold it. Design the constructive
selection heuristic with the objective of minimizing the number of used bins.
'''
