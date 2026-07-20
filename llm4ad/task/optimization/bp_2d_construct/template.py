template_program = '''
from typing import List, Optional, Tuple
import numpy as np

def determine_next_assignment(
    remaining_items: List[Tuple[int, int]],
    point_matrices: List[List[List[int]]],
) -> Tuple[Tuple[int, int], Optional[int]]:
    """Choose the next item and target bin for 2D bin packing.

    Called repeatedly while packing remaining rectangular items.

    Args:
        remaining_items: Items still to pack as (width, height).
        point_matrices: For each open bin, a 2-d occupancy grid where 0 is free
            and 1 is occupied. Grid shape is (bin_width, bin_height).

    Returns:
        (selected_item, bin_index). selected_item must be one of remaining_items.
        bin_index is the chosen open bin, or None if a new bin must be opened.
    """
    selected_item = max(remaining_items, key=lambda item: item[0] * item[1])

    for bin_idx, point_matrix in enumerate(point_matrices):
        bin_width = len(point_matrix)
        bin_height = len(point_matrix[0]) if bin_width > 0 else 0
        if bin_width >= selected_item[0] and bin_height >= selected_item[1]:
            for x in range(bin_width - selected_item[0] + 1):
                for y in range(bin_height - selected_item[1] + 1):
                    if all(
                        point_matrix[x + dx][y + dy] == 0
                        for dx in range(selected_item[0])
                        for dy in range(selected_item[1])
                    ):
                        return selected_item, bin_idx
    return selected_item, None
'''

task_description = (
    "Design a constructive heuristic for 2D bin packing. "
    "At each step choose one remaining rectangle and either place it into an existing "
    "bin or request a new bin. The function receives remaining items as (width, height) "
    "and each open bin as an occupancy grid (0 free, 1 occupied). It must return "
    "(item, bin_index) where item is one of the remaining items and bin_index is an "
    "existing bin index or None to open a new bin. The objective is to minimize the "
    "number of bins; the evaluator returns the negated average bin count."
)
