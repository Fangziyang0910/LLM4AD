import numpy as np


def generate_weibull_dataset(
    num_instances,
    num_items,
    capacity_limit,
    seed=2024,
    max_item_size: int = 100,
):
    """Generate Weibull OBP instances.

    Item sizes are clipped at ``max_item_size`` (paper default 100), independent
    of bin ``capacity_limit``. This matches EoH / ReEvo / MCTS-AHD / PathWise,
    where capacity-500 tests still use items clipped at 100.
    """
    np.random.seed(seed)

    dataset = {}

    for i in range(num_instances):
        instance = {
            'capacity': capacity_limit,
            'num_items': num_items,
            'items': []
        }

        items = []

        # Generate random samples from Weibull(45, 3) distribution
        samples = np.random.weibull(3, num_items) * 45

        # Clip item sizes independently of bin capacity (paper protocol)
        samples = np.clip(samples, 1, max_item_size)

        # Round the item sizes to the nearest integer
        sizes = np.round(samples).astype(int)

        # Add the items to the instance
        for size in sizes:
            items.append(size)

        instance['items'] = np.array(items)

        if num_items not in dataset:
            dataset[f'instance_{i}'] = instance

    return dataset
