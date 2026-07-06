template_program = '''
import numpy as np


def respond_to_change(population: np.ndarray, fitness: np.ndarray,
                      best_position: np.ndarray,
                      bounds: np.ndarray) -> np.ndarray:
    """Return a new population after the objective optimum moves.

    Args:
        population: current population adapted to the old objective, shape (pop_size, n_dims).
        fitness: old-objective fitness values, shape (pop_size,), higher is better.
        best_position: best old-objective individual, shape (n_dims,).
        bounds: array with lower and upper bounds, shape (2, n_dims).

    Returns:
        New population with the same shape as population.
    """
    sigma = (bounds[1] - bounds[0]).mean() * 0.1
    new_population = population + np.random.normal(0.0, sigma, population.shape)
    return np.clip(new_population, bounds[0], bounds[1])
'''

task_description = (
    "Design the response strategy used by an evolutionary algorithm when a dynamic "
    "objective changes. The function receives the population, stale fitness values, "
    "the best old position, and search bounds, then returns an updated population. "
    "The goal is to reduce mean tracking error, measured as the distance from the "
    "best-found individual to the true moving optimum after adaptation."
)
