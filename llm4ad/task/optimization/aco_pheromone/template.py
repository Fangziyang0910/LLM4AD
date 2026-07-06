template_program = '''
import numpy as np


def update_pheromone(pheromone: np.ndarray, ant_tours: list, tour_costs: np.ndarray,
                     best_tour: np.ndarray, best_cost: float,
                     rho: float, iteration: int, max_iterations: int) -> np.ndarray:
    """Return the next ACO pheromone matrix.

    Args:
        pheromone: current pheromone matrix with shape (n, n).
        ant_tours: list of ant tours, each a city permutation.
        tour_costs: tour length for each ant, lower is better.
        best_tour: best tour found so far.
        best_cost: length of best_tour.
        rho: evaporation rate.
        iteration: current zero-based ACO iteration.
        max_iterations: total ACO iterations.

    Returns:
        Updated pheromone matrix with shape (n, n).
    """
    n = pheromone.shape[0]
    pheromone = (1.0 - rho) * pheromone
    for tour, cost in zip(ant_tours, tour_costs):
        delta = 1.0 / cost
        for i in range(n):
            u, v = int(tour[i]), int(tour[(i + 1) % n])
            pheromone[u, v] += delta
            pheromone[v, u] += delta
    return pheromone
'''

task_description = (
    "Design a pheromone update rule for Ant Colony Optimization on Euclidean TSP. "
    "The evaluator uses a fixed probabilistic ant construction rule and calls the "
    "designed function after every iteration with the current pheromone matrix, all "
    "ant tours and costs, the best-so-far tour and cost, evaporation rate, and "
    "iteration progress. The baseline is Ant System evaporation plus deposits from "
    "all ants proportional to inverse tour length. Performance is measured by average "
    "best tour length on fixed Euclidean TSP instances."
)
