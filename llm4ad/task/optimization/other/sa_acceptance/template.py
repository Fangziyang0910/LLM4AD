template_program = '''
import numpy as np


def acceptance_probability(delta_fitness: float, temperature: float,
                           iteration: int, max_iterations: int) -> float:
    """Return the probability of accepting a worse SA candidate.

    Args:
        delta_fitness: f(candidate) - f(current), always positive here.
        temperature: current annealing temperature.
        iteration: current zero-based SA iteration.
        max_iterations: total number of SA iterations.

    Returns:
        A probability in [0, 1]. The evaluator clips the returned value to
        [0, 1] before sampling the acceptance decision.
    """
    return float(np.exp(-delta_fitness / max(temperature, 1e-10)))
'''

task_description = (
    "Design an acceptance probability function for Simulated Annealing on "
    "continuous minimization benchmarks. The function is called only when a "
    "candidate solution is worse than the current solution, so delta_fitness is "
    "positive. It should return the probability of accepting the worse candidate. "
    "The baseline is the Boltzmann criterion exp(-delta_fitness / temperature). "
    "The temperature starts high and decreases geometrically by a factor of 1000 "
    "over the run. Performance is measured by the average final objective value "
    "across Sphere, Rastrigin, Ackley, Rosenbrock, and Griewank benchmark functions."
)
