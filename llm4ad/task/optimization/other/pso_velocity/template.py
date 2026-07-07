template_program = '''
import numpy as np


def update_velocity(
    velocities: np.ndarray,
    positions: np.ndarray,
    pbest_positions: np.ndarray,
    pbest_fitness: np.ndarray,
    gbest_position: np.ndarray,
    gbest_fitness: float,
    w: float,
    c1: float,
    c2: float,
    bounds: np.ndarray,
    iteration: int,
    max_iterations: int,
) -> np.ndarray:
    """Return updated PSO velocities.

    Args:
        velocities: current velocity array with shape (pop_size, dim).
        positions: current particle positions with shape (pop_size, dim).
        pbest_positions: personal best position per particle.
        pbest_fitness: personal best objective value per particle, lower is better.
        gbest_position: global best position found so far.
        gbest_fitness: global best objective value found so far.
        w: inertia weight.
        c1: cognitive coefficient.
        c2: social coefficient.
        bounds: array with shape (dim, 2), [lower, upper] per variable.
        iteration: current zero-based PSO iteration.
        max_iterations: total PSO iterations.

    Returns:
        New velocities with the same shape as velocities.
    """
    pop_size, dim = velocities.shape
    r1 = np.random.rand(pop_size, dim)
    r2 = np.random.rand(pop_size, dim)
    cognitive = c1 * r1 * (pbest_positions - positions)
    social = c2 * r2 * (gbest_position - positions)
    return w * velocities + cognitive + social
'''

task_description = (
    "Design a velocity update rule for Particle Swarm Optimization. The function "
    "receives the current swarm velocities, positions, personal bests, global best, "
    "standard PSO coefficients, bounds, and iteration progress, and must return a "
    "velocity array of the same shape. The baseline is standard PSO: inertia plus "
    "cognitive and social attraction. Performance is measured by final objective "
    "value on Sphere, Rastrigin, Ackley, Rosenbrock, and Griewank minimization benchmarks."
)
