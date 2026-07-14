import numpy as np


# Standard OP max tour length (budget) in the unit square, following
# Kool et al. (2019) / DeepACO / ReEvo: OP50/100/200/500/1000 = 3/4/5/8/12.
# For non-standard problem sizes, fall back to `max_length_ratio * problem_size`.
_STANDARD_OP_MAX_LENGTH = {50: 3.0, 100: 4.0, 200: 5.0, 500: 8.0, 1000: 12.0}


class GetData:
    def __init__(
            self,
            n_instance: int,
            problem_size: int,
            max_length_ratio: float = 0.35,
            seed: int = 2024,
    ):
        self.n_instance = int(n_instance)
        self.problem_size = int(problem_size)
        self.max_length_ratio = float(max_length_ratio)
        self.seed = int(seed)

    def generate_instances(self):
        rng = np.random.default_rng(self.seed)
        instance_data = []

        for _ in range(self.n_instance):
            coordinates = rng.random((self.problem_size, 2))
            coordinates[0] = np.array([0.5, 0.5])

            distances = np.linalg.norm(
                coordinates[:, np.newaxis] - coordinates,
                axis=2,
            )

            # Kool et al. (2019) / DeepACO / ReEvo discrete prize distribution:
            #   p_i = (1 + floor(99 * d_{0i} / max_j d_{0j})) / 100,  depot prize 0.
            depot_distance = distances[0]
            max_depot_distance = float(depot_distance[1:].max())
            if max_depot_distance > 0:
                prizes = (1.0 + np.floor(99.0 * depot_distance / max_depot_distance)) / 100.0
            else:
                prizes = np.ones(self.problem_size)
            prizes[0] = 0.0

            # Budget: ReEvo/DeepACO standard value for canonical sizes
            # (OP50=3, OP100=4, OP200=5, ...); legacy ratio * size otherwise.
            max_length = _STANDARD_OP_MAX_LENGTH.get(self.problem_size)
            if max_length is None:
                max_length = float(self.max_length_ratio * self.problem_size)
            max_length = float(max_length)

            instance_data.append({
                "coordinates": coordinates,
                "distance_matrix": distances,
                "prizes": prizes,
                "start_node": 0,
                "end_node": 0,
                "max_length": max_length,
            })

        return instance_data
