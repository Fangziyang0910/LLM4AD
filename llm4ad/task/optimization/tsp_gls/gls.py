from __future__ import annotations

import random
import time
from typing import Callable

import numpy as np


def tour_cost_2end(dis_m: np.ndarray, tour2end: np.ndarray) -> float:
    cost = 0.0
    start = 0
    end = int(tour2end[0, 1])
    for _ in range(tour2end.shape[0]):
        cost += float(dis_m[start, end])
        start = end
        end = int(tour2end[start, 1])
    return cost


def nearest_neighbor_2end(dis_matrix: np.ndarray, depot: int) -> np.ndarray:
    tour = [depot]
    n = len(dis_matrix)
    nodes = np.arange(n)
    while len(tour) < n:
        current = tour[-1]
        neighbours = [(j, dis_matrix[current, j]) for j in nodes if j not in tour]
        next_node, _ = min(neighbours, key=lambda e: e[1])
        tour.append(int(next_node))
    tour.append(depot)

    route2end = np.zeros((n, 2), dtype=int)
    route2end[0, 0] = tour[-2]
    route2end[0, 1] = tour[1]
    for i in range(1, n):
        route2end[tour[i], 0] = tour[i - 1]
        route2end[tour[i], 1] = tour[i + 1]
    return route2end


def two_opt(tour: np.ndarray, i: int, j: int) -> np.ndarray:
    if i == j:
        return tour
    a = int(tour[i, 0])
    b = int(tour[j, 0])
    tour[i, 0] = tour[i, 1]
    tour[i, 1] = j
    tour[j, 0] = i
    tour[a, 1] = b
    tour[b, 1] = tour[b, 0]
    tour[b, 0] = a
    c = int(tour[b, 1])
    while tour[c, 1] != j:
        d = int(tour[c, 0])
        tour[c, 0] = tour[c, 1]
        tour[c, 1] = d
        c = d
    return tour


def two_opt_cost(tour: np.ndarray, distances: np.ndarray, i: int, j: int) -> float:
    if i == j:
        return 0.0
    a = int(tour[i, 0])
    b = int(tour[j, 0])
    return float(distances[a, b] + distances[i, j] - distances[a, i] - distances[b, j])


def two_opt_a2a(
        tour: np.ndarray,
        distances: np.ndarray,
        nearest_indices: np.ndarray,
        first_improvement: bool = False,
        set_delta: float = 0.0,
) -> tuple[float, np.ndarray]:
    best_move = None
    best_delta = set_delta

    for i in range(0, len(tour) - 1):
        for j in nearest_indices[i]:
            j = int(j)
            if i in tour[j] or j in tour[i]:
                continue
            delta = two_opt_cost(tour, distances, i, j)
            if delta < best_delta and not np.isclose(0, delta):
                best_delta = delta
                best_move = i, j
                if first_improvement:
                    break

    if best_move is not None:
        return best_delta, two_opt(tour, *best_move)
    return 0.0, tour


def two_opt_o2a_all(
        tour: np.ndarray,
        distances: np.ndarray,
        nearest_indices: np.ndarray,
        i: int,
) -> tuple[float, np.ndarray]:
    best_delta = 0.0
    for j in nearest_indices[i]:
        j = int(j)
        if i in tour[j] or j in tour[i]:
            continue
        delta = two_opt_cost(tour, distances, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            tour = two_opt(tour, i, j)
    return best_delta, tour


def relocate(tour: np.ndarray, i: int, j: int) -> np.ndarray:
    a = int(tour[i, 0])
    b = int(tour[i, 1])
    tour[a, 1] = b
    tour[b, 0] = a

    d = int(tour[j, 1])
    tour[d, 0] = i
    tour[i, 0] = j
    tour[i, 1] = d
    tour[j, 1] = i
    return tour


def relocate_cost(tour: np.ndarray, distances: np.ndarray, i: int, j: int) -> float:
    if i == j:
        return 0.0
    a = int(tour[i, 0])
    b = i
    c = int(tour[i, 1])
    d = j
    e = int(tour[j, 1])
    return float(
        -distances[a, b]
        - distances[b, c]
        + distances[a, c]
        - distances[d, e]
        + distances[d, b]
        + distances[b, e]
    )


def relocate_o2a_all(
        tour: np.ndarray,
        distances: np.ndarray,
        nearest_indices: np.ndarray,
        i: int,
) -> tuple[float, np.ndarray]:
    best_delta = 0.0
    for j in nearest_indices[i]:
        j = int(j)
        if tour[j, 1] == i:
            continue
        delta = relocate_cost(tour, distances, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            tour = relocate(tour, i, j)
    return best_delta, tour


def relocate_a2a(
        tour: np.ndarray,
        distances: np.ndarray,
        nearest_indices: np.ndarray,
        first_improvement: bool = False,
        set_delta: float = 0.0,
) -> tuple[float, np.ndarray]:
    best_move = None
    best_delta = set_delta

    for i in range(0, len(tour) - 1):
        for j in nearest_indices[i]:
            j = int(j)
            if tour[j, 1] == i:
                continue
            delta = relocate_cost(tour, distances, i, j)
            if delta < best_delta and not np.isclose(0, delta):
                best_delta = delta
                best_move = i, j
                if first_improvement:
                    break

    if best_move is not None:
        return best_delta, relocate(tour, *best_move)
    return 0.0, tour


def route2tour(route: np.ndarray) -> list[int]:
    start = 0
    tour = []
    for _ in range(len(route)):
        tour.append(int(route[start, 1]))
        start = int(route[start, 1])
    return tour


def local_search(
        init_tour: np.ndarray,
        init_cost: float,
        distances: np.ndarray,
        nearest_indices: np.ndarray,
        first_improvement: bool = False,
) -> tuple[np.ndarray, float]:
    cur_route, cur_cost = init_tour, init_cost
    improved = True
    while improved:
        improved = False

        delta, new_tour = two_opt_a2a(cur_route, distances, nearest_indices, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour

        delta, new_tour = relocate_a2a(cur_route, distances, nearest_indices, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour

    return cur_route, cur_cost


def guided_local_search(
        edge_weight: np.ndarray,
        nearest_indices: np.ndarray,
        init_tour: np.ndarray,
        init_cost: float,
        time_limit: float,
        ite_max: int,
        perturbation_moves: int,
        update_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        first_improvement: bool = False,
) -> tuple[np.ndarray, float, int]:
    random.seed(2024)

    cur_route, cur_cost = local_search(
        init_tour,
        init_cost,
        edge_weight,
        nearest_indices,
        first_improvement,
    )
    best_route, best_cost = cur_route, cur_cost

    n_nodes = len(edge_weight[0])
    iter_i = 0
    edge_penalty = np.zeros((n_nodes, n_nodes), dtype=float)
    deadline = time.time() + time_limit

    while iter_i < ite_max and time.time() < deadline:
        for _ in range(perturbation_moves):
            cur_tour = route2tour(cur_route)
            edge_weight_guided = update_fn(edge_weight, np.array(cur_tour), edge_penalty)
            edge_weight_guided = np.asmatrix(edge_weight_guided)
            edge_weight_gap = edge_weight_guided - edge_weight

            for _topid in range(5):
                max_indices = np.argmin(-edge_weight_gap, axis=None)
                rows, columns = np.unravel_index(max_indices, edge_weight_gap.shape)

                edge_penalty[rows, columns] += 1
                edge_penalty[columns, rows] += 1
                edge_weight_gap[rows, columns] = 0
                edge_weight_gap[columns, rows] = 0

                for node in [rows, columns]:
                    delta, new_route = two_opt_o2a_all(
                        cur_route,
                        edge_weight_guided,
                        nearest_indices,
                        int(node),
                    )
                    if delta < 0:
                        cur_cost = tour_cost_2end(edge_weight, new_route)
                        cur_route = new_route
                    delta, new_route = relocate_o2a_all(
                        cur_route,
                        edge_weight_guided,
                        nearest_indices,
                        int(node),
                    )
                    if delta < 0:
                        cur_cost = tour_cost_2end(edge_weight, new_route)
                        cur_route = new_route

        cur_route, cur_cost = local_search(
            cur_route,
            cur_cost,
            edge_weight,
            nearest_indices,
            first_improvement,
        )
        cur_cost = tour_cost_2end(edge_weight, cur_route)

        if cur_cost < best_cost:
            best_route, best_cost = cur_route, cur_cost
        iter_i += 1

        if iter_i % 50 == 0:
            cur_route, cur_cost = best_route, best_cost

    return best_route, best_cost, iter_i


def solve_instance(
        opt_cost: float,
        dis_matrix: np.ndarray,
        time_limit: float,
        ite_max: int,
        perturbation_moves: int,
        update_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> float:
    try:
        init_tour = nearest_neighbor_2end(dis_matrix, 0).astype(int)
        init_cost = tour_cost_2end(dis_matrix, init_tour)
        nb = min(100, max(1, dis_matrix.shape[0] - 1))
        nearest_indices = np.argsort(dis_matrix, axis=1)[:, 1:nb + 1].astype(int)

        _, best_cost, _ = guided_local_search(
            dis_matrix,
            nearest_indices,
            init_tour,
            init_cost,
            time_limit,
            ite_max,
            perturbation_moves,
            update_fn,
            first_improvement=False,
        )
        gap = (best_cost / opt_cost - 1) * 100
    except Exception:
        gap = 1e10

    return float(gap)
