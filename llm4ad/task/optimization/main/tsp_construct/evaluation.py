# Module Name: TSPEvaluation
# Last Revision: 2025/2/16
# Description: Evaluates the constructive heuristic for Traveling Salseman Problem (TSP).
#              Given a set of locations,
#              the goal is to find optimal route to travel all locations and back to start point
#              while minimizing the total travel distance.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 30).
#    - split: Fixed dataset split used for evaluation. Search should use train; final reporting can use test splits.
#
# 
# References:
#   - Fei Liu, Xialiang Tong, Mingxuan Yuan, and Qingfu Zhang. 
#     "Algorithm Evolution using Large Language Model." arXiv preprint arXiv:2311.15249 (2023).
# 
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------
from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from llm4ad.base import Evaluation, TextFunctionProgramConverter
from llm4ad.task.optimization.main.tsp_construct.dataset import (
    DEFAULT_SPLIT,
    load_split_instances,
)
from llm4ad.task.optimization.main.tsp_construct.template import template_program, task_description

__all__ = ['TSPEvaluation']

_PROCESS_EVA = None
_PROCESS_DATASETS = None
_PROCESS_NEIGHBOR_MATRICES = None
_PROCESS_PROBLEM_SIZE = None


def _tour_cost(instance, solution, problem_size):
    cost = 0
    for j in range(problem_size - 1):
        cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
    cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
    return cost


def _evaluate_instance(eva: callable, dataset_entry, neighbor_matrix, problem_size) -> float | None:
    instance, distance_matrix = dataset_entry
    destination_node = 0
    current_node = 0

    route = np.zeros(problem_size)
    # print(">>> Step 0 : select node "+str(instance[0][0])+", "+str(instance[0][1]))
    for i in range(1, problem_size - 1):
        near_nodes = neighbor_matrix[current_node][1:]
        mask = ~np.isin(near_nodes, route[:i])
        unvisited_near_nodes = near_nodes[mask]
        next_node = eva(current_node, destination_node, unvisited_near_nodes, distance_matrix)

        if next_node in route:
            # print("wrong algorithm select duplicate node, retrying ...")
            return None

        current_node = next_node
        route[i] = current_node

    mask = ~np.isin(np.arange(problem_size), route[:problem_size - 1])
    last_node = np.arange(problem_size)[mask]
    current_node = last_node[0]
    route[problem_size - 1] = current_node

    return _tour_cost(instance, route, problem_size)


def _init_process_worker(program_str, function_name, datasets, neighbor_matrices, problem_size):
    global _PROCESS_EVA, _PROCESS_DATASETS, _PROCESS_NEIGHBOR_MATRICES, _PROCESS_PROBLEM_SIZE
    namespace = {}
    exec(program_str, namespace)
    _PROCESS_EVA = namespace[function_name]
    _PROCESS_DATASETS = datasets
    _PROCESS_NEIGHBOR_MATRICES = neighbor_matrices
    _PROCESS_PROBLEM_SIZE = problem_size


def _evaluate_process_index(index: int) -> float | None:
    return _evaluate_instance(
        _PROCESS_EVA,
        _PROCESS_DATASETS[index],
        _PROCESS_NEIGHBOR_MATRICES[index],
        _PROCESS_PROBLEM_SIZE,
    )


class TSPEvaluation(Evaluation):
    """Evaluator for traveling salesman problem."""

    def __init__(self,
                 timeout_seconds=30,
                 split: str = DEFAULT_SPLIT,
                 eval_workers: int = 1,
                 eval_backend: str = "sequential"):

        """
            Args:
                None
            Raises:
                AttributeError: If the data key does not exist.
                FileNotFoundError: If the specified data file is not found.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self._datasets, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.problem_size = int(self.dataset_metadata["problem_size"])
        self.eval_workers = max(1, int(eval_workers))
        if eval_backend not in {"sequential", "thread", "process"}:
            raise ValueError("eval_backend must be one of: sequential, thread, process")
        if eval_backend == "process" and self.daemon_eval_process:
            raise ValueError("process eval_backend is incompatible with daemon_eval_process=True")
        self.eval_backend = eval_backend
        self._neighborhood_matrices = [
            self.generate_neighborhood_matrix(instance)
            for instance, _ in self._datasets
        ]

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        if self.eval_backend == "process" and self.eval_workers > 1:
            return self._evaluate_program_with_processes(program_str)
        return self.evaluate(callable_func)

    def tour_cost(self, instance, solution, problem_size):
        return _tour_cost(instance, solution, problem_size)

    def generate_neighborhood_matrix(self, instance):
        instance = np.array(instance)
        n = len(instance)
        neighborhood_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            distances = np.linalg.norm(instance[i] - instance, axis=1)
            sorted_indices = np.argsort(distances)  # sort indices based on distances
            neighborhood_matrix[i] = sorted_indices

        return neighborhood_matrix

    def _evaluate_instance(self, eva: callable, dataset_entry, neighbor_matrix) -> float | None:
        return _evaluate_instance(eva, dataset_entry, neighbor_matrix, self.problem_size)

    def evaluate(self, eva: callable) -> float:
        if self.eval_backend != "thread" or self.eval_workers == 1 or self.n_instance <= 1:
            distances = [
                self._evaluate_instance(eva, dataset_entry, neighbor_matrix)
                for dataset_entry, neighbor_matrix in zip(self._datasets, self._neighborhood_matrices)
            ]
        else:
            max_workers = min(self.eval_workers, self.n_instance)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                distances = list(executor.map(
                    lambda args: self._evaluate_instance(eva, *args),
                    zip(self._datasets, self._neighborhood_matrices),
                ))

        if any(distance is None for distance in distances):
            return None

        ave_dis = np.average(distances)
        # print("average dis: ",ave_dis)
        return -ave_dis

    def _evaluate_program_with_processes(self, program_str: str) -> float | None:
        function_name = TextFunctionProgramConverter.text_to_function(program_str).name
        max_workers = min(self.eval_workers, self.n_instance)
        try:
            context = multiprocessing.get_context("fork")
        except ValueError:
            context = multiprocessing.get_context()

        pool = context.Pool(
            processes=max_workers,
            initializer=_init_process_worker,
            initargs=(
                program_str,
                function_name,
                self._datasets,
                self._neighborhood_matrices,
                self.problem_size,
            ),
        )
        try:
            async_result = pool.map_async(_evaluate_process_index, range(self.n_instance))
            timeout = None
            if self.timeout_seconds is not None:
                timeout = max(float(self.timeout_seconds) - 1.0, 0.1)
            distances = async_result.get(timeout=timeout)
            pool.close()
            pool.join()
        except multiprocessing.TimeoutError:
            pool.terminate()
            pool.join()
            return None
        except Exception:
            pool.terminate()
            pool.join()
            raise

        if any(distance is None for distance in distances):
            return None

        return -float(np.average(distances))


if __name__ == '__main__':
    import sys

    print(sys.path)


    def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
        """
        Design a novel algorithm to select the next node in each step.

        Args:
        current_node: ID of the current node.
        destination_node: ID of the destination node.
        unvisited_nodes: Array of IDs of unvisited nodes.
        distance_matrix: Distance matrix of nodes.

        Return:
        ID of the next node to visit.
        """
        distances_to_destination = distance_matrix[current_node][unvisited_nodes]

        # Find the index of the unvisited node with the smallest distance to the destination
        next_node_index = np.argmin(distances_to_destination)

        # Get the ID of the next node to visit
        next_node = unvisited_nodes[next_node_index]

        return next_node


    tsp = TSPEvaluation()
    tsp.evaluate_program('_', select_next_node)
