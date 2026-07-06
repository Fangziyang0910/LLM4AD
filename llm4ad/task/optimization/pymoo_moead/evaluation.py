# Module Name: MOEAD_PYMOO_Evaluation
# Last Revision: 2025/07/14
# Description: Evaluates the Multi-objective problem using the MOEAD algorithm.
#              Problem settings are loaded from fixed dataset splits.
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 20).
#    - split: Fixed dataset split used for evaluation.
#
# References:
#   - Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
#       Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
#       with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
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

import copy
from typing import Callable, Any
import numpy as np
from pymoo.problems import get_problem

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.decomposition.tchebicheff import Tchebicheff

from llm4ad.base import Evaluation
from llm4ad.task.optimization.pymoo_moead.dataset import (
    DEFAULT_SPLIT,
    load_split_case,
)
from llm4ad.task.optimization.pymoo_moead.template import template_program, task_description

class MOEAD_PYMOO_Evaluation(Evaluation):
    def __init__(self,
                 timeout_seconds=100,
                 split: str = DEFAULT_SPLIT):
        """
        Parameter Description:
        This evaluator now receives a decomposition function via the evaluate_program interface.
        """
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.dataset_metadata = load_split_case(split=split)
        self.problem = get_problem(
            self.dataset_metadata["problem_name"],
            n_var=int(self.dataset_metadata["n_var"]),
            n_obj=int(self.dataset_metadata["n_obj"]),
        )
        self.ref_dirs = get_reference_directions(
            "das-dennis",
            int(self.dataset_metadata["n_obj"]),
            n_partitions=int(self.dataset_metadata["n_partitions"]),
        )
        self.n_gen = int(self.dataset_metadata["n_gen"])
        self.hv_ref = np.array(self.dataset_metadata["hv_ref"], dtype=float)
        self.hv_calculator = HV(ref_point=self.hv_ref)
        self.last_result = None


    def evaluate(self, decomposition_func: Callable = None) -> float:
        """
        Core evaluation method. Returns the evaluation score and stores detailed results in self.last_result.
        """
        class DecompAdapter:
            def __init__(self, func):
                self.func = func
            def do(self, F, weights, ideal_point, **kwargs):
                return self.func(F, weights=weights, ideal_point=ideal_point, **kwargs)

        decomposition = DecompAdapter(decomposition_func) if decomposition_func else Tchebicheff()

        hv_values = []
        results = []
        for seed in self.dataset_metadata["seeds"]:
            algorithm = MOEAD(
                ref_dirs=self.ref_dirs,
                n_neighbors=15,
                prob_neighbor_mating=0.7,
                decomposition=decomposition,
            )
            termination = get_termination("n_gen", self.n_gen)
            res = minimize(self.problem, algorithm, termination, seed=int(seed), verbose=False)
            hv_value = float(self.hv_calculator(res.opt.get("F")))
            hv_values.append(hv_value)
            results.append({"seed": int(seed), "hv": hv_value, "pareto_front": res.opt})

        mean_hv = float(np.mean(hv_values))
        self.last_result = {"hv": mean_hv, "runs": results}
        return -mean_hv


    def evaluate_program(self, program_str: str, callable_func: callable) -> Any:
        return self.evaluate(decomposition_func=callable_func)

    def plot_solutions(self, solutions):
        import matplotlib.pyplot as plt
        F = solutions.get("F")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], c='blue', s=30, alpha=0.5)
        ax.set_xlabel('Objective 1'); ax.set_ylabel('Objective 2'); ax.set_zlabel('Objective 3')
        ax.set_title(f'MOEAD on {self.problem.__class__.__name__} (HV = {self.hv_calculator(F):.4f})')
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    def custom_decomposition_tchebycheff(F: np.ndarray, weights: np.ndarray, ideal_point: np.ndarray, **kwargs) -> np.ndarray:
        v = np.abs(F - ideal_point) * weights
        return np.max(v, axis=1)

    evaluator = MOEAD_PYMOO_Evaluation()
    score = evaluator.evaluate_program("", custom_decomposition_tchebycheff)
    results = evaluator.last_result

    print(f"Evaluation Score (Negative HV): {score:.5f}")
    print(f"Hypervolume (HV): {results['hv']:.4f}")

    if evaluator.problem.n_obj == 3 and results:
        evaluator.plot_solutions(results["pareto_front"])
