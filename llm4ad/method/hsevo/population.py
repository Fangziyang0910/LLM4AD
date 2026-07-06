from __future__ import annotations

import copy
import math
from threading import Lock
from typing import Iterable, List

from ...base import Function


class Population:
    def __init__(self, pop_size: int, generation=0, pop: List[Function] | "Population" | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = list(pop)
        else:
            self._population = list(pop._population)

        self._pop_size = pop_size
        self._generation = generation
        self._lock = Lock()
        self._hs_tried_codes: set[str] = set()

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    @property
    def population(self) -> list[Function]:
        return self._population

    @property
    def generation(self) -> int:
        return self._generation

    @staticmethod
    def is_valid_score(score) -> bool:
        if score is None:
            return False
        try:
            return math.isfinite(float(score))
        except (TypeError, ValueError):
            return False

    def valid_functions(self) -> list[Function]:
        return [func for func in self._population if self.is_valid_score(func.score)]

    def _unique_valid(self, funcs: Iterable[Function]) -> list[Function]:
        unique = []
        seen_code = set()
        seen_score = set()
        for func in funcs:
            if not self.is_valid_score(func.score):
                continue
            code_key = str(func)
            score_key = float(func.score)
            if code_key in seen_code or score_key in seen_score:
                continue
            seen_code.add(code_key)
            seen_score.add(score_key)
            unique.append(func)
        return unique

    def set_population(self, funcs: list[Function], *, increment_generation=True):
        with self._lock:
            self._population = self._unique_valid(funcs)
            if increment_generation:
                self._generation += 1

    def extend(self, funcs: list[Function]) -> list[Function]:
        with self._lock:
            accepted = []
            current = self._unique_valid(self._population)
            seen_code = {str(func) for func in current}
            seen_score = {float(func.score) for func in current}
            for func in funcs:
                if not self.is_valid_score(func.score):
                    continue
                code_key = str(func)
                score_key = float(func.score)
                if code_key in seen_code or score_key in seen_score:
                    continue
                seen_code.add(code_key)
                seen_score.add(score_key)
                current.append(func)
                accepted.append(func)
            self._population = current
            return accepted

    def register_function(self, func: Function, *, increment_generation=False) -> bool:
        accepted = self.extend([func])
        if accepted and increment_generation:
            self.advance_generation()
        return bool(accepted)

    def advance_generation(self):
        with self._lock:
            self._generation += 1

    @property
    def elite_function(self) -> Function:
        return copy.deepcopy(max(self.valid_functions(), key=lambda f: f.score))

    def mark_hs_tried(self, func: Function):
        self._hs_tried_codes.add(str(func))

    def has_hs_tried(self, func: Function) -> bool:
        return str(func) in self._hs_tried_codes

    def untried_hs_candidates(self) -> list[Function]:
        return [func for func in self.valid_functions() if not self.has_hs_tried(func)]

