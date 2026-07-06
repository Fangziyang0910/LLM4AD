from __future__ import annotations

import json
import os
from threading import Lock
from typing import Optional

from .population import Population
from ...base import Function
from ...tools.profiler import ProfilerBase


class HSEvoProfiler(ProfilerBase):
    def __init__(
            self,
            log_dir: Optional[str] = None,
            *,
            initial_num_samples=0,
            log_style="complex",
            create_random_path=True,
            **kwargs,
    ):
        super().__init__(
            log_dir=log_dir,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs,
        )
        self._cur_gen = 0
        self._pop_lock = Lock()
        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, "population")
            self._event_dir = os.path.join(self._log_dir, "hsevo")
            os.makedirs(self._ckpt_dir, exist_ok=True)
            os.makedirs(self._event_dir, exist_ok=True)

    def register_population(self, pop: Population):
        if not self._log_dir:
            return
        try:
            self._pop_lock.acquire()
            if self._num_samples == 0 or pop.generation == self._cur_gen:
                return
            funcs_json = []
            for func in pop.population:
                funcs_json.append({
                    "algorithm": func.algorithm,
                    "function": str(func),
                    "operator": func.operator,
                    "score": func.score,
                })
            path = os.path.join(self._ckpt_dir, f"pop_{pop.generation}.json")
            with open(path, "w") as json_file:
                json.dump(funcs_json, json_file, indent=4)
            self._cur_gen = pop.generation
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()

    def _append_event(self, filename: str, content: dict):
        if not self._log_dir:
            return
        path = os.path.join(self._event_dir, filename)
        with open(path, "a") as jsonl_file:
            jsonl_file.write(json.dumps(content) + "\n")

    def register_reflection(self, generation: int, flash: dict[str, str], comprehensive: str):
        self._append_event("reflections.jsonl", {
            "generation": generation,
            "flash": flash,
            "comprehensive": comprehensive,
        })

    def register_harmony_search(self, generation: int, summary: dict):
        content = {"generation": generation}
        content.update(summary)
        self._append_event("harmony_search.jsonl", content)

    def _write_json(self, function: Function, program="", *, record_type="history", record_sep=200):
        assert record_type in ["history", "best"]
        if not self._log_dir:
            return

        sample_order = self._num_samples
        content = {
            "sample_order": sample_order,
            "algorithm": function.algorithm,
            "function": str(function),
            "operator": function.operator,
            "score": function.score,
            "program": program,
        }

        if record_type == "history":
            lower_bound = ((sample_order - 1) // record_sep) * record_sep
            upper_bound = lower_bound + record_sep
            filename = f"samples_{lower_bound + 1}~{upper_bound}.json"
        else:
            filename = "samples_best.json"

        path = os.path.join(self._samples_json_dir, filename)
        try:
            with open(path, "r") as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(content)
        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4)

