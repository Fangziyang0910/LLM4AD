from __future__ import annotations

import concurrent.futures
import copy
import time
from typing import Literal, Optional

import numpy as np

from .population import Population
from .profiler import HSEvoProfiler
from .prompt import HSEvoPrompt
from .sampler import HSEvoSampler
from .observability import (
    close_sampler_llm,
    finish_profiler,
    init_observability,
    is_search_aborted,
    log_event,
    log_llm_call,
    log_state,
    record_sample_failure,
    reset_sample_failures,
    shutdown_executor,
)
from ...base import Evaluation, Function, LLM, Program, SecureEvaluator, TextFunctionProgramConverter
from ...tools.profiler import ProfilerBase


class HSEvo:
    def __init__(
            self,
            llm: LLM,
            evaluation: Evaluation,
            profiler: ProfilerBase = None,
            max_sample_nums: Optional[int] = 450,
            pop_size: int = 10,
            init_pop_size: int = 30,
            mutation_rate: float = 0.5,
            hm_size: int = 5,
            hmcr: float = 0.7,
            par: float = 0.5,
            bandwidth: float = 0.2,
            max_iter: int = 5,
            hs_attempts_per_generation: int = 3,
            num_samplers: int = 1,
            num_evaluators: int = 1,
            *,
            external_knowledge: str = "",
            resume_mode: bool = False,
            debug_mode: bool = False,
            max_consecutive_sample_failures: int = 20,
            multi_thread_or_process_eval: Literal["thread", "process"] = "thread",
            **kwargs,
    ):
        """HSEvo integrated with the LLM4AD method/evaluation interfaces."""
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._init_pop_size = init_pop_size
        self._mutation_rate = mutation_rate
        self._hm_size = hm_size
        self._hmcr = hmcr
        self._par = par
        self._bandwidth = bandwidth
        self._max_iter = max_iter
        self._hs_attempts_per_generation = hs_attempts_per_generation
        self._external_knowledge = external_knowledge or ""

        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        raw_template_program = TextFunctionProgramConverter.text_to_program(self._template_program_str)
        self._template_program: Program = HSEvoSampler.with_global_imports(raw_template_program)
        self._function_to_evolve: Function = copy.deepcopy(self._template_program.functions[0])
        self._function_to_evolve_name = self._function_to_evolve.name

        self._population = Population(pop_size=self._pop_size)
        self._sampler = HSEvoSampler(llm, self._template_program)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler
        self._prompt = HSEvoPrompt(
            self._task_description_str,
            self._function_to_evolve,
            external_knowledge=self._external_knowledge,
        )

        self._tot_sample_nums = 0
        self._seed_function: Function | None = None
        self._elite_function: Function | None = None
        self._flash_memory = {"analyze": "", "exp": ""}
        self._comprehensive_memory = self._external_knowledge
        self._good_reflections: list[str] = []
        self._bad_reflections: list[str] = []
        init_observability(self, max_consecutive_sample_failures)

        assert multi_thread_or_process_eval in ["thread", "process"]
        if multi_thread_or_process_eval == "thread":
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_evaluators)
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_evaluators)

        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

    def _has_budget(self) -> bool:
        return (
            not is_search_aborted(self)
            and (self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums)
        )

    def _debug_print(self, title: str, content: str):
        if self._debug_mode:
            print("--------------------------------------------------------------------")
            print(f"{title}: \n{content}")
            print("--------------------------------------------------------------------\n")

    def _tag_function(self, func: Function, operator: str):
        func.operator = operator
        func.algorithm = operator

    def _register_population(self):
        if isinstance(self._profiler, HSEvoProfiler):
            self._profiler.register_population(self._population)

    def _register_function(self, func: Function, program: Program):
        if self._profiler is not None:
            self._profiler.register_function(func, program=str(program))

    def _register_reflection(self):
        if isinstance(self._profiler, HSEvoProfiler):
            self._profiler.register_reflection(
                self._population.generation,
                self._flash_memory,
                self._comprehensive_memory,
            )

    def _register_harmony_summary(self, summary: dict):
        if isinstance(self._profiler, HSEvoProfiler):
            self._profiler.register_harmony_search(self._population.generation, summary)

    def _update_elite(self, func: Function):
        if not Population.is_valid_score(func.score):
            return
        if self._elite_function is None or func.score > self._elite_function.score:
            self._elite_function = copy.deepcopy(func)

    def _evaluate_function(
            self,
            func: Function,
            operator: str,
            *,
            sample_time: float = 0.0,
            program: Program | None = None,
    ) -> Function | None:
        if not self._has_budget():
            return None

        self._tag_function(func, operator)
        if program is None:
            program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            self._tot_sample_nums += 1
            return None

        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program,
        ).result()

        func.score = score
        func.evaluate_time = eval_time
        func.sample_time = sample_time
        self._tot_sample_nums += 1
        self._register_function(func, program)

        if not Population.is_valid_score(func.score):
            return None

        self._update_elite(func)
        return func

    def _sample_evaluate_register(self, prompt: str, operator: str, **llm_kwargs) -> Function | None:
        if not self._has_budget():
            return None
        sample_start = time.time()
        sample_order = self._tot_sample_nums + 1
        try:
            response = self._sampler.draw_sample(prompt, **llm_kwargs)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage='sample',
                operator=operator,
                sample_order=sample_order,
                prompt=prompt,
                counts_budget=False,
            )
            return None
        sample_time = time.time() - sample_start
        reset_sample_failures(self)
        func = self._sampler.response_to_function(response)
        log_llm_call(
            self,
            stage='generate',
            operator=operator,
            sample_order=sample_order,
            prompt=prompt,
            response=response,
            function_parse_success=func is not None,
        )
        if func is None:
            self._tot_sample_nums += 1
            log_event(
                self,
                event='sample_rejected',
                status='parse_failed',
                operator=operator,
                sample_order=self._tot_sample_nums,
                counts_budget=True,
            )
            return None
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            self._tot_sample_nums += 1
            log_event(
                self,
                event='sample_rejected',
                status='program_parse_failed',
                operator=operator,
                sample_order=self._tot_sample_nums,
                counts_budget=True,
            )
            return None
        result = self._evaluate_function(func, operator, sample_time=sample_time, program=program)
        log_event(
            self,
            event='sample_registered',
            status='accepted' if result is not None else 'evaluated_invalid',
            operator=operator,
            sample_order=self._tot_sample_nums,
            score=getattr(func, 'score', None),
            counts_budget=True,
        )
        return result

    def _init_sampling_kwargs(self) -> dict:
        temperature = getattr(self._sampler.llm, "temperature", None)
        if isinstance(temperature, (int, float)):
            return {"temperature": temperature + 0.3}
        return {}

    def _evaluate_seed(self) -> Function:
        seed_func = copy.deepcopy(self._function_to_evolve)
        seed = self._evaluate_function(
            seed_func,
            "seed",
            sample_time=0.0,
            program=copy.deepcopy(self._template_program),
        )
        if seed is None:
            raise RuntimeError("HSEvo seed function is invalid. Check the evaluation template and task evaluator.")
        self._seed_function = copy.deepcopy(seed)
        return seed

    def _initialize_population(self):
        accepted = []
        for i in range(self._init_pop_size):
            if not self._has_budget():
                break
            seed = HSEvoPrompt.scientist(i)
            prompt = self._prompt.initial_prompt(seed, self._comprehensive_memory)
            self._debug_print("Initial Population Prompt", prompt)
            func = self._sample_evaluate_register(prompt, "init", **self._init_sampling_kwargs())
            if func is not None:
                accepted.append(func)
        self._population.set_population(accepted, increment_generation=True)
        self._register_population()

    def _selection_pool(self) -> list[Function]:
        pool = list(self._population.valid_functions())
        if self._elite_function is not None and Population.is_valid_score(self._elite_function.score):
            if not any(str(func) == str(self._elite_function) for func in pool):
                pool = [copy.deepcopy(self._elite_function)] + pool
        return pool

    def _select_parent_pairs(self) -> list[list[Function]]:
        pool = self._selection_pool()
        if len(pool) < 2:
            raise RuntimeError("HSEvo selection failed: fewer than two valid functions are available.")
        if len({float(func.score) for func in pool}) < 2:
            raise RuntimeError("HSEvo selection failed: valid functions do not contain two distinct scores.")

        selected_pairs = []
        trial = 0
        while len(selected_pairs) < self._pop_size:
            trial += 1
            parent_1, parent_2 = np.random.choice(pool, size=2, replace=False)
            if parent_1 is parent_2 or parent_1.score == parent_2.score:
                continue
            selected_pairs.append([parent_1, parent_2])
            if trial > 1000:
                raise RuntimeError("HSEvo selection failed after 1000 trials.")
        return selected_pairs

    def _flash_reflection(self, selected_population: list[Function]) -> dict[str, str]:
        prompt = self._prompt.flash_reflection_prompt(selected_population)
        self._debug_print("Flash Reflection Prompt", prompt)
        try:
            response = self._sampler.llm.draw_sample(prompt)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage='flash_reflection',
                operator='reflection',
                sample_order=self._tot_sample_nums + 1,
                prompt=prompt,
                counts_budget=False,
            )
            return self._flash_memory
        reset_sample_failures(self)
        self._flash_memory = HSEvoSampler.parse_reflection(response)
        log_llm_call(
            self,
            stage='flash_reflection',
            operator='reflection',
            sample_order=self._tot_sample_nums + 1,
            prompt=prompt,
            response=response,
            parse_success=bool(self._flash_memory.get("analyze") or self._flash_memory.get("exp")),
        )
        self._debug_print("Flash Reflection", str(self._flash_memory))
        return self._flash_memory

    def _comprehensive_reflection(self) -> str:
        good = "\n\n".join(self._good_reflections) if self._good_reflections else "None"
        bad = "\n\n".join(self._bad_reflections) if self._bad_reflections else "None"
        prompt = self._prompt.comprehensive_reflection_prompt(self._flash_memory.get("exp", ""), good, bad)
        self._debug_print("Comprehensive Reflection Prompt", prompt)
        try:
            response = self._sampler.llm.draw_sample(prompt)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage='comprehensive_reflection',
                operator='reflection',
                sample_order=self._tot_sample_nums + 1,
                prompt=prompt,
                counts_budget=False,
            )
            return self._comprehensive_memory
        reset_sample_failures(self)
        self._comprehensive_memory = "\n".join(
            part for part in [self._external_knowledge, response] if part
        )
        log_llm_call(
            self,
            stage='comprehensive_reflection',
            operator='reflection',
            sample_order=self._tot_sample_nums + 1,
            prompt=prompt,
            response=response,
        )
        self._debug_print("Comprehensive Reflection", self._comprehensive_memory)
        self._register_reflection()
        return self._comprehensive_memory

    def _crossover(self, parent_pairs: list[list[Function]]) -> list[Function]:
        crossed = []
        for parents in parent_pairs:
            if not self._has_budget():
                break
            prompt = self._prompt.crossover_prompt(
                parents,
                self._flash_memory.get("analyze", ""),
                self._comprehensive_memory,
            )
            self._debug_print("Crossover Prompt", prompt)
            func = self._sample_evaluate_register(prompt, "crossover")
            if func is not None:
                crossed.append(func)
        return crossed

    def _mutate_elite(self) -> list[Function]:
        if self._elite_function is None:
            raise RuntimeError("HSEvo mutation failed: no valid elite function is available.")
        prompt = self._prompt.mutation_prompt(self._elite_function, self._comprehensive_memory)
        self._debug_print("Mutation Prompt", prompt)
        mutated = []
        for _ in range(int(self._pop_size * self._mutation_rate)):
            if not self._has_budget():
                break
            func = self._sample_evaluate_register(prompt, "mutation")
            if func is not None:
                mutated.append(func)
        return mutated

    def _select_harmony_candidate(self) -> Function | None:
        candidates = self._population.untried_hs_candidates()
        if not candidates:
            return None
        selected = max(candidates, key=lambda f: f.score)
        self._population.mark_hs_tried(selected)
        return selected

    def _initialize_harmony_memory(self, bounds: list[tuple[float, float]]) -> np.ndarray:
        harmony_memory = np.zeros((self._hm_size, len(bounds)))
        for idx, (low, high) in enumerate(bounds):
            harmony_memory[:, idx] = np.random.uniform(low, high, self._hm_size)
        return harmony_memory

    def _create_new_harmony(self, harmony_memory: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
        new_harmony = np.zeros((harmony_memory.shape[1],))
        for idx, (low, high) in enumerate(bounds):
            if np.random.rand() < self._hmcr:
                new_harmony[idx] = harmony_memory[np.random.randint(0, harmony_memory.shape[0]), idx]
                if np.random.rand() < self._par:
                    adjustment = np.random.uniform(-1, 1) * (high - low) * self._bandwidth
                    new_harmony[idx] += adjustment
            else:
                new_harmony[idx] = np.random.uniform(low, high)
            new_harmony[idx] = min(max(new_harmony[idx], low), high)
        return new_harmony

    def _evaluate_harmony_values(
            self,
            function_source: str,
            parameter_names: list[str],
            values: np.ndarray,
    ) -> Function | None:
        if not self._has_budget():
            return None
        value_map = {name: float(values[idx]) for idx, name in enumerate(parameter_names)}
        func = HSEvoSampler.function_with_harmony_values(function_source, value_map)
        if func is None:
            self._tot_sample_nums += 1
            return None
        return self._evaluate_function(func, "harmony_search", sample_time=0.0)

    def _harmony_search(self) -> Function | None:
        candidate = self._select_harmony_candidate()
        if candidate is None:
            self._register_harmony_summary({"status": "no_candidate"})
            return None

        prompt = self._prompt.harmony_search_prompt(candidate)
        self._debug_print("Harmony Search Prompt", prompt)
        try:
            response = self._sampler.llm.draw_sample(prompt)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage='harmony_search',
                operator='harmony_search',
                sample_order=self._tot_sample_nums + 1,
                prompt=prompt,
                counts_budget=False,
                source_score=candidate.score,
            )
            return None
        reset_sample_failures(self)
        parameter_ranges, function_source = HSEvoSampler.parse_harmony_response(response)
        log_llm_call(
            self,
            stage='harmony_search',
            operator='harmony_search',
            sample_order=self._tot_sample_nums + 1,
            prompt=prompt,
            response=response,
            parse_success=parameter_ranges is not None and function_source is not None,
            source_score=candidate.score,
        )
        if parameter_ranges is None or function_source is None:
            self._register_harmony_summary({"status": "parse_failed", "source_score": candidate.score})
            return None

        parameter_names = list(parameter_ranges)
        bounds = [parameter_ranges[name] for name in parameter_names]
        harmony_memory = self._initialize_harmony_memory(bounds)
        evaluated: list[tuple[Function, np.ndarray]] = []

        for values in harmony_memory:
            func = self._evaluate_harmony_values(function_source, parameter_names, values)
            if func is not None:
                evaluated.append((func, values.copy()))
        if not evaluated:
            self._register_harmony_summary({"status": "no_valid_initial_harmony", "source_score": candidate.score})
            return None

        init_best = max(func.score for func, _ in evaluated)
        for _ in range(self._max_iter):
            if not self._has_budget():
                break
            current_memory = np.array([values for _, values in evaluated])
            new_values = self._create_new_harmony(current_memory, bounds)
            new_func = self._evaluate_harmony_values(function_source, parameter_names, new_values)
            if new_func is None:
                continue
            worst_idx = min(range(len(evaluated)), key=lambda idx: evaluated[idx][0].score)
            if new_func.score > evaluated[worst_idx][0].score:
                evaluated[worst_idx] = (new_func, new_values.copy())

        best_func, _ = max(evaluated, key=lambda item: item[0].score)
        accepted = self._population.extend([best_func])
        if accepted:
            self._population.advance_generation()
            self._register_population()
        self._register_harmony_summary({
            "status": "success",
            "source_score": candidate.score,
            "init_best": init_best,
            "best_score": best_func.score,
            "accepted": bool(accepted),
            "parameters": parameter_names,
        })
        return best_func

    def _run_harmony_search_attempts(self):
        for _ in range(self._hs_attempts_per_generation):
            if not self._has_budget():
                break
            if self._harmony_search() is not None:
                break

    def _run_evolution_generation(self):
        parent_pairs = self._select_parent_pairs()
        selected_population = [func for pair in parent_pairs for func in pair]
        elite_before = str(self._elite_function) if self._elite_function is not None else ""

        self._flash_reflection(selected_population)
        self._comprehensive_reflection()

        crossed_population = self._crossover(parent_pairs)
        if crossed_population:
            self._population.set_population(crossed_population, increment_generation=True)
            self._register_population()

        if self._has_budget():
            mutated_population = self._mutate_elite()
            if mutated_population:
                self._population.extend(mutated_population)
                self._population.advance_generation()
                self._register_population()

        elite_after = str(self._elite_function) if self._elite_function is not None else ""
        reflection_exp = self._flash_memory.get("exp", "")
        if elite_before != elite_after:
            self._good_reflections.append(reflection_exp)
        else:
            self._bad_reflections.append(reflection_exp)

        if self._has_budget():
            self._run_harmony_search_attempts()
        log_state(
            self,
            phase='population',
            method='hsevo',
            generation=self._population.generation,
            population_size=len(self._population),
            sample_count=self._tot_sample_nums,
            elite_score=getattr(self._elite_function, 'score', None),
        )

    def run(self):
        try:
            if not self._resume_mode:
                if self._has_budget():
                    self._evaluate_seed()
                if self._has_budget():
                    self._initialize_population()

            while self._has_budget():
                self._run_evolution_generation()
        finally:
            shutdown_executor(self._evaluation_executor)
            finish_profiler(
                self,
                status='aborted' if is_search_aborted(self) else 'finished',
            )
            close_sampler_llm(self._sampler)
