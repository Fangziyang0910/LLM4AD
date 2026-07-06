# Module Name: ReEvo
# Last Revision: 2025/2/16
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#
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

import concurrent.futures
import copy
import time
from typing import Optional, Literal

import numpy as np

from .population import Population
from .profiler import ReEvoProfiler
from .prompt import ReEvoPrompt
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator, SampleTrimmer
)
from ...tools.profiler import ProfilerBase


class ReEvo:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_sample_nums: Optional[int] = 100,
                 pop_size: int = 10,
                 init_pop_size: int = 30,
                 mutation_rate: float = 0.5,
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 *,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        """Reflective Evolution following the original ReEvo mechanics.

        LLM4AD evaluators use higher scores for better programs. Internally,
        every better/worse comparison follows that score convention.
        """
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._init_pop_size = init_pop_size
        self._mutation_rate = mutation_rate

        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        self._population = Population(pop_size=self._pop_size)
        self._sampler = SampleTrimmer(llm)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        self._tot_sample_nums = 0
        self._seed_function: Function | None = None
        self._elite_function: Function | None = None
        self._long_term_reflection_str = ''

        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

    def _has_budget(self) -> bool:
        return self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums

    def _tag_function(self, func: Function, operator: str):
        func.operator = operator
        func.algorithm = operator

    def _debug_print(self, title: str, content: str):
        if self._debug_mode:
            print('--------------------------------------------------------------------')
            print(f'{title}: \n{content}')
            print('--------------------------------------------------------------------\n')

    def _register_population(self):
        if isinstance(self._profiler, ReEvoProfiler):
            self._profiler.register_population(self._population)

    def _register_function(self, func: Function, program: Program):
        if self._profiler is not None:
            self._profiler.register_function(func, program=str(program))

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
            program
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
        generated_code = self._sampler.draw_sample(prompt, **llm_kwargs)
        sample_time = time.time() - sample_start

        func = SampleTrimmer.sample_to_function(generated_code, self._template_program)
        if func is None:
            self._tot_sample_nums += 1
            return None

        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            self._tot_sample_nums += 1
            return None

        return self._evaluate_function(func, operator, sample_time=sample_time, program=program)

    def _init_sampling_kwargs(self) -> dict:
        temperature = getattr(self._sampler.llm, 'temperature', None)
        if isinstance(temperature, (int, float)):
            return {'temperature': temperature + 0.3}
        return {}

    def _evaluate_seed(self) -> Function:
        seed_func = copy.deepcopy(self._function_to_evolve)
        seed = self._evaluate_function(
            seed_func,
            'seed',
            sample_time=0.0,
            program=copy.deepcopy(self._template_program),
        )
        if seed is None:
            raise RuntimeError('ReEvo seed function is invalid. Check the evaluation template and task evaluator.')
        self._seed_function = copy.deepcopy(seed)
        return seed

    def _initialize_population(self):
        prompt = ReEvoPrompt.get_pop_init_prompt(self._task_description_str, self._function_to_evolve)
        self._debug_print('Initial Population Prompt', prompt)
        accepted = []
        for _ in range(self._init_pop_size):
            if not self._has_budget():
                break
            func = self._sample_evaluate_register(prompt, 'init', **self._init_sampling_kwargs())
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
            raise RuntimeError('ReEvo selection failed: fewer than two valid functions are available.')
        if len({float(func.score) for func in pool}) < 2:
            raise RuntimeError('ReEvo selection failed: valid functions do not contain two distinct scores.')

        selected_pairs = []
        trial = 0
        while len(selected_pairs) < self._pop_size:
            trial += 1
            parent_1, parent_2 = np.random.choice(pool, size=2, replace=False)
            if parent_1 is parent_2:
                continue
            if parent_1.score == parent_2.score:
                continue
            selected_pairs.append([parent_1, parent_2])
            if trial > 1000:
                raise RuntimeError('ReEvo selection failed after 1000 trials.')
        return selected_pairs

    def _short_term_reflection(self, parents: list[Function]) -> str:
        prompt = ReEvoPrompt.get_short_term_reflection_prompt(self._task_description_str, parents)
        self._debug_print('Short-term Reflection Prompt', prompt)
        reflection = self._sampler.llm.draw_sample(prompt)
        self._debug_print('Short-term Reflection', reflection)
        return reflection

    def _crossover(self, parents: list[Function], short_term_reflection: str) -> Function | None:
        prompt = ReEvoPrompt.get_crossover_prompt(self._task_description_str, short_term_reflection, parents)
        self._debug_print('Crossover Prompt', prompt)
        return self._sample_evaluate_register(prompt, 'crossover')

    def _long_term_reflection(self, short_term_reflections: list[str]) -> str:
        prompt = ReEvoPrompt.get_long_term_reflection_prompt(
            self._task_description_str,
            self._long_term_reflection_str,
            short_term_reflections,
        )
        self._debug_print('Long-term Reflection Prompt', prompt)
        self._long_term_reflection_str = self._sampler.llm.draw_sample(prompt)
        self._debug_print('Long-term Reflection', self._long_term_reflection_str)
        return self._long_term_reflection_str

    def _mutate_elite(self) -> list[Function]:
        if self._elite_function is None:
            raise RuntimeError('ReEvo mutation failed: no valid elite function is available.')

        prompt = ReEvoPrompt.get_elist_mutation_prompt(
            self._task_description_str,
            self._long_term_reflection_str,
            self._elite_function,
        )
        self._debug_print('Elitist Mutation Prompt', prompt)

        mutated = []
        for _ in range(int(self._pop_size * self._mutation_rate)):
            if not self._has_budget():
                break
            func = self._sample_evaluate_register(prompt, 'mutation')
            if func is not None:
                mutated.append(func)
        return mutated

    def _run_evolution_generation(self):
        parent_pairs = self._select_parent_pairs()
        short_term_reflections = [self._short_term_reflection(pair) for pair in parent_pairs]

        crossed_population = []
        for parents, reflection in zip(parent_pairs, short_term_reflections):
            if not self._has_budget():
                break
            func = self._crossover(parents, reflection)
            if func is not None:
                crossed_population.append(func)

        if not crossed_population:
            return

        self._population.set_population(crossed_population, increment_generation=True)
        self._register_population()

        if not self._has_budget():
            return

        self._long_term_reflection(short_term_reflections)
        mutated_population = self._mutate_elite()
        if mutated_population:
            self._population.extend(mutated_population)
            self._population.advance_generation()
            self._register_population()

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
            try:
                self._evaluation_executor.shutdown(cancel_futures=True)
            except Exception:
                pass
            if self._profiler is not None:
                self._profiler.finish()
            self._sampler.llm.close()
