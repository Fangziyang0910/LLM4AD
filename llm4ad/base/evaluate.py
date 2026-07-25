# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
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
import queue
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from .code import TextFunctionProgramConverter, Program
from .modify_code import ModifyCode
import traceback


@dataclass(frozen=True)
class EvaluationOutcome:
    """Result and a compact reason when generated code cannot be evaluated."""

    result: Any | None
    failure_kind: str | None = None
    error_type: str | None = None
    error: str | None = None


class Evaluation(ABC):
    def __init__(
            self,
            template_program: str | Program,
            task_description: str = '',
            use_numba_accelerate: bool = False,
            use_protected_div: bool = False,
            protected_div_delta: float = 1e-5,
            random_seed: int | None = None,
            timeout_seconds: int | float = None,
            *,
            exec_code: bool = True,
            safe_evaluate: bool = True,
            daemon_eval_process: bool = False,
            fork_proc: Literal['auto'] | bool = 'auto'
    ):
        """Evaluation interface for executing generated code.
        Args:
            use_numba_accelerate: Wrap the function with '@numba.jit(nopython=True)'.
            use_protected_div   : Modify 'a / b' => 'a / (b + delta)'. Maybe useful for mathematical tasks.
            protected_div_delta : Delta value in protected div.
            random_seed         : If is not None, set random seed in the first line of the function body.
            timeout_seconds     : Terminate the evaluation after timeout seconds.
            exec_code           : Using 'exec()' to compile the code and provide the callable function.
                If is set to 'False', the 'callable_func' argument in 'self.evaluate_program' is always 'None'.
                If is set to 'False', the user should provide the score of the program based on 'program_str' argument in 'self.evaluate_program'.
            safe_evaluate       : Evaluate in safe mode using a new process. If is set to False,
                the evaluation will not be terminated after timeout seconds. The user should consider how to
                terminate evaluating in time.
            daemon_eval_process : Set the evaluate process as a daemon process. If set to True,
                you can not set new processes in the evaluator. Which means in self.evaluate_program(),
                you can not create new processes.
            fork_proc           : This arg is valid when safe_evaluate=True, which determines to 'fork' process or 'spawn' a safe process.
                If set to 'auto', the process creating method will depend on OS. Set to 'True' to use 'fork', 'False' to use 'spawn'.

        -Assume that: use_numba_accelerate=True, self.use_protected_div=True, and self.random_seed=2024.
        -The original function:
        --------------------------------------------------------------------------------
        import numpy as np

        def f(a, b):
            a = np.random.random()
            return a / b
        --------------------------------------------------------------------------------
        -In the Evaluation phase, the modified function will be:
        --------------------------------------------------------------------------------
        import numpy as np
        import numba

        @numba.jit(nopython=True)
        def f():
            np.random.seed(2024)
            a = np.random.random()
            return _protected_div(a, b)

        def _protected_div(a, b, delta=1e-5):
            return a / (b + delta)
        --------------------------------------------------------------------------------
        As shown above, the 'import numba', 'numba.jit()' decorator, and '_protected_dev' will be added by this function.
        """
        self.template_program = template_program
        self.task_description = task_description
        self.use_numba_accelerate = use_numba_accelerate
        self.use_protected_div = use_protected_div
        self.protected_div_delta = protected_div_delta
        self.random_seed = random_seed
        self.timeout_seconds = timeout_seconds
        self.exec_code = exec_code
        self.safe_evaluate = safe_evaluate
        self.daemon_eval_process = daemon_eval_process
        self.fork_proc = fork_proc

    @abstractmethod
    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        r"""Evaluate a given function. You can use compiled function (function_callable),
        as well as the original function strings for evaluation.
        Args:
            program_str: The function in string. You can _ignore this argument when implementation. (See below).
            callable_func: The callable heuristic function. You can call it using `callable_func(args, kwargs)`.
        Return:
            Returns the fitness value.

        Assume that: self.use_numba_accelerate = True, self.use_protected_div = True,
        and self.random_seed = 2024, the argument 'function_str' will be something like below:
        --------------------------------------------------------------------------------
        import numpy as np
        import numba

        @numba.jit(nopython=True)
        def f(a, b):
            np.random.seed(2024)
            a = a + np.random.random()
            return _protected_div(a, b)

        def _protected_div(a, b, delta=1e-5):
            return a / (b + delta)
        --------------------------------------------------------------------------------
        As shown above, the 'import numba', 'numba.jit()' decorator,
        and '_protected_dev' will be added by this function.
        """
        raise NotImplementedError('Must provide a evaluator for a function.')


class SecureEvaluator:
    def __init__(self,
                 evaluator: Evaluation,
                 debug_mode=False,
                 **kwargs):
        self._evaluator = evaluator
        self._debug_mode = debug_mode
        fork_proc = self._evaluator.fork_proc

        if self._evaluator.safe_evaluate:
            if fork_proc == 'auto':
                # force MacOS and Linux use 'fork' to generate new process
                if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
                    multiprocessing.set_start_method('fork', force=True)
            elif fork_proc is True:
                multiprocessing.set_start_method('fork', force=True)
            elif fork_proc is False:
                multiprocessing.set_start_method('spawn', force=True)

    def _modify_program_code(self, program_str: str, function_name: str) -> str:
        if self._evaluator.use_numba_accelerate:
            program_str = ModifyCode.add_numba_decorator(
                program_str, function_name=function_name
            )
        if self._evaluator.use_protected_div:
            program_str = ModifyCode.replace_div_with_protected_div(
                program_str, self._evaluator.protected_div_delta, self._evaluator.use_numba_accelerate
            )
        if self._evaluator.random_seed is not None:
            program_str = ModifyCode.add_numpy_random_seed_to_func(
                program_str, function_name, self._evaluator.random_seed
            )
        return program_str

    def evaluate_program(self, program: str | Program, **kwargs):
        return self.evaluate_program_with_details(program, **kwargs).result

    def evaluate_program_with_details(
            self, program: str | Program, **kwargs
    ) -> EvaluationOutcome:
        try:
            program_str = str(program)
            function_name = self._target_function_name()

            program_str = self._modify_program_code(program_str, function_name)
            if self._debug_mode:
                print(f'DEBUG: evaluated program:\n{program_str}\n')

            # safe evaluate
            if self._evaluator.safe_evaluate:
                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=self._evaluate_in_safe_process_with_details,
                    args=(program_str, function_name, result_queue),
                    kwargs=kwargs,
                    daemon=self._evaluator.daemon_eval_process
                )
                process.start()

                try:
                    if self._evaluator.timeout_seconds is None:
                        outcome = result_queue.get()
                    else:
                        outcome = result_queue.get(
                            timeout=self._evaluator.timeout_seconds
                        )
                except queue.Empty:
                    if self._debug_mode:
                        print(
                            'DEBUG: the evaluation time exceeds '
                            f'{self._evaluator.timeout_seconds}s.'
                        )
                    outcome = EvaluationOutcome(
                        result=None,
                        failure_kind='timeout',
                        error_type='TimeoutError',
                        error=(
                            'evaluation exceeded '
                            f'{self._evaluator.timeout_seconds}s'
                        ),
                    )
                finally:
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()
                return outcome
            else:
                return self._evaluate_with_details(
                    program_str, function_name, **kwargs
                )
        except Exception as e:
            if self._debug_mode:
                print("DEBUG: Exception occurred in evaluate_program:")
                traceback.print_exc()  # 这将打印完整红色报错信息
            return self._failure('prepare_error', e)

    def evaluate_program_record_time(self, program: str | Program, **kwargs):
        evaluate_start = time.time()
        result = self.evaluate_program(program, **kwargs)
        return result, time.time() - evaluate_start

    def evaluate_program_record_time_with_details(
            self, program: str | Program, **kwargs
    ) -> tuple[EvaluationOutcome, float]:
        evaluate_start = time.time()
        outcome = self.evaluate_program_with_details(program, **kwargs)
        return outcome, time.time() - evaluate_start

    def _target_function_name(self) -> str:
        source = self._evaluator.template_program
        template = (
            source
            if isinstance(source, Program)
            else TextFunctionProgramConverter.text_to_program(source)
        )
        if template is None or len(template.functions) != 1:
            raise ValueError('evaluation template must define one target function')
        return template.functions[0].name

    def _evaluate_in_safe_process_with_details(
            self,
            program_str: str,
            function_name: str,
            result_queue: multiprocessing.Queue,
            **kwargs,
    ) -> None:
        try:
            if self._evaluator.exec_code:
                all_globals_namespace = {}
                exec(program_str, all_globals_namespace)
                program_callable = all_globals_namespace[function_name]
            else:
                program_callable = None
        except Exception as exc:
            result_queue.put(self._failure('exec_error', exc))
            return

        try:
            res = self._evaluator.evaluate_program(program_str, program_callable, **kwargs)
            result_queue.put(self._outcome(res))
        except Exception as exc:
            if self._debug_mode:
                print("DEBUG: Exception occurred in evaluate_program:")
                traceback.print_exc()  # 这将打印完整红色报错信息
            result_queue.put(self._failure('runtime_error', exc))

    def _evaluate_with_details(self, program_str: str, function_name, **kwargs):
        try:
            if self._evaluator.exec_code:
                all_globals_namespace = {}
                exec(program_str, all_globals_namespace)
                program_callable = all_globals_namespace[function_name]
            else:
                program_callable = None
        except Exception as exc:
            return self._failure('exec_error', exc)

        try:
            res = self._evaluator.evaluate_program(program_str, program_callable, **kwargs)
            return self._outcome(res)
        except Exception as exc:
            if self._debug_mode:
                print("DEBUG: Exception occurred in evaluate_program:")
                traceback.print_exc()  # 这将打印完整红色报错信息
            return self._failure('runtime_error', exc)

    @staticmethod
    def _outcome(result: Any | None) -> EvaluationOutcome:
        if result is None:
            return EvaluationOutcome(
                result=None,
                failure_kind='invalid_result',
                error_type='InvalidEvaluationResult',
                error='evaluator returned None',
            )
        return EvaluationOutcome(result=result)

    @staticmethod
    def _failure(kind: str, exc: Exception) -> EvaluationOutcome:
        message = str(exc)
        if len(message) > 1000:
            message = message[:997] + '...'
        return EvaluationOutcome(
            result=None,
            failure_kind=kind,
            error_type=type(exc).__name__,
            error=message,
        )
