from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Literal, Sequence

from llm4ad.base import TextFunctionProgramConverter

Backend = Literal["sequential", "thread", "process"]

_PROCESS_GENERATED_FUNC = None
_PROCESS_INSTANCE_EVAL = None
_PROCESS_CONTEXT = None


def validate_backend(backend: str, *, daemon_eval_process: bool = False) -> Backend:
    if backend not in {"sequential", "thread", "process"}:
        raise ValueError("eval_backend must be one of: sequential, thread, process")
    if backend == "process" and daemon_eval_process:
        raise ValueError("process eval_backend is incompatible with daemon_eval_process=True")
    return backend


def _init_process_worker(program_str, function_name, instance_eval, context):
    global _PROCESS_GENERATED_FUNC, _PROCESS_INSTANCE_EVAL, _PROCESS_CONTEXT
    namespace = {}
    exec(program_str, namespace)
    _PROCESS_GENERATED_FUNC = namespace[function_name]
    _PROCESS_INSTANCE_EVAL = instance_eval
    _PROCESS_CONTEXT = context


def _evaluate_process_payload(payload):
    return _PROCESS_INSTANCE_EVAL(_PROCESS_GENERATED_FUNC, payload, _PROCESS_CONTEXT)


def evaluate_instances(
        *,
        program_str: str,
        callable_func: Callable | None,
        payloads: Sequence[Any],
        instance_eval: Callable[[Callable, Any, Any], Any],
        context: Any = None,
        backend: Backend = "sequential",
        workers: int = 1,
        timeout_seconds: int | float | None = None,
) -> list[Any]:
    workers = max(1, int(workers))
    if backend == "sequential" or workers == 1 or len(payloads) <= 1:
        if callable_func is None:
            function_name = TextFunctionProgramConverter.text_to_function(program_str).name
            namespace = {}
            exec(program_str, namespace)
            callable_func = namespace[function_name]
        return [instance_eval(callable_func, payload, context) for payload in payloads]

    if backend == "thread":
        if callable_func is None:
            function_name = TextFunctionProgramConverter.text_to_function(program_str).name
            namespace = {}
            exec(program_str, namespace)
            callable_func = namespace[function_name]
        max_workers = min(workers, len(payloads))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(
                lambda payload: instance_eval(callable_func, payload, context),
                payloads,
            ))

    if program_str.strip() == "_":
        raise ValueError("process eval_backend requires the generated program string.")

    function_name = TextFunctionProgramConverter.text_to_function(program_str).name
    max_workers = min(workers, len(payloads))
    try:
        mp_context = multiprocessing.get_context("fork")
    except ValueError:
        mp_context = multiprocessing.get_context()

    pool = mp_context.Pool(
        processes=max_workers,
        initializer=_init_process_worker,
        initargs=(program_str, function_name, instance_eval, context),
    )
    try:
        async_result = pool.map_async(_evaluate_process_payload, payloads)
        timeout = None
        if timeout_seconds is not None:
            timeout = max(float(timeout_seconds) - 1.0, 0.1)
        results = async_result.get(timeout=timeout)
        pool.close()
        pool.join()
        return results
    except multiprocessing.TimeoutError:
        pool.terminate()
        pool.join()
        raise TimeoutError("instance-level process evaluation timed out")
    except Exception:
        pool.terminate()
        pool.join()
        raise
