from llamea import Solution, prepare_namespace
from llm4ad.base import Function, TextFunctionProgramConverter
from llm4ad.base.evaluate import Evaluation


def _solution_score(solution: Solution):
    return getattr(solution, "fitness", getattr(solution, "score", None))


def _solution_to_function(solution: Solution) -> Function:
    function = TextFunctionProgramConverter.text_to_function(solution.code or "")
    if function is None:
        function = Function(name=getattr(solution, "name", "llamea_solution"), args="", body="    pass")
    function.score = _solution_score(solution)
    function.operator = "llamea"
    function.algorithm = getattr(solution, "description", "")
    function.sample_time = None
    function.evaluate_time = None
    return function


def _log_profiler_solution(profiler, solution: Solution, *, status: str, error=None):
    if profiler is None:
        return
    try:
        function = _solution_to_function(solution)
        profiler.register_function(function, program=solution.code or "")
    except Exception as exc:
        logger = getattr(profiler, "log_error", None)
        if callable(logger):
            logger("llamea_profiler_register", exc, status=status)
    event = getattr(profiler, "log_method_event", None)
    if callable(event):
        try:
            event(
                event="solution_evaluated",
                method="llamea",
                status=status,
                solution_name=getattr(solution, "name", None),
                parent_ids=getattr(solution, "parent_ids", None),
                score=_solution_score(solution),
                error=str(error) if error is not None else None,
            )
        except Exception:
            pass


def generate_evaluator(for_instance: Evaluation, profiler=None):
    """A LLaMEA instance works on llamea.Solution object, this generator
    takes the instance of evaluation, that have evaluate member mapping 
    Callable -> float, and returns a function that takes that `float` value 
    to update the `Solution` with appropriate fitness.
    """

    def evaluator(solution: Solution, explogger=None) -> Solution:
        """
            LLaMEA anad llm4ad evaluate functions differently, this function 
            serves as an wrapper to help evaluate the functions properly.

        Args:
            `solution: llamea.Solution`: LLaMEA comes with a `Solution` object that have all
            the arguements necessary for LLaMEA to track it as an individual in population.

            `evaulator: Callable` here is a CVRPEvaluation.evaluate, that takes in a 
            callable function, and returns its score as float.

        Returns:
            `Solution` object with updated score.
        """
        code = solution.code
        possible_issue = None
        local_ns = {}
        try:
            global_ns, possible_issue = prepare_namespace(code, allowed=['pandas', 'numpy', 'numbas'])
            exec(code, global_ns, local_ns)

        except Exception as e:
            solution.set_scores(
                float("-inf"),  # Always maximisation problem in llm4ad.
                (possible_issue if possible_issue else "") + f". Exec block failed to execute.",
                e
            )
            error_logger = getattr(profiler, "log_error", None)
            if callable(error_logger):
                error_logger("llamea_exec", e, method="llamea", counts_budget=True)
            _log_profiler_solution(profiler, solution, status="exec_failed", error=e)
            return solution
        executable = local_ns[solution.name]
        try:
            score = for_instance.evaluate(executable)
            solution.set_scores(
                score,
                f"The average distance of this heursitic is {score}.",
                None
            )
            _log_profiler_solution(profiler, solution, status="evaluated")
            return solution
        except Exception as e:
            solution.set_scores(
                float("-inf"),
                f"Code failed to execute {e}.",
                e
            )
            error_logger = getattr(profiler, "log_error", None)
            if callable(error_logger):
                error_logger("llamea_evaluate", e, method="llamea", counts_budget=True)
            _log_profiler_solution(profiler, solution, status="evaluate_failed", error=e)
            return solution
    return evaluator
