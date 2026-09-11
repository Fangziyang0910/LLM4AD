"""Small output recovery helpers; Python validity stays with the existing parser."""
import ast
import re

from llm4ad.method.traceaad_v10_3.traceaad import _strip_thinking

OUTPUT = (
    'Return an Idea: paragraph describing the resulting decision rule (at most 100 words), '
    'followed by one Python code block containing the complete implementation. '
    'Make the Idea self-contained, without input program labels or provenance. '
    'Include all required imports and helpers, define variables before use, and respect '
    'the given input shapes, target signature and return contract.'
)
ERROR_MESSAGE_MAX_CHARS = 2000


def parse_candidate(response, finish_reason, validate):
    parsed = validate(response, finish_reason)
    if parsed is not None:
        return parsed, 'strict', None
    if finish_reason not in ('stop', 'length', 'unknown'):
        return None, 'failed', f'Unsupported finish reason: {finish_reason}'
    text = _strip_thinking(response)
    fences = list(re.finditer(r'^[ \t]*```([^\r\n]*)\r?$', text, re.MULTILINE))
    if (not fences or len(fences) > 2 or
            fences[0][1].strip().lower() not in ('', 'python', 'py') or
            (len(fences) == 2 and fences[1][1].strip())):
        return None, 'failed', 'Expected one unambiguous Python code block.'
    if len(fences) == 1 and finish_reason == 'length':
        return None, 'failed', 'Output reached the token limit with an unclosed code block.'
    idea = re.search(r'(?im)^[ \t]*(?:Design[ \t]+)?Idea[ \t]*:[ \t]*(\S[\s\S]*)',
                     text[:fences[0].start()])
    if idea is None:
        return None, 'failed', 'Missing Idea: paragraph; preserve the code and supply its description.'
    code = text[fences[0].end():fences[1].start() if len(fences) == 2 else len(text)].strip()
    try:
        compile(ast.parse(code), '<candidate>', 'exec')
    except (SyntaxError, ValueError) as exc:
        return None, 'failed', f'{type(exc).__name__}: {exc}'
    parsed = validate(f'Idea: {idea[1].strip()}\n```python\n{code}\n```', 'stop')
    if parsed is None:
        return None, 'failed', 'Expected exactly one target function with the specified parameter names.'
    return parsed, 'recovered', None


def repair_prompt(task_contract, response, event):
    # Keep the exception message; full traceback and paths remain in the journal.
    message = (event.get('error') or event['reason'] or 'Evaluation failed').strip()
    message = re.sub(r"(['\"])(?:/|[A-Za-z]:[\\/])[^'\"]+\1", '<path>', message)
    message = re.sub(r'(?<!\w)(?:/|[A-Za-z]:[\\/])[^\s,;:]+', '<path>', message)
    feedback = f"{event.get('error_type') or 'Error'}: {message[:ERROR_MESSAGE_MAX_CHARS]}"
    if event['reason'] == 'timeout':
        feedback = ('Evaluation exceeded its total time limit. '
                    'Reduce computation and ensure loops terminate while preserving the main idea.')
    baseline = (f"\nParent fitness: {event['parent_fitness']} (higher is better).\n"
                if event.get('parent_fitness') is not None else '')
    return (f'{task_contract}\n{baseline}\n# Failed output\n{_strip_thinking(response)}\n\n'
            f"# Failure during {event['operator']}\n{feedback}\n\n"
            '# Repair\nMake the smallest change addressing this failure. Preserve the proposed '
            'algorithmic idea, unrelated computations and target signature. For a formatting '
            'failure, preserve the code whenever possible. Return the Idea and full corrected '
            f'program.\n\n{OUTPUT}')
