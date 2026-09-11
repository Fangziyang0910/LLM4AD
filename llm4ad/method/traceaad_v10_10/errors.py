"""Code-first candidate parsing: the program alone decides evaluability."""
import ast
import re

from llm4ad.method.traceaad_v10_3.schema import normalize_code
from llm4ad.method.traceaad_v10_3.traceaad import THINK_BLOCK_RE

OUTPUT = (
    "Describe the algorithm's main decision method and key computations, using up to "
    "about 500 words as needed. Present it as Idea, followed by one Python code block "
    "containing the complete implementation. Include all required imports and helpers, "
    "define variables before use, and respect the given input shapes, target signature "
    "and return contract."
)
PARSE_POLICY = 'code_first_description_extracted_v1'
ERROR_MESSAGE_MAX_CHARS = 2000

FENCE_LINE_RE = re.compile(r'^[ \t]*```([^\r\n]*)\r?$', re.MULTILINE)
IDEA_LABEL_RE = re.compile(
    r'(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?[ \t]*(?:design[ \t]+)?idea\b'
    r'(?:[ \t]*\*\*)?[ \t]*:?[ \t]*(?:\*\*)?[ \t]*')
CANDIDATE_FRAME_RE = re.compile(r'File "<(?:string|candidate)>", line (\d+), in ([^\r\n]+)')


def _signature(args):
    return ([a.arg for a in args.posonlyargs], [a.arg for a in args.args],
            [a.arg for a in args.kwonlyargs],
            args.vararg.arg if args.vararg else None,
            args.kwarg.arg if args.kwarg else None)


def expected_interface(name, args_text):
    """Signature of the task's target function, parsed once per run."""
    expected = ast.parse(f'def target({args_text}):\n    pass').body[0].args
    return name, args_text, _signature(expected)


def _strip_thinking_outside_code(response):
    """Remove thinking blocks for response analysis without rewriting the code.

    Returns the cleaned text plus a map from cleaned offsets back to the
    original response, so the extracted program is exactly what was written.
    """
    segments = []  # (cleaned_start, cleaned_end, raw_start) of the kept spans
    parts = []
    raw_prev, cleaned = 0, 0
    for match in THINK_BLOCK_RE.finditer(response):
        parts.append(response[raw_prev:match.start()])
        segments.append((cleaned, cleaned + match.start() - raw_prev, raw_prev))
        cleaned += match.start() - raw_prev
        raw_prev = match.end()
    parts.append(response[raw_prev:])
    segments.append((cleaned, cleaned + len(response) - raw_prev, raw_prev))
    text = ''.join(parts)

    def to_raw(offset):
        for cleaned_start, cleaned_end, raw_start in segments:
            if offset <= cleaned_end:
                return raw_start + max(offset, cleaned_start) - cleaned_start
        return len(response)

    return text, to_raw


def _extract_description(before, after):
    """Return (description, source) from the non-code text around the program."""
    labels = list(IDEA_LABEL_RE.finditer(before)) + list(IDEA_LABEL_RE.finditer(after))
    if labels:
        # The last explicit label wins, so a description placed after the code
        # is not discarded in favour of an introductory sentence.
        match = labels[-1]
        return match.string[match.end():].strip(), 'tagged'
    prose = '\n\n'.join(part.strip() for part in (before, after) if part.strip())
    return (prose, 'prose') if prose else ('', 'missing')


def parse_candidate(response, finish_reason, interface):
    """Extract one complete program and its description from a response.

    The program alone decides whether the candidate can be evaluated; the
    description is recorded as given (tagged, prose or empty) and never gates
    the code. Returns ((idea, code, code), idea_source, error).
    """
    name, args_text, signature = interface
    if finish_reason not in ('stop', 'length', 'unknown'):
        return None, 'failed', f'Unsupported finish reason: {finish_reason}'
    text, to_raw = _strip_thinking_outside_code(response)
    fences = list(FENCE_LINE_RE.finditer(text))
    if (not fences or len(fences) > 2 or
            fences[0][1].strip().lower() not in ('', 'python', 'py') or
            (len(fences) == 2 and fences[1][1].strip())):
        return None, 'failed', 'Expected one unambiguous Python code block.'
    if len(fences) == 1 and finish_reason == 'length':
        return None, 'failed', 'Output reached the token limit with an unclosed code block.'
    code_end = fences[1].start() if len(fences) == 2 else len(text)
    after_start = fences[1].end() if len(fences) == 2 else len(text)
    code = response[to_raw(fences[0].end()):to_raw(code_end)]
    canonical = normalize_code(code)
    if not canonical:
        return None, 'failed', 'The code block is empty.'
    try:
        tree = ast.parse(canonical)
    except (SyntaxError, ValueError) as exc:
        return None, 'failed', f'{type(exc).__name__}: {exc}'
    targets = [n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(targets) != 1:
        return None, 'failed', f'Expected exactly one `{name}(...)` function definition.'
    if _signature(targets[0].args) != signature:
        return None, 'failed', f'`{name}` must declare the parameters ({args_text}).'
    try:
        compile(canonical, '<candidate>', 'exec')
    except (SyntaxError, ValueError) as exc:
        return None, 'failed', f'{type(exc).__name__}: {exc}'
    idea, source = _extract_description(text[:fences[0].start()], text[after_start:])
    return (idea, canonical, canonical), source, None


def _display_view(response):
    """Show a failed response without thinking blocks, keeping its code as written."""
    text, to_raw = _strip_thinking_outside_code(response)
    fences = list(FENCE_LINE_RE.finditer(text))
    if (fences and len(fences) <= 2 and
            fences[0][1].strip().lower() in ('', 'python', 'py') and
            not (len(fences) == 2 and fences[1][1].strip())):
        code_end = fences[1].start() if len(fences) == 2 else len(text)
        return (text[:fences[0].end()] +
                response[to_raw(fences[0].end()):to_raw(code_end)] +
                text[code_end:])
    return text


def _candidate_location(traceback_text):
    """Deepest frame of the candidate module in an evaluator traceback."""
    frames = CANDIDATE_FRAME_RE.findall(traceback_text or '')
    return frames[-1] if frames else None


def repair_prompt(task_contract, response, event):
    # Keep the exception message and the failing candidate line; full
    # traceback and paths remain in the journal.
    message = (event.get('error') or event['reason'] or 'Evaluation failed').strip()
    message = re.sub(r"(['\"])(?:/|[A-Za-z]:[\\/])[^'\"]+\1", '<path>', message)
    message = re.sub(r'(?<!\w)(?:/|[A-Za-z]:[\\/])[^\s,;:]+', '<path>', message)
    feedback = f"{event.get('error_type') or 'Error'}: {message[:ERROR_MESSAGE_MAX_CHARS]}"
    if event['reason'] == 'timeout':
        feedback = (f"TimeoutError: {message}. The limit covers the complete evaluation "
                    'batch. Reduce the computation of each call and ensure loops terminate.')
    else:
        location = _candidate_location(event.get('traceback'))
        if location:
            feedback += f'\nFailing call: line {location[0]} in {location[1]}().'
    baseline = (f"\nParent fitness: {event['parent_fitness']} (higher is better).\n"
                if event.get('parent_fitness') is not None else '')
    return (f'{task_contract}\n{baseline}\n# Failed output\n{_display_view(response)}\n\n'
            f"# Failure during {event['operator']}\n{feedback}\n\n"
            "# Repair\nCorrect the reported failure while preserving the candidate's "
            f'intended decision method.\n\n{OUTPUT}')
