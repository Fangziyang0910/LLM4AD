"""Idea parsing and target-function extraction for TraceAAD V10.11."""

import ast
import re

from llm4ad.base import TextFunctionProgramConverter
from .core import normalize_code

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"^[ \t]*```(?:python|py)?[ \t]*\r?$", re.MULTILINE | re.IGNORECASE)
OUTPUT = (
    "Return exactly:\n"
    "Idea: <one short paragraph, at most 200 words, describing the new mechanism>\n"
    "Code:\n```python\n<the complete definition of the target function from the template>\n```\n\n"
    "Use the exact function name and signature shown in the template."
)


def signature(args):
    return ([arg.arg for arg in args.posonlyargs], [arg.arg for arg in args.args],
            [arg.arg for arg in args.kwonlyargs],
            args.vararg.arg if args.vararg else None,
            args.kwarg.arg if args.kwarg else None)


def expected_interface(name, args_text):
    args = ast.parse(f"def target({args_text}):\n    pass").body[0].args
    return name, args_text, signature(args)


def parse_candidate(response, finish_reason, interface, template_program):
    if finish_reason not in ("stop", "length", "unknown"):
        return None, "unsupported finish reason"
    text = THINK_BLOCK_RE.sub("", response)
    fences = list(FENCE_RE.finditer(text))
    if len(fences) != 2:
        return None, "format_error: expected exactly one Python code block"
    prefix = text[:fences[0].start()]
    if text[fences[1].end():].strip():
        return None, "format_error: no text is allowed after the code block"
    idea_match = re.fullmatch(r"[ \t]*Idea:[ \t]*(.*?)[ \t]*\n[ \t]*Code:[ \t]*\n?", prefix,
                              re.IGNORECASE | re.DOTALL)
    if idea_match is None or not idea_match.group(1).strip():
        return None, "Idea is required"
    idea = idea_match.group(1).strip()
    if len(idea.split()) > 200:
        return None, "idea_length_error: Idea must be at most 200 words"
    code = text[fences[0].end():fences[1].start()]
    canonical = normalize_code(code)
    if not canonical:
        return None, "The code block is empty"
    try:
        tree = ast.parse(canonical)
    except (SyntaxError, ValueError) as exc:
        return None, f"syntax_error: {type(exc).__name__}: {exc}"
    name, args_text, expected = interface
    targets = [node for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(tree.body) != 1 or len(targets) != 1:
        return None, f"code_shape_error: code must contain only the `{name}(...)` function definition"
    if signature(targets[0].args) != expected:
        return None, f"signature_error: `{name}` must declare the parameters ({args_text})"
    target = TextFunctionProgramConverter.text_to_function(ast.unparse(targets[0]))
    if target is None:
        return None, "code_shape_error: could not extract the target function"
    program = TextFunctionProgramConverter.function_to_program(target, template_program)
    if program is None:
        return None, "template_error: could not rebuild the candidate from the task template"
    rebuilt = normalize_code(str(program))
    try:
        compile(rebuilt, "<candidate-template>", "exec")
    except (SyntaxError, ValueError) as exc:
        return None, f"template_error: {type(exc).__name__}: {exc}"
    return (idea, normalize_code(ast.unparse(targets[0])), rebuilt), None


def repair_prompt(task_contract, response, event):
    message = (event.get("error") or event.get("reason") or "Evaluation failed").strip()
    return (f"{task_contract}\n\n# Failed output\n{THINK_BLOCK_RE.sub('', response)}\n\n"
            f"# Failure\n{event.get('error_type', 'Error')}: {message[:2000]}\n\n"
            "# Repair\nCorrect the failure while preserving the intended decision method, then return the required output format.\n\n"
            + OUTPUT)
