from __future__ import annotations

import ast
import copy
import math
import re
from typing import Any

from ...base import Function, LLM, Program, TextFunctionProgramConverter


GLOBAL_IMPORTS = [
    "import numpy as np",
    "import random",
    "import math",
    "import scipy",
    "try:\n    import torch\nexcept Exception:\n    torch = None",
]


class HSEvoSampler:
    def __init__(self, llm: LLM, template_program: str | Program):
        self.llm = llm
        self._template_program = template_program

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        return self.llm.draw_sample(prompt, *args, **kwargs)

    @staticmethod
    def with_global_imports(program: str | Program) -> Program:
        if isinstance(program, str):
            program = TextFunctionProgramConverter.text_to_program(program)
        else:
            program = copy.deepcopy(program)
        existing = set(line.strip() for line in program.preface.splitlines())
        additions = [line for line in GLOBAL_IMPORTS if line not in existing]
        preface_lines = [line for line in program.preface.splitlines() if line.strip()]
        preface = "\n".join(additions + preface_lines)
        return Program(preface=preface, functions=program.functions)

    @classmethod
    def strip_thinking(cls, text: str) -> str:
        text = text or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]
        return text.strip()

    @classmethod
    def fenced_blocks(cls, text: str) -> list[str]:
        text = cls.strip_thinking(text)
        pattern = r"```(?:python)?\s*(.*?)```"
        return [block.strip() for block in re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)]

    @classmethod
    def _function_source_from_valid_python(cls, source: str) -> str | None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                segment = ast.get_source_segment(source, node)
                if segment:
                    return segment.strip()
        return None

    @classmethod
    def _function_source_from_prefix(cls, source: str) -> str | None:
        lines = source.splitlines()
        def_lines = [idx for idx, line in enumerate(lines) if line.lstrip().startswith("def ")]
        for start in def_lines:
            for end in range(len(lines), start, -1):
                snippet = "\n".join(lines[start:end])
                found = cls._function_source_from_valid_python(snippet)
                if found:
                    return found
        return None

    @classmethod
    def extract_function_source(cls, response: str) -> str | None:
        text = cls.strip_thinking(response)
        candidates = cls.fenced_blocks(text)
        candidates.append(text)
        for candidate in candidates:
            found = cls._function_source_from_valid_python(candidate)
            if found:
                return found
            found = cls._function_source_from_prefix(candidate)
            if found:
                return found
        return None

    @classmethod
    def response_to_function(cls, response: str) -> Function | None:
        function_source = cls.extract_function_source(response)
        if function_source is None:
            return None
        return TextFunctionProgramConverter.text_to_function(function_source)

    @classmethod
    def parse_reflection(cls, response: str) -> dict[str, str]:
        response = cls.strip_thinking(response)
        analysis_marker = "**Analysis:**"
        experience_marker = "**Experience:**"
        if analysis_marker in response and experience_marker in response:
            analysis_start = response.find(analysis_marker) + len(analysis_marker)
            experience_start = response.find(experience_marker)
            analysis = response[analysis_start:experience_start].strip()
            experience = response[experience_start + len(experience_marker):].strip()
            return {"analyze": analysis, "exp": experience}
        return {"analyze": "", "exp": response.strip()}

    @staticmethod
    def _literal_parameter_ranges(source: str) -> dict[str, tuple[float, float]] | None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        value_node = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "parameter_ranges":
                    value_node = node.value
                    break
        if value_node is None:
            return None
        try:
            raw = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(raw, dict):
            return None
        ranges = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.isidentifier():
                return None
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                return None
            low, high = float(value[0]), float(value[1])
            if not math.isfinite(low) or not math.isfinite(high):
                return None
            if high < low:
                low, high = high, low
            ranges[key] = (low, high)
        return ranges

    @classmethod
    def parse_harmony_response(cls, response: str) -> tuple[dict[str, tuple[float, float]] | None, str | None]:
        blocks = cls.fenced_blocks(response)
        if len(blocks) < 2:
            return None, None

        function_source = None
        range_source = None
        for block in blocks:
            if function_source is None:
                function_source = cls.extract_function_source(block)
                if function_source is not None:
                    continue
            if range_source is None and "parameter_ranges" in block:
                range_source = block

        if function_source is None or range_source is None:
            return None, None
        parameter_ranges = cls._literal_parameter_ranges(range_source)
        if not parameter_ranges:
            return None, None
        return parameter_ranges, function_source

    @classmethod
    def function_with_harmony_values(cls, function_source: str, values: dict[str, float]) -> Function | None:
        func = TextFunctionProgramConverter.text_to_function(function_source)
        if func is None:
            return None
        func = copy.deepcopy(func)
        assignments = [f"    {name} = {repr(float(value))}" for name, value in values.items()]
        body_lines = func.body.splitlines()
        func.body = "\n".join(assignments + body_lines)
        return func
