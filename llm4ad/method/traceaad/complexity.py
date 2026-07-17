"""代码复杂度度量 —— 对齐 ShinkaEvolve radon 管线（合成分 + 分项指标）。

合成分权重与 Shinka 一致：0.4*CC + 0.4*Halstead + 0.1*LOC + 0.1*nesting。
归一化常数仅用于合成 raw score；轨迹价值层另做活跃池相对归一化。
"""
from __future__ import annotations

import ast
import math
from typing import Mapping

from radon.complexity import cc_visit
from radon.metrics import h_visit
from radon.raw import analyze


def max_nesting_depth(code: str) -> int:
    """AST 最大嵌套深度（If/For/While/With/Try/FunctionDef）。"""

    class NestingVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.current_depth = 0
            self.max_depth = 0

        def generic_visit(self, node: ast.AST) -> None:
            if isinstance(
                node,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.With,
                    ast.Try,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                super().generic_visit(node)
                self.current_depth -= 1
            else:
                super().generic_visit(node)

    tree = ast.parse(code)
    visitor = NestingVisitor()
    visitor.visit(tree)
    return visitor.max_depth


def _ast_node_count(code: str) -> int:
    try:
        return sum(1 for _ in ast.walk(ast.parse(code)))
    except Exception:
        return code.count("\n") + 1


def _fallback_metrics(code: str) -> dict[str, float]:
    loc = float(code.count("\n") + 1)
    nodes = float(_ast_node_count(code))
    # 粗合成：节点数代理 CC/体积，保证失败时仍有可比标量
    score = min(0.4 * (nodes / 50.0) + 0.6 * (math.log2(loc + 1) / 10.0), 1.0)
    return {
        "cyclomatic_complexity": nodes,
        "halstead_volume": nodes,
        "halstead_difficulty": 0.0,
        "halstead_effort": 0.0,
        "lines_of_code": loc,
        "logical_lines_of_code": loc,
        "max_nesting_depth": 0.0,
        "maintainability_index": 0.0,
        "complexity_score": round(score, 3),
        "ast_node_count": nodes,
    }


def analyze_code_complexity(code: str) -> Mapping[str, float]:
    """返回不可变 metrics；`complexity_score` ∈ [0, 1] 为合成复杂度（越大越复杂）。"""
    try:
        cc_results = cc_visit(code)
        total_cc = float(sum(block.complexity for block in cc_results))

        h_metrics = h_visit(code)
        halstead_total = h_metrics.total if h_metrics.total else None
        halstead_volume = float(halstead_total.volume) if halstead_total else 1.0
        halstead_difficulty = float(halstead_total.difficulty) if halstead_total else 0.0
        halstead_effort = float(halstead_total.effort) if halstead_total else 0.0

        raw_metrics = analyze(code)
        loc = float(raw_metrics.loc)
        lloc = float(raw_metrics.lloc)

        mi = (
            171.0
            - 5.2 * (math.log2(halstead_volume) if halstead_volume > 0 else 0.0)
            - 0.23 * total_cc
            - 16.2 * (math.log2(loc) if loc > 0 else 0.0)
        )
        nesting_depth = float(max_nesting_depth(code))
        ast_nodes = float(_ast_node_count(code))

        norm_cc = total_cc / 10.0
        norm_halstead = math.log2(halstead_volume + 1.0) / 10.0
        norm_loc = math.log2(loc + 1.0) / 10.0
        norm_nesting = nesting_depth / 5.0
        complexity_score = min(
            0.4 * norm_cc + 0.4 * norm_halstead + 0.1 * norm_loc + 0.1 * norm_nesting,
            1.0,
        )
        metrics = {
            "cyclomatic_complexity": total_cc,
            "halstead_volume": halstead_volume,
            "halstead_difficulty": halstead_difficulty,
            "halstead_effort": halstead_effort,
            "lines_of_code": loc,
            "logical_lines_of_code": lloc,
            "max_nesting_depth": nesting_depth,
            "maintainability_index": float(mi),
            "complexity_score": round(complexity_score, 3),
            "ast_node_count": ast_nodes,
        }
        return metrics
    except Exception:
        return _fallback_metrics(code)


def complexity_score_of(code: str) -> float:
    return float(analyze_code_complexity(code)["complexity_score"])
