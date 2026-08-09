"""Deterministic source facts used by TraceAAD V9.2."""

from __future__ import annotations

import ast
import hashlib


def normalized_source(code: str) -> str:
    text = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


class _DocstringRemover(ast.NodeTransformer):
    @staticmethod
    def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        return body or [ast.Pass()]

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body = self._without_docstring(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = self._without_docstring(node.body)
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.body = self._without_docstring(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.body = self._without_docstring(node.body)
        return node


def comment_free_source(code: str) -> str:
    """Return an executable-only representation with no comments or docstrings."""
    tree = ast.parse(normalized_source(code))
    cleaned = _DocstringRemover().visit(tree)
    ast.fix_missing_locations(cleaned)
    return ast.unparse(cleaned).strip()


def nonempty_loc(code: str) -> int:
    return sum(bool(line.strip()) for line in normalized_source(code).splitlines())


def code_hash(code: str) -> str:
    return hashlib.sha256(normalized_source(code).encode("utf-8")).hexdigest()


def code_change_ratio(parent: str, child: str) -> float:
    left = normalized_source(parent).splitlines()
    right = normalized_source(child).splitlines()
    denominator = max(len(left), len(right), 1)
    shared = sum(a == b for a, b in zip(left, right))
    return 1.0 - shared / denominator


__all__ = [
    "code_change_ratio",
    "code_hash",
    "comment_free_source",
    "nonempty_loc",
    "normalized_source",
]
