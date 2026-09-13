"""Prompt construction for the function-level TraceAAD search."""

import ast
import hashlib

from .errors import OUTPUT

OPERATOR_INSTRUCTIONS = {
    "Init": (
        "Study the task and any previous initial algorithms. Identify a promising decision "
        "mechanism and implement one competitive candidate around it."
    ),
    "Refine": (
        "Use the current code and formation history to identify the most valuable next "
        "improvement. Implement one coherent refinement of its decision mechanism."
    ),
    "Tune": (
        "Preserve the current decision method and computational structure. Calibrate a "
        "small coherent set of influential coefficients, thresholds, exponents, or schedules "
        "to improve the decision."
    ),
    "Pivot": (
        "Use the task, current algorithm, and formation history to develop a competitive "
        "alternative. Build it around a different primary decision mechanism."
    ),
    "Fuse": (
        "Compare the host and donor to identify a host limitation that a donor computation "
        "can address. Adapt the relevant donor computation into one coherent host-centered "
        "mechanism and implement the fusion."
    ),
}


class TrajectoryBuilder:
    def __init__(self, llm, task_contract, *, max_tokens, max_events, lookup, all_nodes):
        self.llm = llm
        self.task_contract = task_contract
        self.max_tokens = max_tokens
        self.max_events = max_events
        self.lookup = lookup
        self.all_nodes = all_nodes
        self._counts = {}

    def count(self, text, *, chat=False):
        key = (chat, hashlib.sha256(text.encode()).hexdigest())
        if key not in self._counts:
            self._counts[key] = (self.llm.count_prompt_tokens(text) if chat
                                 else self.llm.count_tokens(text))
        return self._counts[key]

    def function_view(self, node):
        tree = ast.parse(node.code)
        functions = [item for item in tree.body
                     if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1:
            raise ValueError(f"node {node.id} must contain one target function")
        return ast.unparse(functions[0]).strip()

    def formation_edges(self, current):
        edges = []
        for _ in range(self.max_events):
            if current is None or current.parent_id is None:
                break
            source = self.lookup(current.parent_id)
            if source is None:
                raise ValueError("missing formation predecessor")
            edges.append((source, current))
            current = source
        return list(reversed(edges))

    def program(self, node, title):
        return f"# {title}\nFitness: {node.fitness}\n```python\n{self.function_view(node)}\n```"

    def build_initial(self):
        roots = sorted((node for node in self.all_nodes() if node.parent_id is None),
                       key=lambda node: node.id)
        parts = [self.task_contract, "# Search Context\nFitness is higher for better algorithms."]
        if roots:
            parts.append("# Previous Initial Algorithms\nEarlier evaluated functions, in generation order.")
            parts.extend(self.program(node, "Previous Initial Algorithm") for node in roots)
        parts.extend(["# Design Task\n" + OPERATOR_INSTRUCTIONS["Init"], "# Output\n" + OUTPUT])
        text = "\n\n\n".join(parts)
        self.check_capacity(text)
        return text

    def build(self, parent, operator, donor=None):
        parts = [self.task_contract, "# Search Context\nFitness is higher for better algorithms."]
        if parent is not None:
            parts.append(self.program(parent, "Host Algorithm" if operator == "Fuse" else "Current Algorithm"))
            edges = self.formation_edges(parent)
            if edges:
                history = [
                    "# Formation History",
                ]
                for index, (source, target) in enumerate(edges, 1):
                    history.append(f"Step {index} | {target.operator} | Fitness: {source.fitness} -> {target.fitness}")
                    if target.idea:
                        history.append("Idea: " + " ".join(target.idea.split()))
                parts.append("\n\n".join(history))
        if donor is not None:
            parts.append(self.program(donor, "Donor Algorithm"))
        parts.extend(["# Design Task\n" + OPERATOR_INSTRUCTIONS[operator], "# Output\n" + OUTPUT])
        text = "\n\n\n".join(parts)
        self.check_capacity(text)
        return text

    def check_capacity(self, text):
        if self.count(text, chat=True) > self.max_tokens:
            raise ValueError("complete prompt exceeds the model context budget")
