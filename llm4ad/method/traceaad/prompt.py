from __future__ import annotations

import copy

from ...base import Function
from .derivation_graph import DerivationGraph
from .schema import ImprovementEdge, ProgramNode, Trajectory


class TraceAADPrompt:
    @classmethod
    def build_initial_prompt(
            cls,
            *,
            task_description: str,
            template_function: Function,
            diversity_hint: str,
    ) -> str:
        target = copy.deepcopy(template_function)
        target.body = ""
        return "\n".join([
            task_description.strip(),
            "",
            f"Generate a complete implementation for the target Python function. {diversity_hint}",
            "Keep the function name, arguments, return type, and output contract unchanged.",
            "",
            "Output format:",
            "Idea: <brief algorithm idea>",
            "Code:",
            "```python",
            str(target).rstrip(),
            "```",
        ]).strip()

    @classmethod
    def build_action_prompt(
            cls,
            *,
            graph: DerivationGraph,
            trajectory: Trajectory,
            task_description: str,
            max_actions: int = 5,
            action_count: int = 4,
            maximize: bool = True,
            base_node_id: int | None = None,
            base_selection_reason: str | None = None,
    ) -> str:
        if max_actions < 0:
            raise ValueError("max_actions must be non-negative")
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        if len(trajectory.edge_ids) != len(trajectory.node_ids) - 1:
            raise ValueError("trajectory edge count must equal node count minus one")
        if trajectory.endpoint_id != trajectory.node_ids[-1]:
            raise ValueError("trajectory endpoint must be the last node")
        if base_node_id is None:
            base_node_id = trajectory.endpoint_id
        if base_node_id not in trajectory.node_ids:
            raise ValueError("base_node_id must belong to the trajectory")

        node_ids, edge_ids = cls._bounded_ids(trajectory, max_actions, base_node_id=base_node_id)
        nodes = tuple(graph.get_node(node_id) for node_id in node_ids)
        edges = tuple(graph.get_edge(edge_id) for edge_id in edge_ids)
        base_node = graph.get_node(base_node_id)
        reason = base_selection_reason or "endpoint"

        sections = [
            "[Task Description]",
            task_description.strip(),
            "",
            "[Algorithm Improvement Context]",
            "The selected trajectory records attempted modifications and observed outcomes.",
            cls._fitness_direction_hint(maximize),
            cls._format_trajectory(nodes, edges, maximize=maximize),
            "",
            "[Base Program To Modify]",
            f"Continue from Node p{base_node.id}. Selection reason: {reason}.",
            cls._format_current_node(base_node),
            "",
            "[Instruction]",
            "Use the trajectory as a record of attempted modifications and outcomes.",
            "Focus on which action directions improved, regressed, or stopped changing fitness.",
            f"Propose {action_count} next-step modifications for the base program above:",
            "- each modification must change one main algorithmic mechanism only;",
            "- avoid repeating directions that regressed or stayed unchanged;",
            "- a modification may continue a direction that improved, or try a different direction after saturation.",
            "Each modification must be concrete and implementable. Do not output code or rationale.",
            f"Return only a numbered list of exactly {action_count} ideas, one per line.",
        ]
        return "\n".join(sections).strip()

    @classmethod
    def build_code_prompt(
            cls,
            *,
            current_node: ProgramNode,
            action: str,
            task_description: str,
            template_function: Function,
    ) -> str:
        action = action.strip()
        if not action:
            raise ValueError("action must not be empty")
        target = copy.deepcopy(template_function)
        target.body = ""
        return "\n".join([
            "[Task Description]",
            task_description.strip(),
            "",
            "[Current Program]",
            cls._format_current_node(current_node),
            "",
            "[Requested Modification]",
            action,
            "",
            "[Target Function Contract]",
            str(target).rstrip(),
            "",
            "[Instruction]",
            "Implement the requested modification as a new complete implementation of the target function.",
            "Keep the function name, arguments, return type, and output contract unchanged.",
            "Return only the new idea and complete code in this format:",
            "Idea: <brief algorithm idea>",
            "Code:",
            "```python",
            "<complete function implementation>",
            "```",
            "Do not include rationale, analysis, tests, or extra text.",
        ]).strip()

    @staticmethod
    def _bounded_ids(
            trajectory: Trajectory,
            max_actions: int,
            *,
            base_node_id: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if max_actions == 0:
            return (base_node_id,), ()
        edge_count = len(trajectory.edge_ids)
        start = max(0, edge_count - max_actions)
        end = edge_count
        base_index = trajectory.node_ids.index(base_node_id)
        if base_index < start:
            start = base_index
            end = min(edge_count, start + max_actions)
        return trajectory.node_ids[start: end + 1], trajectory.edge_ids[start:end]

    @classmethod
    def _format_trajectory(
            cls,
            nodes: tuple[ProgramNode, ...],
            edges: tuple[ImprovementEdge, ...],
            *,
            maximize: bool,
    ) -> str:
        if not edges:
            return "No previous improvement actions exist yet. This is an initial program."

        lines: list[str] = []
        for step, node in enumerate(nodes[:-1]):
            next_node = nodes[step + 1]
            edge = edges[step]
            lines.extend([
                f"Step {step}: p{node.id} -> p{next_node.id}",
                f"Parent idea: {node.idea}",
                "Action tried:",
                edge.action,
                f"Child idea: {next_node.idea}",
                f"Fitness: {cls._format_fitness(node.fitness)} -> {cls._format_fitness(next_node.fitness)}",
                f"Fitness change: {cls._format_fitness_change(node, next_node)}",
                f"Outcome: {cls._format_outcome(node, next_node, maximize=maximize)}",
                "",
            ])
        return "\n".join(lines).rstrip()

    @classmethod
    def _format_current_node(cls, node: ProgramNode) -> str:
        return "\n".join([
            f"Node p{node.id}",
            f"Idea: {node.idea}",
            f"Fitness: {cls._format_fitness(node.fitness)}",
            "Code:",
            "```python",
            node.code.rstrip(),
            "```",
        ])

    @staticmethod
    def _format_fitness(fitness: float | None) -> str:
        if fitness is None:
            return "unknown"
        return f"{fitness:.6g}"

    @classmethod
    def _format_fitness_change(cls, parent: ProgramNode, child: ProgramNode) -> str:
        if parent.fitness is None or child.fitness is None:
            return "unknown"
        return f"{child.fitness - parent.fitness:+.6g}"

    @classmethod
    def _format_outcome(cls, parent: ProgramNode, child: ProgramNode, *, maximize: bool) -> str:
        if parent.fitness is None or child.fitness is None:
            return "unknown"
        delta = child.fitness - parent.fitness if maximize else parent.fitness - child.fitness
        if delta > 1e-12:
            return "improved"
        if delta < -1e-12:
            return "regressed"
        return "unchanged"

    @staticmethod
    def _fitness_direction_hint(maximize: bool) -> str:
        if maximize:
            return (
                "Fitness is this task's score and higher is better; "
                "a positive fitness change is an improvement."
            )
        return (
            "Fitness is this task's metric and lower is better; "
            "a negative raw fitness change is an improvement."
        )
