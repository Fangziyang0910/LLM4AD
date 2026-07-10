"""DerivationGraph —— Program Memory（结果库，ground truth）。

单父 DAG：每个 child 恰好一条入边。节点存 ProgramNode（含多目标/泛化字段），
边存 ImprovementEdge（含 operator/mechanism_tag/delta/outcome/generalization_signal）。
"""
from __future__ import annotations

from collections import defaultdict

from .schema import EdgeId, ImprovementEdge, NodeId, ProgramNode, StepInfo, Trajectory


class DerivationGraph:
    def __init__(self) -> None:
        self._next_node_id = 0
        self._next_edge_id = 0
        self._nodes: dict[NodeId, ProgramNode] = {}
        self._edges: dict[EdgeId, ImprovementEdge] = {}
        self._incoming_edge_by_child: dict[NodeId, EdgeId] = {}
        self._outgoing_edges_by_parent: dict[NodeId, list[EdgeId]] = defaultdict(list)

    # ---- nodes ----
    def add_node(self, *, code: str, idea: str, fitness: float | None, is_valid: bool, **fields) -> ProgramNode:
        node = ProgramNode(
            id=self._next_node_id,
            code=code,
            idea=idea,
            fitness=fitness,
            is_valid=is_valid,
            runtime=fields.get("runtime", 0.0),
            complexity=fields.get("complexity", 0),
            robustness=fields.get("robustness", 0.0),
            fitness_vector=fields.get("fitness_vector"),
            mechanism_tag=fields.get("mechanism_tag", "other"),
            confidence=fields.get("confidence", 1.0),
            iteration=fields.get("iteration"),
            sample_order=fields.get("sample_order"),
        )
        self._nodes[node.id] = node
        self._next_node_id += 1
        return node

    def get_node(self, node_id: NodeId) -> ProgramNode:
        return self._nodes[node_id]

    def has_node(self, node_id: NodeId) -> bool:
        return node_id in self._nodes

    def nodes(self) -> tuple[ProgramNode, ...]:
        return tuple(self._nodes.values())

    def valid_nodes(self) -> tuple[ProgramNode, ...]:
        return tuple(n for n in self._nodes.values() if n.is_valid and n.fitness is not None)

    # ---- edges ----
    def add_edge(self, *, parent_id: NodeId, child_id: NodeId, action: str, **fields) -> ImprovementEdge:
        if parent_id not in self._nodes:
            raise KeyError(f"unknown parent node: {parent_id}")
        if child_id not in self._nodes:
            raise KeyError(f"unknown child node: {child_id}")
        if parent_id == child_id:
            raise ValueError("an improvement edge cannot point to the same node")
        if child_id in self._incoming_edge_by_child:
            raise ValueError(f"child node already has a parent edge: {child_id}")

        edge = ImprovementEdge(
            id=self._next_edge_id,
            parent_id=parent_id,
            child_id=child_id,
            action=action,
            operator=fields.get("operator", "unknown"),
            mechanism_tag=fields.get("mechanism_tag", "other"),
            delta=fields.get("delta"),
            outcome=fields.get("outcome", "unknown"),
            generalization_signal=fields.get("generalization_signal", 0.0),
            iteration=fields.get("iteration"),
        )
        self._edges[edge.id] = edge
        self._incoming_edge_by_child[child_id] = edge.id
        self._outgoing_edges_by_parent[parent_id].append(edge.id)
        self._next_edge_id += 1
        return edge

    def get_edge(self, edge_id: EdgeId) -> ImprovementEdge:
        return self._edges[edge_id]

    def edges(self) -> tuple[ImprovementEdge, ...]:
        return tuple(self._edges.values())

    def outgoing_edges(self, node_id: NodeId) -> tuple[ImprovementEdge, ...]:
        if node_id not in self._nodes:
            raise KeyError(f"unknown node: {node_id}")
        return tuple(self._edges[eid] for eid in self._outgoing_edges_by_parent[node_id])

    def incoming_edge(self, node_id: NodeId) -> ImprovementEdge | None:
        eid = self._incoming_edge_by_child.get(node_id)
        return None if eid is None else self._edges[eid]

    # ---- trajectory view ----
    def trajectory_steps(self, trajectory: Trajectory) -> tuple[StepInfo, ...]:
        """把 trajectory 的 edge 序列展开成 StepInfo（供 credit/value 使用）。"""
        steps: list[StepInfo] = []
        for edge_id, parent_id, child_id in zip(
            trajectory.edge_ids, trajectory.node_ids[:-1], trajectory.node_ids[1:]
        ):
            edge = self._edges[edge_id]
            steps.append(
                StepInfo(
                    edge_id=edge.id,
                    parent_id=parent_id,
                    child_id=child_id,
                    operator=edge.operator,
                    mechanism_tag=edge.mechanism_tag,
                    delta=edge.delta,
                    outcome=edge.outcome,
                    generalization_signal=edge.generalization_signal,
                )
            )
        return tuple(steps)

    # ---- fitness range (归一化用) ----
    def fitness_range(self) -> tuple[float | None, float | None]:
        values = [n.fitness for n in self._nodes.values() if n.is_valid and n.fitness is not None]
        if not values:
            return None, None
        return min(values), max(values)
