"""Search tree for TraceAAD V10.1 (design doc section 2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

INIT = "Init"
OPERATORS = ("Refine", "Pivot", "Fuse")


def normalize_code(text: str) -> str:
    """Unified newlines, strip leading/trailing whitespace (dedup key)."""
    return "\n".join(text.replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()


@dataclass
class Node:
    id: int
    code: str
    idea: str
    fitness: float
    parent_id: int | None
    origin_operator: str
    donor_id: int | None


class SearchTree:
    """Single-parent program tree; children indexed by ``parent_id``."""

    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.children: dict[int, list[int]] = {}
        self.roots: list[int] = []
        self.next_id = 0

    def ancestors(self, node_id: int) -> list[Node]:
        """Ancestors of a node, nearest parent first."""
        chain: list[Node] = []
        current = self.nodes[node_id].parent_id
        while current is not None:
            chain.append(self.nodes[current])
            current = self.nodes[current].parent_id
        return chain

    def descendants(self, node_id: int) -> set[int]:
        found: set[int] = set()
        stack = list(self.children.get(node_id, []))
        while stack:
            child = stack.pop()
            if child in found:
                continue
            found.add(child)
            stack.extend(self.children.get(child, []))
        return found

    def add(
        self,
        *,
        code: str,
        idea: str,
        fitness: float,
        parent_id: int | None,
        origin_operator: str,
        donor_id: int | None = None,
    ) -> Node | None:
        """Insert a node under the unified dedup rules; return None on duplicate.

        Roots must be globally unique; a child must differ from its parent and
        from every existing sibling.
        """
        if parent_id is None:
            if any(self.nodes[r].code == code for r in self.roots):
                return None
        else:
            if self.nodes[parent_id].code == code:
                return None
            if any(
                self.nodes[c].code == code for c in self.children.get(parent_id, [])
            ):
                return None
        node = Node(
            id=self.next_id,
            code=code,
            idea=idea,
            fitness=fitness,
            parent_id=parent_id,
            origin_operator=origin_operator,
            donor_id=donor_id,
        )
        self.next_id += 1
        self.nodes[node.id] = node
        if parent_id is None:
            self.roots.append(node.id)
        else:
            self.children.setdefault(parent_id, []).append(node.id)
        return node

    def add_raw(self, node: Node) -> None:
        """Restore a serialized node without dedup checks."""
        self.nodes[node.id] = node
        if node.parent_id is None:
            self.roots.append(node.id)
        else:
            self.children.setdefault(node.parent_id, []).append(node.id)
        self.next_id = max(self.next_id, node.id + 1)

    def all_nodes(self) -> list[Node]:
        return list(self.nodes.values())

    def best(self) -> Node:
        return max(self.nodes.values(), key=lambda n: n.fitness)

    def to_state(self) -> list[dict]:
        return [asdict(n) for n in self.nodes.values()]
