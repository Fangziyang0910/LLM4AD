"""Small local search engine primitives for TraceAAD V10.11."""

from dataclasses import dataclass
import hashlib
import json
import math
import os


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_code(text: str) -> str:
    return "\n".join(text.replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_journal(path):
    if not path.exists():
        return []
    records = []
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def ess(probabilities):
    return 1.0 / sum(p * p for p in probabilities)


def softmax(scores, beta):
    maximum = max(scores)
    weights = [math.exp(beta * (score - maximum)) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def _ess(beta, scores):
    return ess(softmax(scores, beta))


def calibrate_beta(scores, target):
    target = min(len(scores), max(float(target), sum(score == max(scores) for score in scores)))
    if len(scores) <= 1 or _ess(0.0, scores) <= target:
        return 0.0, target, _ess(0.0, scores)
    high = 1.0
    for _ in range(60):
        if _ess(high, scores) <= target:
            break
        high *= 10.0
    low = 0.0
    for _ in range(80):
        middle = (low + high) / 2
        if _ess(middle, scores) > target:
            low = middle
        else:
            high = middle
    return high, target, _ess(high, scores)


class UnknownEvaluation(RuntimeError):
    pass


@dataclass
class Node:
    id: int
    code: str
    idea: str
    fitness: float
    evaluation_id: int | None = None
    parent_id: int | None = None
    operator: str = "Init"
    donor_id: int | None = None


class SearchTree:
    def __init__(self):
        self.nodes = {}
        self.children = {}
        self.roots = []
        self.next_id = 0

    def _attach(self, node):
        self.nodes[node.id] = node
        if node.parent_id is None:
            self.roots.append(node.id)
        else:
            self.children.setdefault(node.parent_id, []).append(node.id)

    def add(self, *, code, idea, fitness, evaluation_id, parent_id, operator, donor_id=None):
        node = Node(self.next_id, code, idea, fitness, evaluation_id, parent_id, operator, donor_id)
        self.next_id += 1
        self._attach(node)
        return node

    def add_raw(self, node):
        self._attach(node)
        self.next_id = max(self.next_id, node.id + 1)

    def all_nodes(self):
        return list(self.nodes.values())

    def best(self):
        return max(self.nodes.values(), key=lambda node: node.fitness)
