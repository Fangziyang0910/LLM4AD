"""Programs, anchors, hypotheses, and finalized generation attempts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .schema import Anchor, Attempt, Hypothesis, Outcome, Program
from .source import code_hash


def is_better(candidate: Program, incumbent: Program | None) -> bool:
    if incumbent is None:
        return True
    return (candidate.q, -candidate.length, -candidate.order) > (
        incumbent.q,
        -incumbent.length,
        -incumbent.order,
    )


class Forest:
    """Fact store. Hypothesis identity is an Explore-created trajectory segment."""

    def __init__(self, *, maximize: bool) -> None:
        self.maximize = maximize
        self._programs: dict[int, Program] = {}
        self._by_hash: dict[str, int] = {}
        self._anchors: dict[int, Anchor] = {}
        self._hypotheses: dict[int, Hypothesis] = {}
        self._attempts: dict[int, Attempt] = {}
        self._relations: set[tuple[int, int]] = set()
        self.root_ids: list[int] = []
        self.root_hypothesis_ids: list[int] = []
        self._next_program_id = 0
        self._next_anchor_id = 0
        self._next_hypothesis_id = 0
        self._next_attempt_id = 0

    def programs(self) -> tuple[Program, ...]:
        return tuple(self._programs.values())

    def anchors(self) -> tuple[Anchor, ...]:
        return tuple(self._anchors.values())

    def hypotheses(self) -> tuple[Hypothesis, ...]:
        return tuple(self._hypotheses.values())

    def attempts(self) -> tuple[Attempt, ...]:
        return tuple(self._attempts.values())

    def best(self) -> Program | None:
        best = None
        for program in self._programs.values():
            if is_better(program, best):
                best = program
        return best

    def get_program(self, program_id: int) -> Program:
        return self._programs[program_id]

    def get_anchor(self, anchor_id: int) -> Anchor:
        return self._anchors[anchor_id]

    def get_hypothesis(self, hypothesis_id: int) -> Hypothesis:
        return self._hypotheses[hypothesis_id]

    def get_attempt(self, attempt_id: int) -> Attempt:
        return self._attempts[attempt_id]

    def next_attempt_id(self) -> int:
        attempt_id = self._next_attempt_id
        self._next_attempt_id += 1
        return attempt_id

    def program_for_code(self, code: str) -> Program | None:
        program_id = self._by_hash.get(code_hash(code))
        return None if program_id is None else self.get_program(program_id)

    def add_program(self, *, code: str, fitness: float, order: int) -> Program:
        program = Program(
            id=self._next_program_id,
            code=code,
            fitness=fitness,
            q=fitness if self.maximize else -fitness,
            length=len(code),
            order=order,
        )
        self._next_program_id += 1
        self._programs[program.id] = program
        self._by_hash[code_hash(code)] = program.id
        return program

    def add_root(self, *, program_id: int, order: int) -> tuple[Anchor, Hypothesis]:
        program = self.get_program(program_id)
        anchor_id = self._next_anchor_id
        hypothesis_id = self._next_hypothesis_id
        anchor = Anchor(
            id=anchor_id,
            program_id=program_id,
            parent_id=None,
            attempt_id=None,
            root_id=anchor_id,
            hypothesis_id=hypothesis_id,
            order=order,
        )
        hypothesis = Hypothesis(
            id=hypothesis_id,
            entry_anchor_id=anchor_id,
            parent_hypothesis_id=None,
            creation_attempt_id=None,
            root_id=anchor_id,
            q0=program.q,
            q_base=None,
            order=order,
        )
        self._next_anchor_id += 1
        self._next_hypothesis_id += 1
        self._anchors[anchor.id] = anchor
        self._hypotheses[hypothesis.id] = hypothesis
        self.root_ids.append(anchor.id)
        self.root_hypothesis_ids.append(hypothesis.id)
        return anchor, hypothesis

    def add_refine_child(
        self,
        *,
        parent_id: int,
        program_id: int,
        attempt_id: int,
        order: int,
    ) -> Anchor:
        parent = self.get_anchor(parent_id)
        anchor = Anchor(
            id=self._next_anchor_id,
            program_id=program_id,
            parent_id=parent_id,
            attempt_id=attempt_id,
            root_id=parent.root_id,
            hypothesis_id=parent.hypothesis_id,
            order=order,
        )
        self._next_anchor_id += 1
        self._anchors[anchor.id] = anchor
        self._relations.add((parent_id, program_id))
        return anchor

    def add_explore_child(
        self,
        *,
        parent_id: int,
        program_id: int,
        attempt_id: int,
        order: int,
    ) -> tuple[Anchor, Hypothesis]:
        parent_anchor = self.get_anchor(parent_id)
        parent_program = self.get_program(parent_anchor.program_id)
        program = self.get_program(program_id)
        hypothesis_id = self._next_hypothesis_id
        anchor = Anchor(
            id=self._next_anchor_id,
            program_id=program_id,
            parent_id=parent_id,
            attempt_id=attempt_id,
            root_id=parent_anchor.root_id,
            hypothesis_id=hypothesis_id,
            order=order,
        )
        hypothesis = Hypothesis(
            id=hypothesis_id,
            entry_anchor_id=anchor.id,
            parent_hypothesis_id=parent_anchor.hypothesis_id,
            creation_attempt_id=attempt_id,
            root_id=parent_anchor.root_id,
            q0=program.q,
            q_base=parent_program.q,
            order=order,
        )
        self._next_anchor_id += 1
        self._next_hypothesis_id += 1
        self._anchors[anchor.id] = anchor
        self._hypotheses[hypothesis.id] = hypothesis
        self._relations.add((parent_id, program_id))
        return anchor, hypothesis

    def add_attempt(self, attempt: Attempt) -> None:
        self._attempts[attempt.id] = attempt

    def relation_exists(self, parent_id: int, program_id: int) -> bool:
        return (parent_id, program_id) in self._relations

    def ancestor_ids(self, anchor_id: int) -> tuple[int, ...]:
        path: list[int] = []
        current: int | None = anchor_id
        while current is not None:
            path.append(current)
            current = self.get_anchor(current).parent_id
        return tuple(reversed(path))

    def ancestor_program_ids(self, anchor_id: int) -> tuple[int, ...]:
        return tuple(
            self.get_anchor(item).program_id for item in self.ancestor_ids(anchor_id)[:-1]
        )

    def parent_path_ids(self, anchor_id: int) -> tuple[int, ...]:
        ids: list[int] = []
        for path_id in self.ancestor_ids(anchor_id)[1:]:
            attempt_id = self.get_anchor(path_id).attempt_id
            if attempt_id is not None:
                ids.append(attempt_id)
        return tuple(ids)

    def anchors_in_hypothesis(self, hypothesis_id: int) -> tuple[Anchor, ...]:
        return tuple(
            anchor for anchor in self.anchors() if anchor.hypothesis_id == hypothesis_id
        )

    def anchors_in_route(self, root_id: int) -> tuple[Anchor, ...]:
        return tuple(anchor for anchor in self.anchors() if anchor.root_id == root_id)

    def hypothesis_frontier(self, hypothesis_id: int) -> float:
        anchors = self.anchors_in_hypothesis(hypothesis_id)
        if not anchors:
            raise ValueError("hypothesis has no anchors")
        return max(self.get_program(anchor.program_id).q for anchor in anchors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximize": self.maximize,
            "next_program_id": self._next_program_id,
            "next_anchor_id": self._next_anchor_id,
            "next_hypothesis_id": self._next_hypothesis_id,
            "next_attempt_id": self._next_attempt_id,
            "root_ids": list(self.root_ids),
            "root_hypothesis_ids": list(self.root_hypothesis_ids),
            "programs": [asdict(item) for item in self.programs()],
            "anchors": [asdict(item) for item in self.anchors()],
            "hypotheses": [asdict(item) for item in self.hypotheses()],
            "attempts": [asdict(item) for item in self.attempts()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Forest:
        forest = cls(maximize=bool(payload["maximize"]))
        for item in payload["programs"]:
            program = Program(**item)
            forest._programs[program.id] = program
            forest._by_hash[code_hash(program.code)] = program.id
        for item in payload["anchors"]:
            anchor = Anchor(**item)
            forest._anchors[anchor.id] = anchor
            if anchor.parent_id is not None:
                forest._relations.add((anchor.parent_id, anchor.program_id))
        for item in payload["hypotheses"]:
            hypothesis = Hypothesis(**item)
            forest._hypotheses[hypothesis.id] = hypothesis
        for item in payload["attempts"]:
            converted = dict(item)
            if converted["outcome"] is not None:
                converted["outcome"] = Outcome(converted["outcome"])
            attempt = Attempt(**converted)
            forest._attempts[attempt.id] = attempt
        forest.root_ids = [int(item) for item in payload["root_ids"]]
        forest.root_hypothesis_ids = [
            int(item) for item in payload["root_hypothesis_ids"]
        ]
        forest._next_program_id = int(payload["next_program_id"])
        forest._next_anchor_id = int(payload["next_anchor_id"])
        forest._next_hypothesis_id = int(payload["next_hypothesis_id"])
        forest._next_attempt_id = int(payload["next_attempt_id"])
        return forest


__all__ = ["Forest", "is_better"]
