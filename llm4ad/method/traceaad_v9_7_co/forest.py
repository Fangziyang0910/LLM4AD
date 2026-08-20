"""Programs, anchors, and finalized generation attempts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .schema import Anchor, Attempt, Outcome, Program
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
    def __init__(self, *, maximize: bool) -> None:
        self.maximize = maximize
        self._programs: dict[int, Program] = {}
        self._by_hash: dict[str, int] = {}
        self._anchors: dict[int, Anchor] = {}
        self._attempts: dict[int, Attempt] = {}
        self._relations: set[tuple[int, int]] = set()
        self.root_ids: list[int] = []
        self._next_program_id = 0
        self._next_anchor_id = 0
        self._next_attempt_id = 0

    def programs(self) -> tuple[Program, ...]:
        return tuple(self._programs.values())

    def anchors(self) -> tuple[Anchor, ...]:
        return tuple(self._anchors.values())

    def attempts(self) -> tuple[Attempt, ...]:
        return tuple(self._attempts.values())

    def get_program(self, program_id: int) -> Program:
        return self._programs[program_id]

    def get_anchor(self, anchor_id: int) -> Anchor:
        return self._anchors[anchor_id]

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

    def add_root(self, *, program_id: int, order: int) -> Anchor:
        anchor = Anchor(
            id=self._next_anchor_id,
            program_id=program_id,
            parent_id=None,
            attempt_id=None,
            root_id=self._next_anchor_id,
            order=order,
        )
        self._next_anchor_id += 1
        self._anchors[anchor.id] = anchor
        self.root_ids.append(anchor.id)
        return anchor

    def add_child(
        self,
        *,
        parent_id: int,
        program_id: int,
        attempt_id: int,
        order: int,
    ) -> Anchor:
        anchor = Anchor(
            id=self._next_anchor_id,
            program_id=program_id,
            parent_id=parent_id,
            attempt_id=attempt_id,
            root_id=self.get_anchor(parent_id).root_id,
            order=order,
        )
        self._next_anchor_id += 1
        self._anchors[anchor.id] = anchor
        self._relations.add((parent_id, program_id))
        return anchor

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximize": self.maximize,
            "next_program_id": self._next_program_id,
            "next_anchor_id": self._next_anchor_id,
            "next_attempt_id": self._next_attempt_id,
            "root_ids": list(self.root_ids),
            "programs": [asdict(item) for item in self.programs()],
            "anchors": [asdict(item) for item in self.anchors()],
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
        for item in payload["attempts"]:
            if item["outcome"] is not None:
                item = {**item, "outcome": Outcome(item["outcome"])}
            attempt = Attempt(**item)
            forest._attempts[attempt.id] = attempt
        forest.root_ids = list(payload["root_ids"])
        forest._next_program_id = payload["next_program_id"]
        forest._next_anchor_id = payload["next_anchor_id"]
        forest._next_attempt_id = payload["next_attempt_id"]
        return forest


__all__ = ["Forest", "is_better"]
