"""Programs, anchors, actions, and windowed subtree queries."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .schema import Action, ActionStatus, Anchor, Intent, Outcome, Program
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
    """Fact store. Online allocation reads anchors, actions, and formation paths."""

    def __init__(self, *, maximize: bool) -> None:
        self.maximize = maximize
        self._programs: dict[int, Program] = {}
        self._by_hash: dict[str, int] = {}
        self._anchors: dict[int, Anchor] = {}
        self._actions: dict[int, Action] = {}
        self._actions_by_arm: dict[tuple[int, str], list[Action]] = {}
        self._children: dict[int, list[int]] = {}
        self._relations: set[tuple[int, int]] = set()
        self.root_ids: list[int] = []
        self._next_program_id = 0
        self._next_anchor_id = 0
        self._next_action_id = 0

    def programs(self) -> tuple[Program, ...]:
        return tuple(self._programs.values())

    def anchors(self) -> tuple[Anchor, ...]:
        return tuple(self._anchors.values())

    def actions(self) -> tuple[Action, ...]:
        return tuple(self._actions.values())

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

    def get_action(self, action_id: int) -> Action:
        return self._actions[action_id]

    def next_action_id(self) -> int:
        action_id = self._next_action_id
        self._next_action_id += 1
        return action_id

    def program_for_code(self, code: str) -> Program | None:
        program_id = self._by_hash.get(code_hash(code))
        return None if program_id is None else self.get_program(program_id)

    def add_program(self, *, code: str, fitness: float, order: int) -> Program:
        if self.program_for_code(code) is not None:
            raise ValueError("program code is already present")
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
        self._by_hash[code_hash(program.code)] = program.id
        return program

    def add_root(self, *, program_id: int, order: int) -> Anchor:
        anchor_id = self._next_anchor_id
        anchor = Anchor(
            id=anchor_id,
            program_id=program_id,
            parent_id=None,
            action_id=None,
            root_id=anchor_id,
            order=order,
        )
        self._next_anchor_id += 1
        self._anchors[anchor.id] = anchor
        self._children.setdefault(anchor.id, [])
        self.root_ids.append(anchor.id)
        return anchor

    def add_child(
        self,
        *,
        parent_id: int,
        program_id: int,
        action_id: int,
        order: int,
    ) -> Anchor:
        parent = self.get_anchor(parent_id)
        anchor = Anchor(
            id=self._next_anchor_id,
            program_id=program_id,
            parent_id=parent_id,
            action_id=action_id,
            root_id=parent.root_id,
            order=order,
        )
        self._next_anchor_id += 1
        self._anchors[anchor.id] = anchor
        self._children.setdefault(anchor.id, [])
        self._children[parent_id].append(anchor.id)
        self._relations.add((parent_id, program_id))
        return anchor

    def add_action(self, action: Action) -> None:
        self._actions[action.id] = action
        if action.intent is not None:
            self._actions_by_arm.setdefault((action.anchor_id, action.intent), []).append(
                action
            )

    def actions_for_arm(self, anchor_id: int, intent: Intent | str) -> tuple[Action, ...]:
        key = (anchor_id, Intent(intent).value)
        return tuple(self._actions_by_arm.get(key, ()))

    def children(self, anchor_id: int) -> tuple[int, ...]:
        return tuple(self._children.get(anchor_id, ()))

    def window_stats(self, anchor_id: int, *, max_depth: int) -> tuple[float, int]:
        """Best observed descendant quality and depth within max_depth.

        The anchor itself counts at depth 0. The reported depth is capped at
        max_depth: settlement only distinguishes whether the observation
        window has been reached.
        """
        anchor = self.get_anchor(anchor_id)
        best_q = self.get_program(anchor.program_id).q
        depth_found = 0
        frontier: list[tuple[int, int]] = [(anchor_id, 0)]
        while frontier:
            current_id, depth = frontier.pop()
            if depth > 0:
                program = self.get_program(self.get_anchor(current_id).program_id)
                if program.q > best_q:
                    best_q = program.q
                if depth > depth_found:
                    depth_found = depth
            if depth >= max_depth:
                continue
            for child_id in self.children(current_id):
                frontier.append((child_id, depth + 1))
        return best_q, depth_found

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
            action_id = self.get_anchor(path_id).action_id
            if action_id is not None:
                ids.append(action_id)
        return tuple(ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximize": self.maximize,
            "next_program_id": self._next_program_id,
            "next_anchor_id": self._next_anchor_id,
            "next_action_id": self._next_action_id,
            "root_ids": list(self.root_ids),
            "programs": [asdict(item) for item in self.programs()],
            "anchors": [asdict(item) for item in self.anchors()],
            "actions": [asdict(item) for item in self.actions()],
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
            forest._children.setdefault(anchor.id, [])
            if anchor.parent_id is not None:
                forest._relations.add((anchor.parent_id, anchor.program_id))
                forest._children.setdefault(anchor.parent_id, []).append(anchor.id)
        for item in payload["actions"]:
            converted = dict(item)
            if converted["outcome"] is not None:
                converted["outcome"] = Outcome(converted["outcome"])
            converted["status"] = ActionStatus(converted["status"])
            action = Action(**converted)
            forest._actions[action.id] = action
            if action.intent is not None:
                forest._actions_by_arm.setdefault(
                    (action.anchor_id, action.intent), []
                ).append(action)
        forest.root_ids = [int(item) for item in payload["root_ids"]]
        forest._next_program_id = int(payload["next_program_id"])
        forest._next_anchor_id = int(payload["next_anchor_id"])
        forest._next_action_id = int(payload["next_action_id"])
        return forest


__all__ = ["Forest", "is_better"]
