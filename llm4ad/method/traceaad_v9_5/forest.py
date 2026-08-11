"""Program artifacts, history-specific anchor states, and finalized attempts."""

from __future__ import annotations

from .schema import AnchorState, AttemptRecord, ProgramArtifact
from .source import nonempty_loc, text_hash


def is_artifact_better(
    candidate: ProgramArtifact, incumbent: ProgramArtifact | None
) -> bool:
    if incumbent is None:
        return True
    return (
        candidate.directed_fitness,
        -candidate.code_length,
        -candidate.first_discovery_order,
    ) > (
        incumbent.directed_fitness,
        -incumbent.code_length,
        -incumbent.first_discovery_order,
    )


class SearchForest:
    """The complete factual state of one V9.5 search run."""

    def __init__(self, evaluator_contract_hash: str, *, maximize: bool) -> None:
        self.evaluator_contract_hash = evaluator_contract_hash
        self.maximize = maximize
        self._artifacts: dict[int, ProgramArtifact] = {}
        self._artifact_keys: dict[tuple[str, str], int] = {}
        self._states: dict[int, AnchorState] = {}
        self._attempts: dict[int, AttemptRecord] = {}
        self._relations: set[tuple[int, int]] = set()
        self.root_state_ids: list[int] = []
        self._next_artifact_id = 0
        self._next_state_id = 0
        self._next_attempt_id = 0

    def artifacts(self) -> tuple[ProgramArtifact, ...]:
        return tuple(self._artifacts.values())

    def states(self) -> tuple[AnchorState, ...]:
        return tuple(self._states.values())

    def attempts(self) -> tuple[AttemptRecord, ...]:
        return tuple(self._attempts.values())

    def get_artifact(self, artifact_id: int) -> ProgramArtifact:
        return self._artifacts[artifact_id]

    def get_state(self, state_id: int) -> AnchorState:
        return self._states[state_id]

    def get_attempt(self, attempt_id: int) -> AttemptRecord:
        return self._attempts[attempt_id]

    def next_attempt_id(self) -> int:
        attempt_id = self._next_attempt_id
        self._next_attempt_id += 1
        return attempt_id

    def artifact_for_code(self, evaluator_input_code: str) -> ProgramArtifact | None:
        key = (self.evaluator_contract_hash, text_hash(evaluator_input_code))
        artifact_id = self._artifact_keys.get(key)
        return None if artifact_id is None else self.get_artifact(artifact_id)

    def add_artifact(
        self,
        *,
        evaluator_input_code: str,
        fitness: float,
        discovery_order: int,
    ) -> ProgramArtifact:
        evaluator_input_hash = text_hash(evaluator_input_code)
        key = (self.evaluator_contract_hash, evaluator_input_hash)
        if key in self._artifact_keys:
            raise ValueError("artifact already exists")
        artifact = ProgramArtifact(
            artifact_id=self._next_artifact_id,
            evaluator_contract_hash=self.evaluator_contract_hash,
            evaluator_input_hash=evaluator_input_hash,
            evaluator_input_code=evaluator_input_code,
            fitness=fitness,
            directed_fitness=fitness if self.maximize else -fitness,
            code_length=len(evaluator_input_code),
            program_loc=nonempty_loc(evaluator_input_code),
            first_discovery_order=discovery_order,
        )
        self._next_artifact_id += 1
        self._artifacts[artifact.artifact_id] = artifact
        self._artifact_keys[key] = artifact.artifact_id
        return artifact

    def add_root_state(self, *, artifact_id: int, creation_order: int) -> AnchorState:
        state = AnchorState(
            state_id=self._next_state_id,
            artifact_id=artifact_id,
            parent_state_id=None,
            incoming_attempt_id=None,
            depth=0,
            creation_order=creation_order,
        )
        self._next_state_id += 1
        self._states[state.state_id] = state
        self.root_state_ids.append(state.state_id)
        return state

    def add_child_state(
        self,
        *,
        parent_state_id: int,
        artifact_id: int,
        attempt_id: int,
        creation_order: int,
    ) -> AnchorState:
        relation = (parent_state_id, artifact_id)
        if relation in self._relations:
            raise ValueError("parent-state artifact relation already exists")
        lineage_artifacts = {
            self.get_state(state_id).artifact_id
            for state_id in self.ancestor_state_ids(parent_state_id)
        }
        if artifact_id in lineage_artifacts:
            raise ValueError(
                "current or ancestral artifact cannot create a child state"
            )
        parent = self.get_state(parent_state_id)
        state = AnchorState(
            state_id=self._next_state_id,
            artifact_id=artifact_id,
            parent_state_id=parent_state_id,
            incoming_attempt_id=attempt_id,
            depth=parent.depth + 1,
            creation_order=creation_order,
        )
        self._next_state_id += 1
        self._states[state.state_id] = state
        self._relations.add(relation)
        return state

    def add_attempt(self, attempt: AttemptRecord) -> None:
        if attempt.attempt_id in self._attempts:
            raise ValueError("attempt already finalized")
        self._attempts[attempt.attempt_id] = attempt

    def relation_exists(self, parent_state_id: int, artifact_id: int) -> bool:
        return (parent_state_id, artifact_id) in self._relations

    def ancestor_state_ids(self, state_id: int) -> tuple[int, ...]:
        path: list[int] = []
        current: int | None = state_id
        while current is not None:
            path.append(current)
            current = self.get_state(current).parent_state_id
        return tuple(reversed(path))

    def ancestor_artifact_ids(self, state_id: int) -> tuple[int, ...]:
        path = self.ancestor_state_ids(state_id)
        return tuple(self.get_state(item).artifact_id for item in path[:-1])

    def formation_attempt_ids(self, state_id: int) -> tuple[int, ...]:
        ids: list[int] = []
        for path_state_id in self.ancestor_state_ids(state_id)[1:]:
            attempt_id = self.get_state(path_state_id).incoming_attempt_id
            if attempt_id is not None:
                ids.append(attempt_id)
        return tuple(ids)

    def direct_attempt_ids(self, state_id: int) -> tuple[int, ...]:
        return tuple(
            attempt.attempt_id
            for attempt in self._attempts.values()
            if attempt.anchor_state_id == state_id
        )

    def best_artifact(self) -> ProgramArtifact | None:
        best: ProgramArtifact | None = None
        for artifact in self._artifacts.values():
            if is_artifact_better(artifact, best):
                best = artifact
        return best


__all__ = ["SearchForest", "is_artifact_better"]
