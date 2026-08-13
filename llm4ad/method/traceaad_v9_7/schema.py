"""Frozen search facts for TraceAAD V9.7."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

PROTOCOL_ID: Final[str] = "traceaad-v9.7-route-refine-explore"

# History selection and rendering are byte-identical to V9.6, so the policy
# id keeps the V9.6 value on purpose: same string, same mechanism.
HISTORY_SELECTOR_ID: Final[str] = (
    "v96_recent_formation_fill_capped_improve_regress_direct_v1"
)
GENERATION_POLICY_ID: Final[str] = "v97_intent_refine_explore_optional_idea_full_code_v1"
CANDIDATE_MULTIPLICITY_POLICY_ID: Final[str] = "v96_single_candidate_reselect_v1"
BUDGET_POLICY_ID: Final[str] = (
    "v97_route_then_anchor_quality_optimistic_allocation_v1"
)
INTENT_POLICY_ID: Final[str] = (
    "v97_fixed_mixture_refine07_explore03_seed_iteration_hash_v1"
)
INITIALIZATION_POLICY_ID: Final[str] = "v96_k_independent_roots_one_bootstrap_v1"
OPTIMISM_SCALE_POLICY_ID: Final[str] = "v96_median_valid_bootstrap_abs_delta_v1"
STATE_IDENTITY_POLICY_ID: Final[str] = (
    "v96_parent_state_artifact_relation_no_ancestral_return_v1"
)
CANDIDATE_ACCOUNTING_POLICY_ID: Final[str] = "v96_real_evaluator_call_budget_v1"
STOP_POLICY_ID: Final[str] = "v96_evaluator_budget_exhaustion_v1"
NORMALIZATION_POLICY_ID: Final[str] = "v96_evaluator_input_is_artifact_identity_v1"

REFINE_PROBABILITY: Final[float] = 0.7


class GenerationIntent(StrEnum):
    REFINE = "refine"
    EXPLORE = "explore"


class AttemptKind(StrEnum):
    ROOT_NEW = "root_new"
    ROOT_DUPLICATE = "root_duplicate"
    NEW_ARTIFACT = "new_artifact"
    CACHED_ARTIFACT = "cached_artifact"
    NO_OP = "no_op"
    REPEATED_DUPLICATE = "repeated_duplicate"
    ANCESTRAL_RETURN = "ancestral_return"
    INVALID = "invalid"


class DirectOutcome(StrEnum):
    IMPROVE = "improve"
    PLATEAU = "plateau"
    REGRESS = "regress"
    INVALID = "invalid"


class PendingStage(StrEnum):
    RESPONSE_RECEIVED = "response_received"
    PARSED = "parsed"
    EVALUATED = "evaluated"


@dataclass(frozen=True, slots=True)
class ProgramArtifact:
    artifact_id: int
    evaluator_contract_hash: str
    evaluator_input_hash: str
    evaluator_input_code: str
    fitness: float
    directed_fitness: float
    code_length: int
    program_loc: int
    first_discovery_order: int


@dataclass(slots=True)
class AnchorState:
    state_id: int
    artifact_id: int
    parent_state_id: int | None
    incoming_attempt_id: int | None
    depth: int
    creation_order: int
    generation_count_n: int = 0


@dataclass(frozen=True, slots=True)
class DiffStatistics:
    added_lines: int
    removed_lines: int
    changed_lines: int


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt_id: int
    status: Literal["finalized"]
    anchor_state_id: int | None
    child_state_id: int | None
    artifact_id: int | None
    intent: str | None
    declared_idea: str | None
    raw_code_hash: str | None
    evaluator_input_hash: str | None
    actual_diff: str | None
    diff_statistics: DiffStatistics | None
    parent_fitness: float | None
    child_fitness: float | None
    directed_delta: float | None
    direct_outcome: DirectOutcome | None
    attempt_kind: AttemptKind
    failure_category: str | None
    failure_feedback: str | None
    evaluator_called: bool
    candidate_order: int
    creation_time: str
    stage: str
    iteration: int | None


@dataclass(slots=True)
class PendingAttempt:
    attempt_id: int
    anchor_state_id: int | None
    stage_name: str
    iteration: int | None
    candidate_order: int
    intent: str | None
    response: str
    prompt: str
    prompt_tokens: int
    response_tokens: int
    sample_time: float
    generation_seed: int | None
    processing_stage: PendingStage = PendingStage.RESPONSE_RECEIVED
    declared_idea: str | None = None
    raw_code: str | None = None
    raw_code_hash: str | None = None
    evaluator_input_code: str | None = None
    evaluator_input_hash: str | None = None
    actual_diff: str | None = None
    diff_statistics: DiffStatistics | None = None
    evaluator_called: bool = False
    evaluated_fitness: float | None = None
    failure_category: str | None = None
    failure_feedback: str | None = None
    evaluate_time: float | None = None


__all__ = [
    "BUDGET_POLICY_ID",
    "CANDIDATE_ACCOUNTING_POLICY_ID",
    "CANDIDATE_MULTIPLICITY_POLICY_ID",
    "GENERATION_POLICY_ID",
    "HISTORY_SELECTOR_ID",
    "INITIALIZATION_POLICY_ID",
    "INTENT_POLICY_ID",
    "NORMALIZATION_POLICY_ID",
    "OPTIMISM_SCALE_POLICY_ID",
    "PROTOCOL_ID",
    "REFINE_PROBABILITY",
    "STATE_IDENTITY_POLICY_ID",
    "STOP_POLICY_ID",
    "AnchorState",
    "AttemptKind",
    "AttemptRecord",
    "DiffStatistics",
    "DirectOutcome",
    "GenerationIntent",
    "PendingAttempt",
    "PendingStage",
    "ProgramArtifact",
]
