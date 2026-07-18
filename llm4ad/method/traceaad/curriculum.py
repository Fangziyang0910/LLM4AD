"""Elite curriculum memory for TraceAAD.

This module turns sparse, validated graph events into bounded generation
evidence.  It deliberately does not create nodes, edges, or active
trajectories: those remain the responsibility of the search loop.
"""
from __future__ import annotations

from collections import defaultdict

from .credit import directed_delta
from .derivation_graph import DerivationGraph
from .schema import (
    ChampionEvent,
    CurriculumPacket,
    CurriculumTrace,
    ImprovementEdge,
    TraceStep,
)


class EliteCurriculum:
    """Deep module that owns constructed-trace retrieval and feedback."""

    def __init__(
        self,
        graph: DerivationGraph,
        *,
        maximize: bool = True,
        max_champion_events: int = 4,
        max_positive_traces: int = 2,
    ) -> None:
        if max_champion_events <= 0:
            raise ValueError("max_champion_events must be positive")
        if max_positive_traces <= 0:
            raise ValueError("max_positive_traces must be positive")
        self._graph = graph
        self._maximize = maximize
        self._max_champion_events = max_champion_events
        self._max_positive_traces = max_positive_traces
        self._champion_events: list[ChampionEvent] = []
        self._packet_count = 0
        self._usage: defaultdict[str, int] = defaultdict(int)
        self._reward: defaultdict[str, float] = defaultdict(float)

    def record_best_event(
        self,
        *,
        previous_best_node_id: int | None,
        new_best_node_id: int,
        operator: str,
        iteration: int | None = None,
        sample_order: int | None = None,
    ) -> ChampionEvent:
        """Record one real incumbent update without manufacturing a graph edge."""
        new_node = self._graph.get_node(new_best_node_id)
        source_edge = self._graph.incoming_edge(new_best_node_id)
        previous_fitness = (
            None
            if previous_best_node_id is None
            else self._graph.get_node(previous_best_node_id).fitness
        )
        delta_to_previous = directed_delta(
            previous_fitness,
            new_node.fitness,
            self._maximize,
        )
        event = ChampionEvent(
            previous_best_node_id=previous_best_node_id,
            new_best_node_id=new_best_node_id,
            source_parent_node_id=None if source_edge is None else source_edge.parent_id,
            source_edge_id=None if source_edge is None else source_edge.id,
            operator=operator,
            delta_to_previous_best=delta_to_previous,
            delta_to_parent=None if source_edge is None else source_edge.delta,
            iteration=iteration,
            sample_order=sample_order,
        )
        self._champion_events.append(event)
        return event

    def build(
        self,
        *,
        operator: str,
        base_node_id: int | None,
        selected_trajectory_id: int | None,
        iteration: int,
        stagnation: int,
    ) -> CurriculumPacket:
        """Assemble a bounded packet for one generation call.

        ``base_node_id`` and ``selected_trajectory_id`` are part of the
        interface so retrieval can become base-aware without changing callers.
        The first implementation uses them only to keep the seam explicit.
        """
        del selected_trajectory_id
        self._packet_count += 1
        derived = self._derived_traces()
        champion = self._champion_trace()
        improve = self._rank_for_operator(
            [trace for trace in derived if trace.kind == "improve_chain"],
            operator,
            base_node_id=base_node_id,
        )
        repair = self._rank_for_operator(
            [trace for trace in derived if trace.kind == "prefix_repair"],
            operator,
            base_node_id=base_node_id,
        )
        contrast = self._rank_for_operator(
            [trace for trace in derived if trace.kind == "contrastive"],
            operator,
            base_node_id=base_node_id,
        )
        donor = self._rank_for_operator(
            [trace for trace in derived if trace.kind == "elite_recombine"],
            operator,
            base_node_id=base_node_id,
        )

        positive: list[CurriculumTrace] = []
        if champion is not None and operator != "novelty_jump":
            positive.append(champion)
        if operator == "backtrack_branch" and repair:
            positive.extend(improve[:1])
        elif operator == "mechanism_crossover":
            positive.extend(improve[:1])
        elif operator != "novelty_jump":
            positive.extend(improve[: self._max_positive_traces])
        positive = self._unique(positive)[: self._max_positive_traces]

        packet = CurriculumPacket(
            id=f"curriculum-packet-{self._packet_count}",
            positive_traces=tuple(positive),
            primary_trace_id=None if not positive else positive[0].id,
            repair_trace=repair[0] if repair and (operator == "backtrack_branch" or stagnation > 0) else None,
            contrast_trace=contrast[0] if contrast and operator != "novelty_jump" else None,
            donor_trace=donor[0] if operator == "mechanism_crossover" and donor else None,
            instructions=self._instructions(operator, iteration),
        )
        for trace_id in packet.trace_ids:
            self._usage[trace_id] += 1
        return packet

    def record_outcome(
        self,
        packet: CurriculumPacket | None,
        *,
        outcome: str,
        global_best: bool = False,
        near_record: bool = False,
    ) -> None:
        """Update packet-level teaching utility, never edge-level facts."""
        if packet is None:
            return
        if global_best:
            reward = 1.0
        elif near_record:
            reward = 0.4
        elif outcome == "improve":
            reward = 0.2
        elif outcome == "plateau":
            reward = -0.1
        else:
            reward = -0.4
        for trace_id in packet.trace_ids:
            self._reward[trace_id] += (
                reward if trace_id == packet.primary_trace_id else 0.25 * reward
            )

    def champion_events(self) -> tuple[ChampionEvent, ...]:
        return tuple(self._champion_events)

    def snapshot(self) -> dict[str, object]:
        return {
            "champion_events": len(self._champion_events),
            "packets": self._packet_count,
            "trace_usage": dict(self._usage),
            "trace_reward": dict(self._reward),
        }

    def _champion_trace(self) -> CurriculumTrace | None:
        if not self._champion_events:
            return None
        events = self._champion_events[-self._max_champion_events :]
        steps: list[TraceStep] = []
        for event in events:
            node = self._graph.get_node(event.new_best_node_id)
            edge = (
                None
                if event.source_edge_id is None
                else self._graph.get_edge(event.source_edge_id)
            )
            steps.append(
                TraceStep(
                    source_node_id=event.new_best_node_id,
                    source_edge_id=event.source_edge_id,
                    parent_node_id=event.previous_best_node_id,
                    operator=event.operator,
                    action="" if edge is None else edge.action,
                    fitness_before=(
                        None
                        if event.previous_best_node_id is None
                        else self._graph.get_node(event.previous_best_node_id).fitness
                    ),
                    fitness_after=node.fitness,
                    delta_to_parent=event.delta_to_parent,
                    delta_to_incumbent=event.delta_to_previous_best,
                    outcome="improve",
                    evidence_type="champion",
                    causal_status="jump",
                )
            )
        gain = sum(
            max(0.0, event.delta_to_previous_best or 0.0) for event in events
        )
        trace_id = f"champion-{events[-1].new_best_node_id}-{len(events)}"
        return CurriculumTrace(
            id=trace_id,
            kind="champion",
            steps=tuple(steps),
            terminal_node_id=events[-1].new_best_node_id,
            quality_gain=gain,
            causal_coherence=0.0,
            novelty=1.0,
            confidence=0.9,
        )

    def _derived_traces(self) -> tuple[CurriculumTrace, ...]:
        edges = tuple(
            edge
            for edge in self._graph.edges()
            if edge.action and edge.delta is not None
        )
        incoming = {
            edge.child_id: edge
            for edge in edges
        }
        traces: list[CurriculumTrace] = []
        for edge in edges:
            if edge.outcome == "improve":
                traces.append(self._improve_trace(edge, incoming))
            elif edge.outcome == "regress":
                traces.append(self._prefix_trace(edge))
        improve = [trace for trace in traces if trace.kind == "improve_chain"]
        regress = [trace for trace in traces if trace.kind == "prefix_repair"]
        traces.extend(self._contrast_traces(improve, regress))
        traces.extend(self._recombine_traces(improve))
        return tuple(self._unique(traces))

    def _improve_trace(
        self,
        edge: ImprovementEdge,
        incoming: dict[int, ImprovementEdge],
    ) -> CurriculumTrace:
        chain = [edge]
        previous = incoming.get(edge.parent_id)
        while (
            previous is not None
            and previous.outcome == "improve"
            and len(chain) < 3
        ):
            chain.insert(0, previous)
            previous = incoming.get(previous.parent_id)
        steps = tuple(self._step_from_edge(item, "causal") for item in chain)
        return CurriculumTrace(
            id="improve-" + "-".join(str(item.id) for item in chain),
            kind="improve_chain",
            steps=steps,
            terminal_node_id=edge.child_id,
            quality_gain=sum(item.delta or 0.0 for item in chain),
            causal_coherence=1.0,
            novelty=1.0,
            confidence=0.85,
        )

    def _prefix_trace(self, edge: ImprovementEdge) -> CurriculumTrace:
        parent = self._graph.get_node(edge.parent_id)
        child = self._graph.get_node(edge.child_id)
        step = self._step_from_edge(edge, "prefix")
        return CurriculumTrace(
            id=f"prefix-{edge.id}",
            kind="prefix_repair",
            steps=(step,),
            terminal_node_id=parent.id,
            quality_gain=self._quality(parent.fitness),
            causal_coherence=0.8,
            novelty=1.0,
            confidence=0.75 if child.fitness is not None else 0.5,
        )

    def _contrast_traces(
        self,
        improve: list[CurriculumTrace],
        regress: list[CurriculumTrace],
    ) -> list[CurriculumTrace]:
        by_operator: dict[str, list[CurriculumTrace]] = defaultdict(list)
        for trace in improve:
            by_operator[trace.steps[-1].operator].append(trace)
        output: list[CurriculumTrace] = []
        for failure in regress:
            operator = failure.steps[-1].operator
            candidates = by_operator.get(operator) or improve
            if not candidates:
                continue
            success = max(candidates, key=self._trace_score)
            output.append(
                CurriculumTrace(
                    id=f"contrast-{success.id}-{failure.id}",
                    kind="contrastive",
                    steps=success.steps[-1:] + failure.steps,
                    terminal_node_id=failure.terminal_node_id,
                    quality_gain=max(0.0, success.quality_gain),
                    causal_coherence=0.4,
                    novelty=1.0,
                    confidence=0.45,
                )
            )
        return output

    def _recombine_traces(
        self,
        improve: list[CurriculumTrace],
    ) -> list[CurriculumTrace]:
        if len(improve) < 2:
            return []
        ranked = sorted(improve, key=self._trace_score, reverse=True)
        first = ranked[0]
        second = next(
            (trace for trace in ranked[1:] if trace.steps[-1].operator != first.steps[-1].operator),
            ranked[1],
        )
        return [
            CurriculumTrace(
                id=f"recombine-{first.id}-{second.id}",
                kind="elite_recombine",
                steps=first.steps[-1:] + second.steps[-1:],
                terminal_node_id=first.terminal_node_id,
                quality_gain=max(0.0, first.quality_gain) + max(0.0, second.quality_gain),
                causal_coherence=0.0,
                novelty=1.0,
                confidence=0.4,
            )
        ]

    def _step_from_edge(self, edge: ImprovementEdge, evidence_type: str) -> TraceStep:
        parent = self._graph.get_node(edge.parent_id)
        child = self._graph.get_node(edge.child_id)
        return TraceStep(
            source_node_id=child.id,
            source_edge_id=edge.id,
            parent_node_id=parent.id,
            operator=edge.operator,
            action=edge.action,
            fitness_before=parent.fitness,
            fitness_after=child.fitness,
            delta_to_parent=edge.delta,
            delta_to_incumbent=None,
            outcome=edge.outcome,
            evidence_type=evidence_type,
            causal_status="direct",
        )

    def _rank_for_operator(
        self,
        traces: list[CurriculumTrace],
        operator: str,
        base_node_id: int | None = None,
    ) -> list[CurriculumTrace]:
        preferred = [
            trace
            for trace in traces
            if any(step.operator == operator for step in trace.steps)
        ]
        fallback = [trace for trace in traces if trace not in preferred]
        return sorted(
            preferred,
            key=lambda trace: self._trace_score(trace, base_node_id),
            reverse=True,
        ) + sorted(
            fallback,
            key=lambda trace: self._trace_score(trace, base_node_id),
            reverse=True,
        )

    def _trace_score(
        self,
        trace: CurriculumTrace,
        base_node_id: int | None = None,
    ) -> float:
        usage_penalty = 0.02 * self._usage.get(trace.id, 0)
        reward_bonus = 0.05 * self._reward.get(trace.id, 0.0)
        base_bonus = (
            0.15
            if base_node_id is not None
            and (
                trace.terminal_node_id == base_node_id
                or any(step.source_node_id == base_node_id for step in trace.steps)
            )
            else 0.0
        )
        return (
            trace.quality_gain
            + 0.25 * trace.causal_coherence
            + 0.15 * trace.confidence
            + 0.1 * trace.novelty
            + reward_bonus
            + base_bonus
            - usage_penalty
        )

    @staticmethod
    def _unique(traces: list[CurriculumTrace]) -> list[CurriculumTrace]:
        output: list[CurriculumTrace] = []
        seen: set[str] = set()
        for trace in traces:
            if trace.id in seen:
                continue
            seen.add(trace.id)
            output.append(trace)
        return output

    @staticmethod
    def _instructions(operator: str, iteration: int) -> tuple[str, ...]:
        del iteration
        if operator == "backtrack_branch":
            return (
                "Treat the prefix as a strong state and repair the failed next step.",
                "Do not repeat the recorded regression action.",
            )
        if operator == "mechanism_crossover":
            return (
                "Transfer exactly one donor idea while preserving the recipient program.",
                "The donor trace is evidence, not a complete program to copy.",
            )
        if operator == "novelty_jump":
            return (
                "Use curriculum only to avoid direct repetition; explore a genuinely different direction.",
            )
        return (
            "Use elite traces as evidence of observed improvements, not guaranteed rules.",
            "Propose one concrete next-step modification rather than replaying a whole trace.",
        )

    def _quality(self, fitness: float | None) -> float:
        values = [
            node.fitness
            for node in self._graph.nodes()
            if node.fitness is not None
        ]
        if fitness is None or not values:
            return 0.0
        lo, hi = min(values), max(values)
        if abs(hi - lo) < 1e-12:
            return 0.5
        raw = (fitness - lo) / (hi - lo)
        return max(0.0, min(1.0, raw if self._maximize else 1.0 - raw))


__all__ = ["EliteCurriculum"]
