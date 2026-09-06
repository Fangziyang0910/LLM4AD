"""Bounded implementation summaries and algorithm-facing design instructions."""

from dataclasses import dataclass
import hashlib
import re

from llm4ad.method.traceaad_v10_5.prompts import PromptBuilder as BaseBuilder, formation_events

HISTORY_GUIDANCE = (
    "The development history describes how the current algorithm was built and how\n"
    "previous versions performed. Use it as context for this algorithm design task."
)
INSTRUCTIONS = {
    'Init': 'Design a competitive algorithm for this task and implement it using the provided\nfunction interface.',
    'Refine': "Build on the current algorithm's main idea to design an improved version for\nthis task. Choose the implementation changes most likely to improve its performance.",
    'Pivot': 'Design an algorithm for this task using a different main idea from the current\nalgorithm. Use the current algorithm as a reference for developing a promising\nnew approach.',
    'Fuse': 'Design an improved algorithm for this task by combining useful ideas from the\ncurrent and reference algorithms. Choose and adapt the parts that work well\ntogether, aiming to outperform both algorithms.',
}
OUTPUT = """Return one complete Python implementation followed by its summary in this format:

```python
<complete target implementation, including all required imports and helpers>
```
Summary: <implementation summary>

Write approximately 500 words in 2–3 paragraphs, scaled to the implementation's
complexity. Explain how the algorithm implemented above works, including its main
idea and the important formulas, parameter values, and steps in the code.
Explain when its key rules apply."""
COMPARISON = 'Describe the main changes relative to the current algorithm.'
TASK_CONTEXTS = {
    'tsp_construct': """Design a constructive heuristic for the Traveling Salesman Problem, minimizing
the total length of a tour visiting every node once and returning to its start.
At each step, the function receives current and destination node IDs, unvisited
candidate IDs ordered by distance from the current node, and the distance matrix.
Return one candidate node ID. The framework appends the last remaining node and
computes the closed-tour length.""",
    'cvrp_aco': """Design an edge-prior heuristic for Capacitated Vehicle Routing with Ant Colony
Optimization (ACO), minimizing total route length while serving each customer
once with routes respecting vehicle capacity and starting and ending at depot 0.
The function receives pairwise distances, coordinates, demands, and total vehicle
capacity. It computes one finite (n, n) prior matrix per instance. ACO uses
pheromone**alpha * prior**beta * visit_mask * capacity_mask as transition weights
and updates routes and remaining capacity during construction. The evaluator
applies maximum(prior + 1e-9, 1e-9) before ACO uses the matrix.""",
    'op_aco': """Design an edge-prior heuristic for the Orienteering Problem with Ant Colony
Optimization (ACO), maximizing collected prize on a tour starting at depot 0
and returning within the total travel budget maxlen. The function receives node
prizes, pairwise distances, and maxlen and computes one finite (n, n) prior matrix
per instance. ACO combines the prior with pheromone to sample feasible moves,
updating visited nodes, traveled distance, and return-budget feasibility during
construction. Relative prior weights affect transition probabilities. The
evaluator applies maximum(prior + 1e-9, 1e-9) before ACO uses the matrix.""",
    'online_bin_packing': """Design a priority function for online one-dimensional bin packing, minimizing
the number of bins used. Each item is placed immediately on arrival. The function
receives the item size and remaining capacities of currently feasible bins and
returns a finite floating-point score vector of the same shape. The framework
places the item in the bin with highest score (argmax). Strictly increasing
transformations preserve this ordering. Inputs are normally integer-valued;
use floating-point arrays for calculations involving fractional adjustments.""",
    'vrptw_construct': """Design a constructive heuristic for Vehicle Routing with Time Windows,
minimizing total travel cost with capacity and time-window feasibility. Each call
receives current and depot node IDs, feasible unvisited customer IDs, remaining
capacity, current time, demands, distances, and time windows. Return a customer
from the supplied feasible set; while at a customer, returning the depot also
closes the route. The framework filters customers for capacity, time-window,
and return-to-depot feasibility and updates time, service, and capacity.""",
}
TEMPLATE_HASH = hashlib.sha256((HISTORY_GUIDANCE + str(INSTRUCTIONS) + OUTPUT + COMPARISON + str(TASK_CONTEXTS)).encode()).hexdigest()


def build_task_contract(evaluation, task_name=None):
    module_parts = type(evaluation).__module__.split('.')
    task_name = task_name or (module_parts[-2] if len(module_parts) > 1 else module_parts[0])
    description = TASK_CONTEXTS.get(task_name, evaluation.task_description)
    template = evaluation.template_program.replace(
        'Design a novel algorithm to select the next node in each step.',
        'Select the next node for the constructed solution.')
    runtime = (f'The formal evaluation runtime limit is {evaluation.timeout_seconds} seconds.'
               if evaluation.timeout_seconds is not None else 'Use an efficient implementation.')
    return (f'# Task Contract\nDesign an algorithm for the task below by implementing the provided Python function.\n\n'
            f'{description}\n\nTarget interface:\n```python\n{template.strip()}\n```\n'
            'Objective: maximize evaluator fitness (higher is better).\n'
            'Use the information supplied through this interface and produce its specified\n'
            f'output within the stated runtime limit.\n{runtime}')


@dataclass(frozen=True)
class Prompt:
    text: str
    tokens: int
    history_ids: tuple[int, ...]
    omissions: tuple[str, ...]
    history_tokens: int
    summaries: dict


class PromptBuilder(BaseBuilder):
    def __init__(self, *args, summary_tokens=1024, lookup=None, **kwargs):
        super().__init__(*args, **kwargs)
        if summary_tokens < 1:
            raise ValueError('summary_tokens must be positive')
        self.summary_tokens = summary_tokens
        self.lookup = lookup or (lambda _: None)
        self._summaries = {}

    def summary(self, node):
        if node.id not in self._summaries:
            raw = node.idea.strip()
            original = self.count(raw) if raw else 0
            view, status = raw, 'present' if raw else 'unavailable'
            if original > self.summary_tokens:
                marker = '[Remaining summary paragraphs omitted.]'
                kept = []
                for paragraph in re.split(r'\n\s*\n', raw):
                    candidate = '\n\n'.join([*kept, paragraph, marker])
                    if self.count(candidate) > self.summary_tokens:
                        break
                    kept.append(paragraph)
                view = '\n\n'.join([*kept, marker]) if kept else ''
                status = 'prefix' if kept else 'oversized'
            self._summaries[node.id] = (view, {'raw_tokens': original,
                'view_tokens': self.count(view) if view else 0, 'status': status})
        return self._summaries[node.id]

    def render_history(self, events):
        lines = ['# Development History', HISTORY_GUIDANCE]
        for index, (child, parent) in enumerate(events, 1):
            donor = self.lookup(child.donor_id) if child.donor_id is not None else None
            summary, _ = self.summary(child)
            text = f'Step {index}\nPrevious version fitness: {parent.fitness}\n'
            if donor is not None:
                text += f'Reference algorithm fitness: {donor.fitness}\n'
            text += f'Resulting version fitness: {child.fitness}\nImplementation Summary: '
            lines.append(text + (summary or '[Summary unavailable; refer to implementation.]'))
        return '\n\n'.join(lines)

    def assemble(self, current, operator, donor, events, omitted=None):
        omitted = omitted or set()
        parts = [self.task_contract]
        for title, node, show_summary in [('Current Algorithm', current, current is not None and current.parent_id is None),
                                           ('Reference Algorithm', donor, True)]:
            if node is None:
                continue
            text = f'# {title}\nFitness: {node.fitness}\n\n```python\n{self._code(node)}\n```'
            if show_summary and node.id not in omitted:
                summary, _ = self.summary(node)
                text += '\nImplementation Summary: ' + (summary or '[Summary unavailable; refer to implementation.]')
            parts.append(text)
        if events:
            parts.append(self.render_history(events))
        parts.append('# Algorithm Design Task\n' + INSTRUCTIONS[operator])
        parts.append('# Output\n' + OUTPUT + ('\n' + COMPARISON if current else ''))
        return '\n\n\n'.join(parts)

    def fits(self, current, operator, donor=None):
        omitted = {n.id for n in [current, donor] if n is not None}
        return self.count(self.assemble(current, operator, donor, [], omitted), chat=True) <= self.max_tokens

    def build(self, current, ancestors, operator, donor=None):
        events = formation_events(current, ancestors, self.max_events)
        reasons, omitted = [], set()
        while events and self.count(self.render_history(events)) > self.history_tokens:
            reasons.append(f'history_budget:{events.pop(0)[0].id}')
        while True:
            text = self.assemble(current, operator, donor, events, omitted)
            tokens = self.count(text, chat=True)
            if tokens <= self.max_tokens:
                break
            if events:
                reasons.append(f'context_budget:{events.pop(0)[0].id}')
                continue
            candidate = next((n for n in [donor, current] if n is not None and
                              n.id not in omitted and (n is donor or n.parent_id is None)), None)
            if candidate is None:
                raise ValueError('minimum complete prompt exceeds the model context budget')
            omitted.add(candidate.id)
            reasons.append(f'context_summary:{candidate.id}')
        summaries = {}
        shown = [edge[0] for edge in events]
        shown += [n for n in [donor, current] if n is not None and n.id not in omitted
                  and (n is donor or n.parent_id is None)]
        for n in shown:
            _, info = self.summary(n)
            summaries[str(n.id)] = info
            if info['status'] in ['prefix', 'oversized']:
                reasons.append(f'summary_{info["status"]}:{n.id}')
        return Prompt(text, tokens, tuple(n.id for n, _ in events), tuple(reasons),
                      self.count(self.render_history(events)) if events else 0, summaries)
