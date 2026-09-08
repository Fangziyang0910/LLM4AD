from __future__ import annotations

import json

import pytest

from llm4ad.base import Evaluation
from llm4ad.method.traceaad_v10_5.traceaad import read_journal
from llm4ad.method.traceaad_v10_7.prompts import TrajectoryBuilder
from llm4ad.method.traceaad_v10_7.traceaad import TraceAADV107


class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(
            template_program='def score(x):\n    pass',
            task_description='Return a numeric score.', safe_evaluate=False,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1)


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def count_tokens(self, text):
        return len(text.split())

    def count_prompt_tokens(self, text):
        return self.count_tokens(text) + 7

    def draw_sample_with_details(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        response = next(self.responses)
        return response if isinstance(response, dict) else {
            'content': response, 'finish_reason': 'stop',
            'usage': {'completion_tokens': 20},
        }


def response(value=1, idea='Implemented idea.'):
    return f'Idea: {idea}\n```python\ndef score(x):\n    return {value}\n```'


def method(path, llm, **kwargs):
    return TraceAADV107(
        evaluation=TinyEvaluation(), llm=llm, run_dir=path,
        **{'budget': 1, 'n_roots': 1, **kwargs},
    )


def test_prompt_requests_only_idea_and_code():
    builder = TrajectoryBuilder(
        FakeLLM(), 'TASK', max_tokens=1000, history_tokens=8, max_events=0,
    )
    text, _, _, _, _, _ = builder.trajectory(None, [], 'Init')
    assert 'Idea:' in text and '```python' in text
    assert 'Implementation Summary' not in text
    assert 'approximately 500 words' not in text
    assert 'First describe' not in text
    assert 'without referring to Algorithm numbers' in text


def test_one_call_stores_idea_and_removes_second_call_fields(tmp_path):
    llm = FakeLLM(response(7, 'Return the constant seven.'))
    runner = method(tmp_path, llm)
    runner.run()

    assert len(llm.calls) == 1
    assert runner.tree.best().fitness == 7
    assert runner.tree.best().idea == 'Return the constant seven.'
    state = json.loads(runner.state_path.read_text())
    assert state['version'] == 1071
    assert state['mechanism']['generation'] == 'idea_code_single_call_self_contained_v1'
    assert state['mechanism']['context_policy'] == 'task_evidence_v1'
    assert state['mechanism']['inherited_unused'] == {'donor_topk': 5, 'traj_gens': 8}
    assert 'donor_topk' not in state['mechanism'] and 'traj_gens' not in state['mechanism']
    assert 'summary_tokens' not in state['mechanism']

    call = read_journal(runner.llm_calls_path)[0]
    event = read_journal(runner.events_path)[0]
    assert 'stage' not in call
    for field in [
        'design_idea', 'summary_status', 'summary_tokens', 'context_summaries',
        'summary_prompt', 'summary_prompt_tokens', 'summary_attempts',
        'summary_completion', 'implementation_idea', 'generation_llm_seconds',
        'summary_llm_seconds',
    ]:
        assert field not in event


def test_next_prompt_uses_first_call_idea_and_v106_history_shape(tmp_path):
    llm = FakeLLM(
        response(1, 'Root idea.'),
        response(2, 'Child idea.'),
    )
    runner = method(tmp_path, llm, budget=2)
    runner.run()

    assert len(llm.calls) == 2
    assert 'Design note: Root idea.' in llm.calls[1][0]
    assert 'Implementation Summary' not in llm.calls[1][0]
    assert [node.idea for node in runner.tree.all_nodes()] == ['Root idea.', 'Child idea.']


def test_max_context_programs_is_bounded(tmp_path):
    with pytest.raises(ValueError, match='max_context_programs'):
        method(tmp_path, FakeLLM(), max_context_programs=4)


def test_closed_idea_and_code_at_length_limit_are_still_evaluated(tmp_path):
    llm = FakeLLM({'content': response(3), 'finish_reason': 'length'})
    runner = method(tmp_path, llm)
    runner.run()

    assert runner.tree.best().fitness == 3
    assert len(llm.calls) == len(read_journal(runner.evaluations_path)) == 1
