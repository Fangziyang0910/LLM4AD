import ast
import inspect
import sys

sys.path.insert(0, 'tests/method')
from test_traceaad_v105 import FakeLLM, TinyEvaluation, response
from llm4ad.method.traceaad_v10_11 import TraceAADV1011
from llm4ad.method.traceaad_v10_11 import traceaad, trajectory


def test_v1011_is_local_and_requires_one_idea(tmp_path):
    sources = inspect.getsource(traceaad) + inspect.getsource(trajectory)
    assert 'traceaad_v10_10' not in sources
    method = TraceAADV1011(evaluation=TinyEvaluation(), llm=FakeLLM(),
                           run_dir=tmp_path, budget=1, n_roots=1)
    assert method.parse_response('```python\ndef score(x):\n    return 1\n```') is None
    parsed = method.parse_response('Idea: constant score\nCode:\n```python\ndef score(x):\n    return 1\n```')
    assert parsed[0] == 'constant score'
    assert ast.parse(parsed[1]).body[0].name == 'score'


def test_v1011_runs_function_through_template(tmp_path):
    method = TraceAADV1011(evaluation=TinyEvaluation(),
                           llm=FakeLLM('Idea: Return the requested constant.\nCode:\n```python\ndef score(x):\n    return 7\n```'), run_dir=tmp_path,
                           budget=1, n_roots=1)
    method.run()
    assert method.tree.best().fitness == 7
    assert method.tree.best().code.startswith('def score')


def test_v1011_rejects_external_program_content_and_very_long_idea(tmp_path):
    method = TraceAADV1011(evaluation=TinyEvaluation(), llm=FakeLLM(),
                           run_dir=tmp_path, budget=1, n_roots=1)
    assert method.parse_response(
        'Idea: valid summary\nCode:\n```python\nimport math\ndef score(x):\n    return 1\n```') is None
    long_idea = 'word ' * 1700
    assert method.parse_response(
        f'Idea: {long_idea}\nCode:\n```python\ndef score(x):\n    return 1\n```') is None
    assert method.parse_response(
        'Idea: valid summary\nCode:\n```python\ndef score(x):\n    return 1\n```\nExtra explanation') is None


def test_v1011_allows_a_concise_idea_below_the_token_limit(tmp_path):
    method = TraceAADV1011(evaluation=TinyEvaluation(), llm=FakeLLM(),
                           run_dir=tmp_path, budget=1, n_roots=1)
    idea = ' '.join(['mechanism'] * 180)
    parsed = method.parse_response(
        f'Idea: {idea}\nCode:\n```python\ndef score(x):\n    return 1\n```')
    assert parsed[0] == idea


def test_v1011_returns_target_function_and_keeps_small_comments(tmp_path):
    method = TraceAADV1011(evaluation=TinyEvaluation(), llm=FakeLLM(),
                           run_dir=tmp_path, budget=1, n_roots=1)
    parsed = method.parse_response(
        'Idea: Add a constant offset.\nCode:\n```python\ndef score(x):\n    # one useful comment\n    return x + 1\n```')
    assert parsed[1].startswith('def score')
    assert not parsed[1].lstrip().startswith('#')


def test_v1011_history_preserves_the_full_bounded_idea(tmp_path):
    method = TraceAADV1011(evaluation=TinyEvaluation(), llm=FakeLLM(),
                           run_dir=tmp_path, budget=1, n_roots=1)
    root = method.tree.add(code='def score(x):\n    return 1', idea='root', fitness=1,
                           evaluation_id=1, parent_id=None, operator='Init')
    child = method.tree.add(code='def score(x):\n    return x', idea=' '.join(['long'] * 200), fitness=2,
                            evaluation_id=2, parent_id=root.id, operator='Refine')
    prompt = method.builder.build(child, 'Refine')
    assert prompt.count('long') == 200
