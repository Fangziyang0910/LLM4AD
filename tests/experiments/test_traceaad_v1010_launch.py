from collections import Counter
import json
import pytest

from experiments.traceaad_v10_10 import launch
from experiments.traceaad_v10_10.run import build_parser


def test_fifteen_distinct_queued_runs_use_free_slots_on_resume():
    plan = launch.build_plan('test', 'v1010')
    assert len(plan) == len({r['session'] for r in plan}) == len({r['run_name'] for r in plan}) == 15
    assert set(Counter(r['task'] for r in plan).values()) == {3}
    for task in {r['task'] for r in plan}:
        assert {r['seed'] for r in plan if r['task'] == task} == {0, 1, 2}
    assignments = launch.allocate(plan, {'local': 1, 'server1': 1})
    assert len(assignments) == 2
    for row, backend in assignments:
        row['backend'] = backend
        args = build_parser().parse_args(list(launch.launch_item(row).command())[3:])
        assert args.budget == 1000 and args.n_roots == 8
    row = assignments[0][0]
    assert launch.allocate([row], {row['backend']: 0}) == []
    other = 'server3' if row['backend'] != 'server3' else 'server1'
    assert launch.allocate([row], {other: 1}) == [(row, other)]


def test_refresh_blocks_uncertain_evaluations_and_limits_resume(monkeypatch):
    plan = launch.build_plan('test', 'v1010')[:4]
    for row in plan:
        row['attempts'] = 3
    statuses = {plan[0]['run_name']: 'finished', plan[1]['run_name']: 'blocked'}
    monkeypatch.setattr(launch, 'get_summary_status', lambda p: statuses.get(p.name))
    monkeypatch.setattr(launch, 'item_is_running', lambda i: i.session == plan[2]['session'])
    launch.refresh(plan, 3)
    assert [r['status'] for r in plan] == ['finished', 'blocked', 'running', 'stopped']


def test_thinking_flag_stamps_every_run_command():
    plan = launch.build_plan('test', 'v1010t', thinking=True)
    assert all(row.get('thinking') is True for row in plan)
    for row in plan:
        row['backend'] = 'local'
        argv = list(launch.launch_item(row).command())[3:]
        args = build_parser().parse_args(argv)
        assert args.thinking is True and args.task == row['task']
    plain = launch.build_plan('other', 'v1010')
    assert not any(row.get('thinking') for row in plain)
    assert '--thinking' not in launch.launch_item({**plain[0], 'backend': 'local'}).command()


def test_llm_pipeline_carries_the_thinking_flag(tmp_path):
    from experiments.infra.base import build_llm_client, llm_payload
    payload = llm_payload(base_url='http://127.0.0.1:8001/v1', model='Qwen3.8-27B',
                          no_proxy='127.0.0.1', max_tokens=64, enable_thinking=True)
    assert payload['enable_thinking'] is True
    assert llm_payload(base_url='http://127.0.0.1:8001/v1', model='m', no_proxy='n',
                       max_tokens=64)['enable_thinking'] is False
    client = build_llm_client(base_url='http://127.0.0.1:8001/v1', model='Qwen3.8-27B',
                              no_proxy='127.0.0.1', max_tokens=64, enable_thinking=True)
    try:
        assert client.enable_thinking is True
        extra = client._merged_extra_body(None)
        assert extra['enable_thinking'] is True
        assert extra['chat_template_kwargs']['enable_thinking'] is True
        assert extra['thinking'] == {'type': 'enabled'}
    finally:
        client.close()


def test_approx_token_count_skips_the_server_and_key_resolves(monkeypatch):
    import os
    from experiments.infra.base import build_llm_client, llm_payload
    from llm4ad.tools.env import resolve_llm_api_key
    client = build_llm_client(base_url='https://api.x5m5x.com/v1', model='m',
                              no_proxy='x', max_tokens=64, chars_per_token=3.0)
    try:
        def boom(payload):
            raise AssertionError('server tokenize must not be called')
        monkeypatch.setattr(client, '_request_token_count', boom)
        assert client.count_tokens('x' * 3000) == 1000
        # _build_messages strips the trailing space: 1199 chars -> 399 + overhead 32.
        assert client.count_prompt_tokens('hello world ' * 100) == 399 + 32
    finally:
        client.close()
    payload = llm_payload(base_url='https://api.x5m5x.com/v1', model='m', no_proxy='n',
                          max_tokens=64, chars_per_token=3.0)
    assert payload['chars_per_token'] == 3.0
    assert 'chars_per_token' not in llm_payload(base_url='u', model='m', no_proxy='n',
                                                max_tokens=64)
    assert resolve_llm_api_key(base_url='https://api.x5m5x.com/v1') == \
        os.environ.get('X5M5X_API_KEY', 'EMPTY')


def test_live_launcher_requires_frozen_source():
    with pytest.raises(ValueError, match='freeze the reviewed source'):
        launch.verify_runtime()
