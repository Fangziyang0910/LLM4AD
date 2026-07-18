from __future__ import annotations

from types import SimpleNamespace

import pytest

import llm4ad.tools.llm.llm_api_openai as openai_api
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI


def make_response(content="OK", reasoning=None, reasoning_content=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, reasoning=reasoning)
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason=finish_reason,
            )
        ]
    )


def install_fake_openai(monkeypatch, response):
    clients = []

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return response

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())
            self.closed = False
            clients.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(openai_api.openai, "OpenAI", FakeOpenAI)
    return clients


def test_openai_api_disables_thinking_by_default(monkeypatch):
    clients = install_fake_openai(monkeypatch, make_response())

    llm = OpenAIAPI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        model="Qwen3.6-27B",
        temperature=0,
        top_p=0.9,
    )
    assert llm.draw_sample(" Say OK ") == "OK"

    client = clients[0]
    assert client.init_kwargs["base_url"] == "http://127.0.0.1:8001/v1"
    assert client.init_kwargs["api_key"] == "EMPTY"
    assert client.init_kwargs["timeout"] == 60

    call = client.chat.completions.calls[0]
    assert call["model"] == "Qwen3.6-27B"
    assert call["messages"] == [{"role": "user", "content": "Say OK"}]
    assert call["max_tokens"] == 16384
    assert call["temperature"] == 0
    assert call["top_p"] == 0.9
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_openai_api_allows_request_overrides(monkeypatch):
    clients = install_fake_openai(monkeypatch, make_response("custom"))
    llm = OpenAIAPI(
        base_url="http://222.201.145.8:8080/v1",
        api_key="EMPTY",
        model="qwen3.6-27b-awq",
        max_tokens=1024,
        temperature=1.0,
        extra_body={"chat_template_kwargs": {"custom_flag": True}},
    )

    output = llm.draw_sample(
        [{"role": "user", "content": "hello"}],
        max_tokens=7,
        temperature=0.2,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "guided_choice": ["custom"],
        },
    )

    assert output == "custom"
    call = clients[0].chat.completions.calls[0]
    assert call["max_tokens"] == 7
    assert call["temperature"] == 0.2
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert call["extra_body"]["chat_template_kwargs"] == {
        "custom_flag": True,
        "enable_thinking": True,
    }
    assert call["extra_body"]["guided_choice"] == ["custom"]


def test_openai_api_reports_empty_content(monkeypatch):
    install_fake_openai(
        monkeypatch,
        make_response(content=None, reasoning="thinking only", finish_reason="length"),
    )
    llm = OpenAIAPI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        model="Qwen3.6-27B",
        max_tokens=16,
    )

    with pytest.raises(RuntimeError, match="empty message.content.*enable_thinking.*max_tokens"):
        llm.draw_sample("Say OK only.")


def test_openai_api_reports_reasoning_content_only_response(monkeypatch):
    install_fake_openai(
        monkeypatch,
        make_response(content="", reasoning_content="thinking only", finish_reason="length"),
    )
    llm = OpenAIAPI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        model="Qwen3.6-27B",
        max_tokens=16,
    )

    with pytest.raises(RuntimeError, match="empty message.content.*enable_thinking.*max_tokens"):
        llm.draw_sample("Say OK only.")


def test_openai_api_close_delegates_to_client(monkeypatch):
    clients = install_fake_openai(monkeypatch, make_response())
    llm = OpenAIAPI(
        base_url="http://222.201.145.8:8080/v1",
        api_key="EMPTY",
        model="qwen3.6-27b-awq",
    )

    llm.close()

    assert clients[0].closed is True
