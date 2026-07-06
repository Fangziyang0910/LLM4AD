from __future__ import annotations

from types import SimpleNamespace

import pytest

import llm4ad.tools.llm.vllm_openai_api as openai_compatible
from llm4ad.tools.llm import LlamaCppOpenAIAPI


def make_response(content="OK", reasoning_content=None, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content=reasoning_content),
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
            clients.append(self)

        def close(self):
            pass

    monkeypatch.setattr(openai_compatible.openai, "OpenAI", FakeOpenAI)
    return clients


def test_llama_cpp_openai_api_defaults_disable_thinking(monkeypatch):
    clients = install_fake_openai(monkeypatch, make_response())

    llm = LlamaCppOpenAIAPI(temperature=0)
    assert llm.draw_sample("Say OK only.") == "OK"

    client = clients[0]
    assert client.init_kwargs["base_url"] == "http://127.0.0.1:8001/v1"
    assert client.init_kwargs["api_key"] == "EMPTY"

    call = client.chat.completions.calls[0]
    assert call["model"] == "Qwen3.6-27B"
    assert call["max_tokens"] == 16384
    assert call["temperature"] == 0
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_llama_cpp_openai_api_reports_reasoning_only_response(monkeypatch):
    install_fake_openai(
        monkeypatch,
        make_response(content="", reasoning_content="thinking only", finish_reason="length"),
    )
    llm = LlamaCppOpenAIAPI(max_tokens=16)

    with pytest.raises(RuntimeError, match="empty message.content.*enable_thinking.*max_tokens"):
        llm.draw_sample("Say OK only.")
