# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------
from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import openai
import requests

from llm4ad.base import LLM


TOKENIZE_RETRY_LIMIT = 3


class TokenizationError(RuntimeError):
    """The serving model could not count tokens for the exact request."""


class OpenAIAPI(LLM):
    """Generic OpenAI-compatible chat client (vLLM, llama.cpp, cloud, etc.)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60,
        max_tokens: int = 16384,
        temperature: float = 1.0,
        top_p: float | None = None,
        stop: str | Sequence[str] | None = None,
        enable_thinking: bool | None = False,
        extra_body: dict[str, Any] | None = None,
        do_auto_trim: bool = True,
        debug_mode: bool = False,
        **client_kwargs: Any,
    ):
        super().__init__(do_auto_trim=do_auto_trim, debug_mode=debug_mode)
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop
        self.enable_thinking = enable_thinking
        self.extra_body = copy.deepcopy(extra_body) if extra_body else {}
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            **client_kwargs,
        )

    def draw_sample(self, prompt: str | Any, *args: Any, **kwargs: Any) -> str:
        messages = self._build_messages(prompt, kwargs.pop("messages", None))
        request = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "stream": False,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", self.temperature),
        }

        top_p = kwargs.pop("top_p", self.top_p)
        stop = kwargs.pop("stop", self.stop)
        if top_p is not None:
            request["top_p"] = top_p
        if stop is not None:
            request["stop"] = stop

        request_extra_body = kwargs.pop("extra_body", None)
        extra_body = self._merged_extra_body(request_extra_body)
        if extra_body:
            request["extra_body"] = extra_body

        request.update(kwargs)
        response = self._client.chat.completions.create(**request)
        return self._content_from_response(response)

    def count_tokens(self, text: str) -> int:
        """Count raw text tokens with the tokenizer serving this model."""
        return self._request_token_count({"model": self.model, "prompt": text})

    def count_prompt_tokens(self, prompt: str | Any) -> int:
        """Count the exact chat-templated tokens used by ``draw_sample``."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(prompt, None),
            "add_generation_prompt": True,
        }
        extra_body = self._merged_extra_body(None)
        for key in ("chat_template", "chat_template_kwargs"):
            if key in extra_body:
                payload[key] = extra_body[key]
        return self._request_token_count(payload)

    def _request_token_count(self, payload: dict[str, Any]) -> int:
        base = self.base_url.rstrip("/")
        tokenize_url = (
            base[: -len("/v1")] + "/tokenize"
            if base.endswith("/v1")
            else base + "/tokenize"
        )
        headers = (
            {}
            if not self.api_key or self.api_key == "EMPTY"
            else {"Authorization": f"Bearer {self.api_key}"}
        )
        last_error: Exception | None = None
        for _ in range(TOKENIZE_RETRY_LIMIT):
            try:
                response = requests.post(
                    tokenize_url,
                    json=payload,
                    headers=headers,
                    timeout=min(float(self.timeout), 30.0),
                )
                response.raise_for_status()
                count = self._token_count_from_payload(response.json())
                if count == 0 and payload:
                    raise ValueError("tokenizer returned no tokens for non-empty text")
                return count
            except Exception as exc:
                last_error = exc
        raise TokenizationError(
            f"model tokenizer failed after {TOKENIZE_RETRY_LIMIT} attempts"
        ) from last_error

    @property
    def token_count_mode(self) -> str:
        return "llm_count_tokens"

    @property
    def prompt_token_count_mode(self) -> str:
        return "llm_chat_template_tokens"

    @staticmethod
    def _token_count_from_payload(payload: Any) -> int:
        if isinstance(payload, dict):
            if isinstance(payload.get("count"), int):
                return int(payload["count"])
            for key in ("tokens", "token_ids"):
                if isinstance(payload.get(key), list):
                    return len(payload[key])
        raise ValueError("tokenizer response does not contain a token count")

    @staticmethod
    def _build_messages(
        prompt: str | Any, messages: Any | None
    ) -> list[dict[str, Any]]:
        if messages is not None:
            if isinstance(messages, dict):
                return [messages]
            return list(messages)
        if not isinstance(prompt, str):
            if isinstance(prompt, dict):
                return [prompt]
            return list(prompt)
        return [{"role": "user", "content": prompt.strip()}]

    def _merged_extra_body(
        self, request_extra_body: dict[str, Any] | None
    ) -> dict[str, Any]:
        extra_body = copy.deepcopy(self.extra_body)
        if self.enable_thinking is not None:
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
            chat_template_kwargs.setdefault("enable_thinking", self.enable_thinking)
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        if request_extra_body:
            self._deep_update(extra_body, request_extra_body)
        return extra_body

    @classmethod
    def _deep_update(cls, target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                cls._deep_update(target[key], value)
            else:
                target[key] = value

    def _content_from_response(self, response: Any) -> str:
        choice = response.choices[0]
        message = choice.message
        content = message.content
        reasoning = getattr(message, "reasoning", None) or getattr(
            message, "reasoning_content", None
        )
        if content is None or (content == "" and reasoning):
            finish_reason = getattr(choice, "finish_reason", None)
            reasoning_note = " with reasoning output" if reasoning else ""
            raise RuntimeError(
                f"{self.__class__.__name__} received empty message.content{reasoning_note} "
                f"from model {self.model!r}; finish_reason={finish_reason!r}. "
                "Check that chat_template_kwargs.enable_thinking is false and max_tokens is large enough."
            )
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    def close(self):
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
