from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .vllm_openai_api import VLLMOpenAIAPI


class LlamaCppOpenAIAPI(VLLMOpenAIAPI):
    """OpenAI-compatible llama.cpp endpoint used by local experiments."""

    DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
    DEFAULT_MODEL = "Qwen3.6-27B"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        model: str = DEFAULT_MODEL,
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
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
            do_auto_trim=do_auto_trim,
            debug_mode=debug_mode,
            **client_kwargs,
        )
