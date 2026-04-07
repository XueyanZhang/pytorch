"""
LLM backend abstraction for fusion reasoning.

Selects backend via X_LLM_BACKEND env var (format: "provider:model").

Supported providers:
  claude  — Anthropic Claude API (requires anthropic SDK, ANTHROPIC_API_KEY)
  openai  — OpenAI API (requires openai SDK, OPENAI_API_KEY)
  gemini  — Google Gemini via OpenAI-compatible API (requires openai SDK, GEMINI_API_KEY)
  vllm    — vLLM self-hosted (requires openai SDK, format: vllm:base_url::model)
  hf      — HuggingFace transformers local (requires transformers, torch)

Examples:
  X_LLM_BACKEND=claude:claude-sonnet-4-6
  X_LLM_BACKEND=openai:gpt-4o
  X_LLM_BACKEND=gemini:gemini-2.5-flash
  X_LLM_BACKEND=vllm:http://localhost:8000/v1::meta-llama/Llama-3-70B
  X_LLM_BACKEND=hf:meta-llama/Llama-3-70B-Instruct
  X_LLM_BACKEND=hf:/path/to/local/model

Deprecated: X_LLM_REASON_MODEL (falls back to claude provider if X_LLM_BACKEND unset).
"""

import logging
import os
import warnings

log = logging.getLogger("torch._inductor.fusion")


def _resolve_backend() -> tuple[str, str]:
    """Parse env vars and return (provider, model_string)."""
    backend = os.environ.get("X_LLM_BACKEND", "")
    if backend:
        provider, _, model = backend.partition(":")
        if not model:
            raise ValueError(
                f"X_LLM_BACKEND must be 'provider:model', got {backend!r}"
            )
        return provider, model

    # Deprecated fallback
    legacy_model = os.environ.get("X_LLM_REASON_MODEL", "")
    if legacy_model:
        warnings.warn(
            "X_LLM_REASON_MODEL is deprecated, use X_LLM_BACKEND=claude:<model> instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return "claude", legacy_model

    return "claude", "claude-sonnet-4-6"


def call_llm(
    system_prompt: str,
    messages: list[dict],
    max_tokens: int = 65536,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """
    Call the configured LLM backend.

    Returns (response_text, usage_dict) where usage_dict has
    keys "input_tokens" and "output_tokens".
    """
    provider, model_str = _resolve_backend()
    log.info("LLM backend: provider=%s model=%s", provider, model_str)

    if provider == "claude":
        from torch._inductor.x_llm_backend.claude import call
        return call(system_prompt, messages, model_str, max_tokens, temperature)

    elif provider == "gemini":
        from torch._inductor.x_llm_backend.openai_compat import call

        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        api_key = os.environ.get("GEMINI_API_KEY", "")
        return call(
            system_prompt, messages, model_str, max_tokens, temperature,
            base_url=base_url, api_key=api_key,
        )

    elif provider in ("openai", "vllm"):
        from torch._inductor.x_llm_backend.openai_compat import call

        base_url = None
        model = model_str

        if provider == "vllm":
            if "::" in model_str:
                base_url, _, model = model_str.partition("::")
            else:
                base_url, _, model = model_str.rpartition("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"

        return call(
            system_prompt, messages, model, max_tokens, temperature,
            base_url=base_url,
        )

    elif provider == "hf":
        from torch._inductor.x_llm_backend.huggingface import call
        return call(system_prompt, messages, model_str, max_tokens, temperature)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Supported: claude, openai, vllm, hf."
        )
