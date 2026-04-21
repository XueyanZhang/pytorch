"""OpenAI-compatible backend (works with OpenAI, vLLM, Gemini, etc.)."""

import logging
import os
import time

log = logging.getLogger("torch._inductor.fusion")


def call(
    system_prompt: str,
    messages: list[dict],
    model: str,
    max_tokens: int = 65536,
    temperature: float = 0.0,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[str, dict]:
    """Call OpenAI-compatible API. Returns (text, usage_dict)."""
    import openai

    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    elif base_url:
        kwargs["api_key"] = os.environ.get("VLLM_API_KEY", "EMPTY")

    client = openai.OpenAI(**kwargs)  # uses OPENAI_API_KEY by default

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
    )
    elapsed = time.time() - t0

    text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }

    provider = "vllm" if base_url else "openai"
    log.info(
        "LLM [%s] done: %.1fs, in=%d out=%d, model=%s",
        provider, elapsed, usage["input_tokens"], usage["output_tokens"], model,
    )
    return text, usage
