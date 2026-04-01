"""Anthropic Claude backend."""

import logging
import time

log = logging.getLogger("torch._inductor.fusion")


def call(
    system_prompt: str,
    messages: list[dict],
    model: str,
    max_tokens: int = 65536,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """Call Claude API with streaming. Returns (text, usage_dict)."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY

    t0 = time.time()
    text_parts: list[str] = []
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for chunk in stream.text_stream:
            text_parts.append(chunk)
    elapsed = time.time() - t0

    response = stream.get_final_message()
    text = "".join(text_parts)
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    log.info(
        "LLM [claude] done: %.1fs, in=%d out=%d, model=%s",
        elapsed, usage["input_tokens"], usage["output_tokens"], response.model,
    )
    return text, usage
