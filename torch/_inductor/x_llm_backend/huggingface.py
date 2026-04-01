"""HuggingFace transformers local backend.

Loads a model via transformers and runs inference in-process.
Supports both HF Hub model IDs and local paths.

Examples:
  X_LLM_BACKEND=hf:meta-llama/Llama-3-70B-Instruct
  X_LLM_BACKEND=hf:/path/to/local/model
"""

import logging
import time

log = logging.getLogger("torch._inductor.fusion")

# Cache loaded model + tokenizer to avoid reloading across multiple calls.
_cache: dict = {}


def _load_model(model_name: str):
    """Load and cache model + tokenizer."""
    if model_name in _cache:
        return _cache[model_name]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading HF model: %s (this may take a while)...", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    _cache[model_name] = (model, tokenizer)
    log.info("HF model loaded: %s", model_name)
    return model, tokenizer


def call(
    system_prompt: str,
    messages: list[dict],
    model_name: str,
    max_tokens: int = 65536,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """Run local HF inference. Returns (text, usage_dict)."""
    import torch

    model, tokenizer = _load_model(model_name)

    # Build chat messages: prepend system prompt
    chat = [{"role": "system", "content": system_prompt}] + messages

    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)

    input_len = input_ids.shape[1]

    generate_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
    )
    if temperature > 0:
        generate_kwargs["temperature"] = temperature

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(input_ids, **generate_kwargs)
    elapsed = time.time() - t0

    # Decode only the new tokens
    new_ids = output_ids[0, input_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    output_len = len(new_ids)
    usage = {
        "input_tokens": input_len,
        "output_tokens": output_len,
    }

    log.info(
        "LLM [hf] done: %.1fs, in=%d out=%d, model=%s",
        elapsed, input_len, output_len, model_name,
    )
    return text, usage
