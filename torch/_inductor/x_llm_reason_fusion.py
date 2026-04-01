"""
LLM-based fusion reasoning (Step 2 of the LLM fusion pipeline).

Takes the pre-fusion scheduler nodes, builds a graph text representation,
sends it to Claude for fusion analysis, and writes a JSON file that
x_llm_fusion.py (Step 3) can consume.

Env vars:
  X_LLM_REASON_FUSION   — set to "1" to enable (default: off)
  X_LLM_REASON_MODEL    — Claude model (default: claude-sonnet-4-6)
  X_LLM_REASON_FORMAT   — graph format: "adj" or "jsonl" (default: adj)
  X_LLM_REASON_DUMP_DIR — dump response & groups for debug (default: off)
  X_LLM_REASON_STRATEGY — prompt strategy: "direct", "pattern", "pairwise" (default: pattern)
"""

import itertools
import json
import logging
import os
import re
import time

reason_log = logging.getLogger("torch._inductor.fusion")


# ═══════════════════════════════════════════════════════════════════════
#  Graph text generation (reuses dump module internals)
# ═══════════════════════════════════════════════════════════════════════


def _build_graph_text(nodes, scheduler, fmt: str) -> str:
    """Build graph text representation from scheduler nodes."""
    if fmt == "adj":
        from torch._inductor.x_dump_graph_adj import _buf_to_idx_map, _build_adj_text
        buf_idx = _buf_to_idx_map(nodes)
        return _build_adj_text(nodes, buf_idx)
    elif fmt == "jsonl":
        from torch._inductor.x_dump_graph_jsonl import _build_jsonl
        lines = _build_jsonl(nodes, scheduler)
        return "\n".join(lines)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'adj' or 'jsonl'.")


# ═══════════════════════════════════════════════════════════════════════
#  Prompt construction
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert compiler engineer specializing in GPU kernel fusion "
    "for PyTorch Inductor / Triton. You analyze computation graphs and decide "
    "which operations should be fused into single GPU kernels for optimal performance. "
    "Be precise and follow the fusion rules strictly."
)

FUSION_RULES = """\
## PyTorch Inductor Fusion Rules

1. **Node types:**
   - `pw` (pointwise): element-wise ops (add, mul, relu, etc.)
   - `red` (reduction): reductions (var_mean, sum, amax, etc.)
   - `ext` (extern): opaque kernels (mm, addmm, sdpa, conv) — these are **fusion barriers**

2. **Fusion legality:**
   - ext nodes CANNOT be fused with any other node (unless they have a Triton template implementation)
   - Only nodes with compatible shapes can be fused (broadcast from reduced shape to full shape is allowed)
   - Fused nodes must form a connected subgraph in the DAG (no gaps)
   - Fusing nodes must not create a dependency cycle
   - In-place mutation ops can only fuse if read/write indices match exactly

3. **Fusion opportunities:**
   - pw + pw with same or broadcastable shapes → fuse (horizontal fusion)
   - pw + pw sharing common input buffers → fuse (horizontal, reduces memory reads)
   - red + upstream pw → fuse (read-write fusion, the pw feeds directly into the red)
   - red + downstream pw → fuse (the pw consumes the red output)
   - Consecutive reductions on the same input can sometimes fuse (e.g., var_mean pairs)
   - Reductions along different dims of the same tensor → fuse (mix-order reduction)

4. **Fusion barriers:**
   - ext nodes (without template impl) break fusion chains
   - Shape incompatibility prevents fusion
   - Dependency conflicts (fusing would create a cycle) prevent fusion
   - Insufficient shared memory savings between nodes
"""

OUTPUT_FORMAT = """\
## Output format

Output one fusion group per line as JSONL (one JSON object per line, NO wrapping array):
```
{"nodes": [3, 4, 5, 6], "reason": "LayerNorm: var_mean pair + rsqrt + scale, all S0->S1 compatible"}
{"nodes": [16, 17, 18, 19], "reason": "LayerNorm post-attention"}
```

Rules for output:
- Each node id should appear in AT MOST one group
- ext nodes must NOT appear in any group
- Only include groups with 2+ nodes
- Keep reasons concise (one line)
- One JSON object per line — do NOT wrap in an array
"""


def _strategy_direct(graph_text: str, fmt: str) -> list[dict[str, str]]:
    """Direct fusion grouping — give full graph, ask for fusion groups."""
    return [
        {
            "role": "user",
            "content": (
                f"Below is a PyTorch Inductor computation graph in **{fmt}** format. "
                "Analyze it and determine which nodes should be fused into the same Triton kernel.\n\n"
                f"{FUSION_RULES}\n"
                f"{OUTPUT_FORMAT}\n"
                f"## Graph\n\n```\n{graph_text}\n```\n"
            ),
        }
    ]


def _strategy_pattern(graph_text: str, fmt: str) -> list[dict[str, str]]:
    """Pattern recognition — identify repeating blocks first, then fuse."""
    return [
        {
            "role": "user",
            "content": (
                "You are analyzing a PyTorch Inductor computation graph for kernel fusion.\n\n"
                f"{FUSION_RULES}\n"
                "## Instructions\n\n"
                "Follow these steps:\n\n"
                "**Step 1: Pattern identification**\n"
                "Identify repeating subgraph patterns (e.g., Transformer blocks, LayerNorm, FFN). "
                "List each unique pattern with its node structure.\n\n"
                "**Step 2: Per-pattern fusion strategy**\n"
                "For each pattern, propose an optimal fusion grouping and explain why.\n\n"
                "**Step 3: Apply to all instances**\n"
                "Apply the per-pattern strategy to every instance in the graph. "
                "Output the complete list of fusion groups.\n\n"
                f"{OUTPUT_FORMAT}\n"
                f"## Graph ({fmt} format)\n\n```\n{graph_text}\n```\n"
            ),
        }
    ]


def _strategy_pairwise(graph_text: str, fmt: str) -> list[dict[str, str]]:
    """Pairwise feasibility — check adjacent pairs, then group."""
    return [
        {
            "role": "user",
            "content": (
                "Below is a computation graph. For each **directly connected** pair of "
                "non-ext nodes (pw or red), determine whether they can be fused.\n\n"
                "For each pair, output:\n"
                "- `YES` or `NO`\n"
                "- A one-line reason\n\n"
                "Then, at the end, group all YES pairs into maximal fusion groups.\n\n"
                f"{FUSION_RULES}\n"
                f"{OUTPUT_FORMAT}\n"
                f"## Graph ({fmt} format)\n\n```\n{graph_text}\n```\n"
            ),
        }
    ]


STRATEGIES = {
    "direct": _strategy_direct,
    "pattern": _strategy_pattern,
    "pairwise": _strategy_pairwise,
}


# ═══════════════════════════════════════════════════════════════════════
#  LLM call
# ═══════════════════════════════════════════════════════════════════════


def _call_llm(messages: list[dict], model: str) -> str:
    """Call Claude API with streaming, return response text."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    t0 = time.time()
    text_parts = []
    with client.messages.stream(
        model=model,
        max_tokens=65536,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text_chunk in stream.text_stream:
            text_parts.append(text_chunk)
    elapsed = time.time() - t0

    response = stream.get_final_message()
    text = "".join(text_parts)
    usage = response.usage

    reason_log.info(
        "LLM call done: %.1fs, in=%d out=%d, model=%s",
        elapsed, usage.input_tokens, usage.output_tokens, response.model,
    )
    return text


# ═══════════════════════════════════════════════════════════════════════
#  Response parsing
# ═══════════════════════════════════════════════════════════════════════


def _parse_fusion_groups(response_text: str) -> list[dict]:
    """Parse JSONL fusion groups from LLM response.

    Takes only the *last* contiguous block of JSONL lines so that
    "thinking" JSONL emitted earlier in the response is ignored.
    """
    # Primary: JSONL — collect contiguous blocks, keep the last one
    blocks: list[list[dict]] = []
    current_block: list[dict] = []
    for line in response_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue
        try:
            obj = json.loads(line)
            if "nodes" in obj:
                current_block.append(obj)
            else:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
        except json.JSONDecodeError:
            if current_block:
                blocks.append(current_block)
                current_block = []
    if current_block:
        blocks.append(current_block)
    if blocks:
        return blocks[-1]

    # Fallback: JSON array in code block
    json_block = re.search(
        r"```(?:json)?\s*\n(\[.*?\])\s*\n```", response_text, re.DOTALL
    )
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: bare JSON array
    array_match = re.search(r"\[\s*\{.*?\}\s*\]", response_text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    return []


# ═══════════════════════════════════════════════════════════════════════
#  Debug dump
# ═══════════════════════════════════════════════════════════════════════

_dump_counter = itertools.count()


def _dump_reason(graph_text: str, fmt: str, response_text: str, groups: list[dict]) -> None:
    """Dump input graph, LLM response, and parsed groups to disk for debugging."""
    dump_dir = os.environ.get("X_LLM_REASON_DUMP_DIR", "")
    if not dump_dir:
        return

    graph_ext = "jsonl" if fmt == "jsonl" else "txt"

    for n in _dump_counter:
        path = os.path.join(dump_dir, f"reason_{n:04d}")
        if os.path.exists(path):
            continue
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, f"graph.{graph_ext}"), "w", encoding="utf-8") as f:
            f.write(graph_text)

        with open(os.path.join(path, "response.txt"), "w", encoding="utf-8") as f:
            f.write(response_text)

        with open(os.path.join(path, "groups.jsonl"), "w", encoding="utf-8") as f:
            json.dump(groups, f, indent=2)

        reason_log.info("dumped reason to %s", path)
        return


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════


def reason_fusion(nodes: list, scheduler) -> list[dict]:
    """
    Main entry point for LLM fusion reasoning.

    Builds a graph representation, sends it to Claude, parses the response,
    and returns the fusion groups in memory.

    Returns list of fusion group dicts: [{"nodes": [...], "reason": "..."}, ...]
    """
    model = os.environ.get("X_LLM_REASON_MODEL", "claude-sonnet-4-6")
    fmt = os.environ.get("X_LLM_REASON_FORMAT", "adj")
    strategy = os.environ.get("X_LLM_REASON_STRATEGY", "direct")

    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Use one of: {', '.join(STRATEGIES)}"
        )

    reason_log.info("model=%s format=%s strategy=%s", model, fmt, strategy)

    # Step 1: Build graph text
    graph_text = _build_graph_text(nodes, scheduler, fmt)
    reason_log.info("graph: %d nodes, %d chars", len(nodes), len(graph_text))

    # Step 2: Build prompt and call LLM
    messages = STRATEGIES[strategy](graph_text, fmt)
    response_text = _call_llm(messages, model)

    # Step 3: Parse fusion groups
    groups = _parse_fusion_groups(response_text)
    reason_log.info("parsed %d fusion groups", len(groups))

    # Optional: dump for debug
    _dump_reason(graph_text, fmt, response_text, groups)

    return groups
