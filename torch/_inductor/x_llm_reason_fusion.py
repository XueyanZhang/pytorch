"""
LLM-based fusion reasoning (Step 2 of the LLM fusion pipeline).

Takes the pre-fusion scheduler nodes, builds a graph text representation,
sends it to an LLM for fusion analysis, and writes a JSON file that
x_llm_fusion.py (Step 3) can consume.

Env vars:
  X_LLM_BACKEND         — LLM backend "provider:model" (default: claude:claude-sonnet-4-6)
                            See x_llm_backend/__init__.py for supported providers.
  X_LLM_REASON_MODEL    — DEPRECATED, use X_LLM_BACKEND instead
  X_LLM_REASON_FORMAT   — graph format: "adj" or "jsonl" (default: adj)
  X_LLM_REASON_DUMP_DIR — dump response & groups for debug (default: off)
  X_LLM_REASON_STRATEGY — prompt strategy: "direct", "pattern", "pairwise" (default: direct)
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
   - ext nodes CANNOT be fused with any other node — no exceptions
   - Only nodes with compatible shapes can be fused (broadcast from reduced shape to full shape is allowed)
   - Fused nodes must form a connected subgraph in the DAG (no gaps)
   - Fusing nodes must not create a dependency cycle
   - In-place mutation ops can only fuse if read/write indices match exactly

3. **Fusion opportunities (ranked by priority — apply higher priority first):**

   **Priority 1 — Vertical fusion (producer-consumer):**
   These eliminate intermediate buffers from global memory, yielding the largest speedup.
   - red + upstream pw → fuse (the pw feeds directly into the red)
   - red + downstream pw → fuse (the pw consumes the red output)
   - pw chain: producer pw feeds consumer pw with no other consumer → fuse

   **Priority 2 — Canonical patterns (always fuse when present):**
   - **LayerNorm / RMSNorm:** a welford `var_mean` reduction pair (two `red.welford` nodes
     on the same input) + the downstream normalization `pw` (rsqrt, mul, sub, etc.)
     These MUST be fused as a group — they are a single logical operation split into nodes.
   - **Residual + LayerNorm:** if a `pw(add ...)` node computes a residual sum and feeds
     directly into a LayerNorm pattern above, absorb it into the same group.

   **Priority 3 — Horizontal fusion (independent nodes):**
   These save memory reads when nodes share inputs, but do NOT eliminate intermediate buffers.
   - pw + pw with same or broadcastable shapes sharing common inputs → fuse
   - Reductions along different dims of the same tensor → fuse (mix-order reduction)

4. **Fusion barriers:**
   - ext nodes break fusion chains — each ext node is a hard boundary
   - Shape incompatibility prevents fusion
   - Dependency conflicts (fusing would create a cycle) prevent fusion

5. **Reading the graph — important:**
   The adjacency list only shows **producer edges** (`<- [deps]`). There are NO consumer
   edges. To find which nodes **consume** node X's output, scan all nodes and check whether
   X appears in their `<- [...]` dependency list. You MUST do this reverse-lookup to
   identify downstream fusion candidates (e.g., finding the pw that consumes a reduction).
"""

OUTPUT_FORMAT = """\
## Output format

Output one fusion group per line as JSONL (one JSON object per line, NO wrapping array):
```
{"nodes": [3, 4, 5, 6], "reason": "LayerNorm: var_mean pair + rsqrt + scale, all S0->S1 compatible"}
{"nodes": [16, 17, 18, 19], "reason": "LayerNorm post-attention"}
```

Rules for output:
- **CRITICAL — no overlap:** Each node id must appear in AT MOST one group.
  If a node appears in multiple groups, ALL groups containing that node are discarded.
  BAD example (node 4 appears in two groups — BOTH groups are thrown away):
  ```
  {"nodes": [4, 5], "reason": "..."}
  {"nodes": [4, 20], "reason": "..."}
  ```
  Instead, pick the single best group for each node.
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
                "## Analysis steps\n\n"
                "Follow these steps before producing output:\n\n"
                "1. **Identify ext barriers:** List all ext nodes. These split the graph into "
                "segments of pw/red nodes between consecutive ext boundaries.\n"
                "2. **Analyze each segment:** Within each segment, find fusion opportunities "
                "following the priority order (vertical first, then canonical patterns, then horizontal). "
                "For each node, reverse-lookup its consumers from the dependency lists.\n"
                "3. **Emit groups:** Output the final JSONL fusion groups. Double-check that no "
                "node id appears in more than one group.\n\n"
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


def _dump_reason(graph_text: str, fmt: str, response_text: str, groups: list[dict]) -> str | None:
    """Dump input graph, LLM response, and parsed groups to disk.

    Returns the allocated dump directory path, or None if dump is disabled.
    """
    dump_dir = os.environ.get("X_LLM_REASON_DUMP_DIR", "")
    if not dump_dir:
        return None

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
            for g in groups:
                f.write(json.dumps(g) + "\n")

        reason_log.info("dumped reason to %s", path)
        return path

    return None


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════


def reason_fusion(nodes: list, scheduler) -> tuple[list[dict], str | None]:
    """
    Main entry point for LLM fusion reasoning.

    Builds a graph representation, sends it to Claude, parses the response,
    and returns the fusion groups in memory.

    Returns (groups, dump_dir):
        groups   — list of fusion group dicts: [{"nodes": [...], "reason": "..."}, ...]
        dump_dir — path to the reason dump directory, or None if dumping is off
    """
    fmt = os.environ.get("X_LLM_REASON_FORMAT", "adj")
    strategy = os.environ.get("X_LLM_REASON_STRATEGY", "direct")

    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Use one of: {', '.join(STRATEGIES)}"
        )

    reason_log.info("format=%s strategy=%s", fmt, strategy)

    # Step 1: Build graph text
    graph_text = _build_graph_text(nodes, scheduler, fmt)
    reason_log.info("graph: %d nodes, %d chars", len(nodes), len(graph_text))

    # Step 2: Build prompt and call LLM
    from torch._inductor import metrics
    from torch._inductor.x_llm_backend import call_llm

    messages = STRATEGIES[strategy](graph_text, fmt)
    t0 = time.perf_counter()
    response_text, usage = call_llm(SYSTEM_PROMPT, messages)
    t1 = time.perf_counter()
    llm_elapsed = t1 - t0
    metrics.llm_latency_s += llm_elapsed
    metrics.llm_input_tokens += usage.get("input_tokens", 0)
    metrics.llm_output_tokens += usage.get("output_tokens", 0)
    reason_log.info("LLM call took %.2fs", llm_elapsed)

    # Step 3: Parse fusion groups
    groups = _parse_fusion_groups(response_text)
    reason_log.info("parsed %d fusion groups", len(groups))

    # Dump reasoning artifacts (graph, response, groups)
    dump_dir = _dump_reason(graph_text, fmt, response_text, groups)

    return groups, dump_dir
