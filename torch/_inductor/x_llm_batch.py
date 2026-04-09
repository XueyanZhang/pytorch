"""
Batch-mode utilities for 3-phase LLM fusion pipeline.

Phase 1 (dump):  dump_graph() — serialize pre-fusion graph to disk
Phase 3 (apply): load_groups() — load pre-computed fusion groups from disk

Env vars:
  X_LLM_FUSION_DUMP_DIR   — Phase 1: directory to write graph_NNNN.txt
  X_LLM_FUSION_GROUPS_DIR — Phase 3: directory to read graph_NNNN_groups.jsonl
"""

import hashlib
import itertools
import json
import logging
import os

log = logging.getLogger("torch._inductor.fusion")


_dump_counter = itertools.count()
_load_counter = itertools.count()


def reset_counters() -> None:
    """Reset dump/load counters. Call before each model in multi-model runs."""
    global _dump_counter, _load_counter
    _dump_counter = itertools.count()
    _load_counter = itertools.count()


def _graph_hash(graph_text: str) -> str:
    """Short content hash for graph text verification."""
    return hashlib.sha256(graph_text.encode()).hexdigest()[:16]


def dump_graph(nodes, scheduler, dump_dir: str) -> str:
    """Phase 1: dump graph text to disk.

    Writes graph_NNNN.txt and graph_NNNN_meta.json under *dump_dir*.
    Returns the path to the written graph file.
    """
    fmt = os.environ.get("X_LLM_REASON_FORMAT", "adj")

    from torch._inductor.x_llm_reason_fusion import _build_graph_text

    graph_text = _build_graph_text(nodes, scheduler, fmt)

    os.makedirs(dump_dir, exist_ok=True)
    idx = next(_dump_counter)
    prefix = os.path.join(dump_dir, f"graph_{idx:04d}")

    graph_ext = "jsonl" if fmt == "jsonl" else "txt"
    graph_path = f"{prefix}.{graph_ext}"
    with open(graph_path, "w", encoding="utf-8") as f:
        f.write(graph_text)

    meta = {
        "num_nodes": len(nodes),
        "fmt": fmt,
        "strategy": os.environ.get("X_LLM_REASON_STRATEGY", "direct"),
        "graph_hash": _graph_hash(graph_text),
    }
    with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Phase 1 dump: %s (%d nodes)", graph_path, len(nodes))
    return graph_path


def load_groups(groups_dir: str, nodes=None, scheduler=None) -> tuple[list[dict], float]:
    """Phase 3: load pre-computed fusion groups from disk.

    Reads graph_NNNN_groups.jsonl and graph_NNNN_llm_meta.json.
    If *nodes* and *scheduler* are provided, verifies graph hash matches.
    Returns (groups, llm_latency_s).
    """
    idx = next(_load_counter)
    prefix = os.path.join(groups_dir, f"graph_{idx:04d}")

    groups_path = f"{prefix}_groups.jsonl"
    if not os.path.exists(groups_path):
        log.warning("Phase 3: groups file not found: %s — skipping LLM fusion for this subgraph", groups_path)
        return [], 0.0

    with open(groups_path, "r", encoding="utf-8") as f:
        groups = [json.loads(line) for line in f if line.strip()]

    # Verify graph hash if possible
    meta_path = f"{prefix}_meta.json"
    if nodes is not None and scheduler is not None and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            dump_meta = json.load(f)
        expected_hash = dump_meta.get("graph_hash")
        if expected_hash:
            from torch._inductor.x_llm_reason_fusion import _build_graph_text
            fmt = dump_meta.get("fmt", "adj")
            current_hash = _graph_hash(_build_graph_text(nodes, scheduler, fmt))
            if current_hash != expected_hash:
                log.warning(
                    "Phase 3: graph hash mismatch for %s (expected %s, got %s) — "
                    "subgraph may differ between Phase 1 and Phase 3",
                    prefix, expected_hash, current_hash,
                )

    llm_latency = 0.0
    llm_meta_path = f"{prefix}_llm_meta.json"
    if os.path.exists(llm_meta_path):
        with open(llm_meta_path, "r", encoding="utf-8") as f:
            llm_meta = json.load(f)
        llm_latency = llm_meta.get("llm_latency_s", 0.0)

    log.info("Phase 3 load: %s (%d groups, llm_latency=%.2fs)", groups_path, len(groups), llm_latency)
    return groups, llm_latency
