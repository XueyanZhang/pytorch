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

    # op_map: maps scheduler node op names (e.g. "op3") to graph node indices
    # (e.g. 2). Op IDs may skip values for graph inputs (parameters/constants),
    # so opN != nodeN in general. Needed for fusion_result.json → node ID conversion.
    op_map = {node.get_name(): i for i, node in enumerate(nodes)}

    meta = {
        "num_nodes": len(nodes),
        "fmt": fmt,
        "strategy": os.environ.get("X_LLM_REASON_STRATEGY", "direct"),
        "graph_hash": _graph_hash(graph_text),
        "op_map": op_map,
    }
    with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Phase 1 dump: %s (%d nodes)", graph_path, len(nodes))
    return graph_path


_EMPTY_LLM_META = {"llm_latency_s": 0.0, "input_tokens": 0, "output_tokens": 0, "strategy": "", "fmt": ""}


def load_groups(groups_dir: str, nodes=None, scheduler=None) -> tuple[list[dict], dict]:
    """Phase 3: load pre-computed fusion groups from disk.

    *groups_dir* is the method-specific subdir, e.g. .../inference/qwen3-8b/.
    Groups and llm_meta live here; graph meta.json lives in the parent dir
    (alongside the shared graph files).

    Returns (groups, llm_meta_info).
    """
    idx = next(_load_counter)
    name = f"graph_{idx:04d}"
    prefix = os.path.join(groups_dir, name)

    groups_path = f"{prefix}_groups.jsonl"
    if not os.path.exists(groups_path):
        log.warning("Phase 3: groups file not found: %s — skipping LLM fusion for this subgraph", groups_path)
        return [], dict(_EMPTY_LLM_META)

    with open(groups_path, "r", encoding="utf-8") as f:
        groups = [json.loads(line) for line in f if line.strip()]

    # Verify graph hash if possible.
    # meta.json is alongside the graph file (parent of method subdir).
    graph_prefix = os.path.join(os.path.dirname(groups_dir), name)
    meta_path = f"{graph_prefix}_meta.json"
    if nodes is not None and scheduler is not None and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            dump_meta = json.load(f)
        expected_hash = dump_meta.get("graph_hash")
        if expected_hash:
            from torch._inductor.x_llm_reason_fusion import _build_graph_text
            fmt = dump_meta.get("fmt", "adj")
            current_hash = _graph_hash(_build_graph_text(nodes, scheduler, fmt))
            if current_hash != expected_hash:
                from torch._inductor import metrics as inductor_metrics
                inductor_metrics.graph_hash_mismatches += 1
                log.warning(
                    "Phase 3: graph hash mismatch for %s (expected %s, got %s) — "
                    "subgraph may differ between Phase 1 and Phase 3",
                    graph_prefix, expected_hash, current_hash,
                )

    llm_meta_info = dict(_EMPTY_LLM_META)
    llm_meta_path = f"{prefix}_llm_meta.json"
    if os.path.exists(llm_meta_path):
        with open(llm_meta_path, "r", encoding="utf-8") as f:
            llm_meta = json.load(f)
        llm_meta_info["llm_latency_s"] = llm_meta.get("llm_latency_s", 0.0)
        llm_meta_info["input_tokens"] = llm_meta.get("input_tokens", 0)
        llm_meta_info["output_tokens"] = llm_meta.get("output_tokens", 0)
        llm_meta_info["strategy"] = llm_meta.get("strategy", "")
        llm_meta_info["fmt"] = llm_meta.get("fmt", "")

    log.info("Phase 3 load: %s (%d groups, llm_latency=%.2fs, tokens=%d/%d)",
             groups_path, len(groups), llm_meta_info["llm_latency_s"],
             llm_meta_info["input_tokens"], llm_meta_info["output_tokens"])
    return groups, llm_meta_info
