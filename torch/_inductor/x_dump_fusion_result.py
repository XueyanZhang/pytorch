"""
Dump post-fusion results: which nodes were fused together.

Walks self.nodes after fusion, extracts snodes from each FusedSchedulerNode.

Output:
  fusion_result.json

Env vars:
  X_DUMP_RESULT    — set to "1" to enable (default: off)
  X_DUMP_FUSION_DIR — output base directory (default: <repo_root>/xfusion/x_dump_fusion_result)
"""

import itertools
import json
import os

_counter = itertools.count()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_DIR = os.path.join(_REPO_ROOT, "xfusion", "x_dump_fusion_result")


def _get_output_dir() -> str:
    base = os.environ.get("X_DUMP_FUSION_DIR", _DEFAULT_DIR)
    for n in _counter:
        path = os.path.join(base, f"graph_{n:04d}")
        if os.path.exists(os.path.join(path, "fusion_result.json")):
            continue
        os.makedirs(path, exist_ok=True)
        return path
    raise RuntimeError("unreachable")


def _node_id(name: str) -> int:
    """Extract integer id from buffer name like 'buf123'."""
    try:
        return int(name.replace("buf", ""))
    except Exception:
        return -1


def _collect_fusion_groups(nodes) -> list[dict]:
    """Walk final scheduler nodes. For each FusedSchedulerNode, collect its snodes."""
    from torch._inductor.scheduler import FusedSchedulerNode

    groups = []
    for node in nodes:
        if isinstance(node, FusedSchedulerNode):
            snode_names = [sn.get_name() for sn in node.snodes]
            groups.append({
                "fused_name": node.get_name(),
                "snodes": snode_names,
                "size": len(snode_names),
            })
    groups.sort(key=lambda g: _node_id(g["snodes"][0]) if g["snodes"] else -1)
    return groups


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════

def dump_fusion_result(nodes, scheduler, out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = _get_output_dir()

    groups = _collect_fusion_groups(nodes)

    result = {
        "num_fused_groups": len(groups),
        "total_fused_nodes": sum(g["size"] for g in groups),
        "groups": groups,
    }

    out_path = os.path.join(out_dir, "fusion_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[x_dump_fusion_result] {len(groups)} fused groups -> {out_path}")
