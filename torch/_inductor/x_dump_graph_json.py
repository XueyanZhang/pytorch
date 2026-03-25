"""
Dump pre-fusion graph as structured features JSON.

Usage:
  X_DUMP_GRAPH=1 python ...

Env vars:
  X_DUMP_GRAPH         - set to "1" to enable (default: off)
  X_DUMP_GRAPH_DIR     - output base directory
                         (default: /home/pdd/xyz/fusionr1/xfusion/x_dump_graph_json)
"""

import itertools
import json
import os

_counter = itertools.count()

_DEFAULT_DIR = "/home/pdd/xyz/fusionr1/xfusion/x_dump_graph_json"


def _get_output_dir() -> str:
    base = os.environ.get("X_DUMP_GRAPH_DIR", _DEFAULT_DIR)
    for n in _counter:
        path = os.path.join(base, f"graph_{n:04d}")
        if os.path.exists(os.path.join(path, "graph_features.json")):
            continue
        os.makedirs(path, exist_ok=True)
        return path
    raise RuntimeError("unreachable")


def _to_int(x):
    try:
        return int(x)
    except Exception:
        return str(x)


# ── Short dtype ──

_DTYPE_SHORT = {
    "torch.float32": "f32", "torch.float16": "f16", "torch.bfloat16": "bf16",
    "torch.float64": "f64", "torch.int64": "i64", "torch.int32": "i32",
    "torch.int16": "i16", "torch.int8": "i8", "torch.uint8": "u8",
    "torch.bool": "bool",
}


def _short_dtype(dtype_str: str) -> str:
    return _DTYPE_SHORT.get(dtype_str, dtype_str.replace("torch.", ""))


# ── Extern op name extraction ──

_NOP_IR = frozenset({"NopKernel", "ConcatKernel"})
_VIEW_IR = frozenset({
    "BaseView", "ExpandView", "PermuteView", "SqueezeView",
    "GenericView", "View", "ReinterpretView",
})


def _get_extern_op_name(node) -> str:
    try:
        inner = getattr(node, "node", None) or getattr(node, "snodes", [None])[0]
        if inner is None:
            return "unknown"
        raw = getattr(inner, "python_kernel_name", None)
        if raw is None:
            ov = getattr(inner, "op_overload", None)
            if ov is not None:
                raw = str(ov)
        if raw:
            parts = raw.split(".")
            return parts[-2] if raw.endswith(".default") and len(parts) >= 2 else parts[-1]
        if hasattr(inner, "get_origins"):
            for origin in inner.get_origins():
                if origin.op == "call_function":
                    target = str(origin.target)
                    if "aten." in target:
                        return target.split("aten.")[-1].split(".")[0]
                    return target.split(".")[-1]
        cls_name = type(inner).__name__
        if cls_name in _NOP_IR:
            return "nop"
        if cls_name in _VIEW_IR:
            return "view"
        for base in type(inner).__mro__:
            bn = base.__name__
            if bn in _NOP_IR:
                return "nop"
            if bn in _VIEW_IR or bn == "BaseView":
                return "view"
    except Exception:
        pass
    return "unknown"


# ── Reduction helpers ──

def _get_reduction_type(node) -> str:
    for snode in node.get_nodes():
        if hasattr(snode, "node") and snode.node:
            data = getattr(snode.node, "data", None)
            if data and hasattr(data, "get_reduction_type"):
                return str(data.get_reduction_type())
    return "unknown"


def _get_reduction_numel(node) -> str:
    for snode in node.get_nodes():
        if hasattr(snode, "node") and snode.node:
            data = getattr(snode.node, "data", None)
            if data and hasattr(data, "get_reduction_size"):
                try:
                    return str(data.get_reduction_size())
                except Exception:
                    pass
    return "unknown"


# ── Origin ops ──

def _get_origin_ops(node) -> list[str]:
    ops: list[str] = []
    for snode in node.get_nodes():
        if hasattr(snode, "node") and snode.node and hasattr(snode.node, "get_origins"):
            for o in snode.node.get_origins():
                if o.op == "call_function":
                    target = str(o.target)
                    if "prims." in target:
                        continue
                    if "aten." in target:
                        target = target.split("aten.")[-1].split(".")[0]
                    else:
                        target = target.split(".")[-1]
                    ops.append(target)
    return sorted(set(ops))


# ── Shape / dtype helpers ──

def _node_shape_dtype(node) -> tuple[str, list]:
    for buf in node.get_outputs():
        if hasattr(buf.node, "layout") and hasattr(buf.node.layout, "size"):
            dtype = _short_dtype(str(buf.node.layout.dtype))
            shape = [_to_int(s) for s in buf.node.layout.size]
            return dtype, shape
    return "", []


def _extern_shape_dtype(node) -> tuple[str, list]:
    try:
        inner = getattr(node, "node", None)
        if inner is not None:
            dtype = ""
            if hasattr(inner, "layout") and hasattr(inner.layout, "dtype"):
                dtype = _short_dtype(str(inner.layout.dtype))
            elif hasattr(inner, "get_dtype"):
                dtype = _short_dtype(str(inner.get_dtype()))
            shape: list = []
            if hasattr(inner, "get_size"):
                shape = [_to_int(s) for s in inner.get_size()]
            elif hasattr(inner, "layout") and hasattr(inner.layout, "size"):
                shape = [_to_int(s) for s in inner.layout.size]
            return dtype, shape
    except Exception:
        pass
    return "", []


# ── Dependency resolution ──

def _buf_to_idx_map(nodes) -> dict[str, int]:
    m: dict[str, int] = {}
    for i, node in enumerate(nodes):
        for buf in node.get_outputs():
            m[buf.get_name()] = i
    return m


def _resolve_deps(node, buf_to_idx: dict, self_idx: int) -> list[int]:
    dep_ids: set[int] = set()
    for dep in node.read_writes.reads:
        if hasattr(dep, "name") and dep.name in buf_to_idx:
            prod = buf_to_idx[dep.name]
            if prod != self_idx:
                dep_ids.add(prod)
    return sorted(dep_ids)


# ── Node type classification ──

def _classify_node_type(node) -> str:
    if node.is_extern():
        return "extern"
    if node.is_template():
        return "template"
    if node.is_reduction():
        return "reduction"
    return "pointwise"


# ── Byte estimation ──

def _dep_bytes(dep) -> int:
    try:
        if hasattr(dep, "size"):
            numel = 1
            for s in dep.size:
                numel *= int(s)
            return numel * 4
    except Exception:
        pass
    return 0


# ═══════════════════════════════════════════════════════════════════════
#  Feature extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_structured_features(nodes, scheduler) -> dict:
    buf_idx = _buf_to_idx_map(nodes)
    node_features = []

    for i, node in enumerate(nodes):
        feat: dict = {}

        # Basic
        feat["id"] = i
        feat["name"] = node.get_name()
        feat["type"] = _classify_node_type(node)
        feat["device"] = str(node.get_device()) if node.get_device() else "unknown"

        # Compute
        feat["is_reduction"] = node.is_reduction()
        feat["is_extern"] = node.is_extern()
        feat["is_template"] = node.is_template()

        if node.is_extern():
            feat["extern_op"] = _get_extern_op_name(node)
            dtype, shape = _extern_shape_dtype(node)
        else:
            dtype, shape = _node_shape_dtype(node)

        if node.is_reduction():
            feat["reduction_type"] = _get_reduction_type(node)
            feat["reduction_numel"] = _get_reduction_numel(node)

        # Shape / dtype
        feat["dtype"] = dtype
        feat["shape"] = shape

        # Ops
        feat["aten_ops"] = _get_origin_ops(node)

        # Memory
        feat["num_reads"] = len(node.read_writes.reads)
        feat["num_writes"] = len(node.read_writes.writes)
        feat["num_unmet_deps"] = len(node.unmet_dependencies)

        # Outputs
        outputs = []
        for buf in node.get_outputs():
            out: dict = {"name": buf.get_name()}
            if hasattr(buf.node, "layout") and hasattr(buf.node.layout, "size"):
                out["dtype"] = _short_dtype(str(buf.node.layout.dtype))
                out["size"] = [_to_int(s) for s in buf.node.layout.size]
            outputs.append(out)
        feat["outputs"] = outputs

        # Iteration ranges
        if hasattr(node, "group") and node.group:
            _, iteration = node.group
            feat["iteration_ranges"] = [_to_int(x) for x in iteration]

        # Dependencies
        feat["read_buffers"] = sorted({
            dep.name for dep in node.read_writes.reads if hasattr(dep, "name")
        })
        feat["write_buffers"] = sorted({
            dep.name for dep in node.read_writes.writes if hasattr(dep, "name")
        })
        feat["dep_ids"] = _resolve_deps(node, buf_idx, i)

        # Consumers
        consumers = []
        for buf in node.get_outputs():
            for user in buf.users:
                if hasattr(user.node, "get_name"):
                    consumers.append({
                        "node": user.node.get_name(),
                        "can_inplace": user.can_inplace,
                    })
        feat["consumers"] = consumers

        feat["num_ancestors"] = len(node.ancestors) if hasattr(node, "ancestors") else 0

        # Byte estimates
        read_bytes = sum(_dep_bytes(dep) for dep in node.read_writes.reads)
        write_bytes = sum(_dep_bytes(dep) for dep in node.read_writes.writes)
        feat["estimated_read_bytes"] = read_bytes
        feat["estimated_write_bytes"] = write_bytes

        node_features.append(feat)

    # Edges
    edges = _build_edges(nodes, buf_idx, scheduler)

    # Graph stats
    graph_stats = {
        "num_nodes": len(nodes),
        "num_pointwise": sum(1 for f in node_features if f["type"] == "pointwise"),
        "num_reduction": sum(1 for f in node_features if f["type"] == "reduction"),
        "num_extern": sum(1 for f in node_features if f["type"] == "extern"),
        "num_template": sum(1 for f in node_features if f["type"] == "template"),
    }

    return {"graph_stats": graph_stats, "nodes": node_features, "edges": edges}


def _build_edges(nodes, buf_idx: dict, scheduler) -> list:
    merged: dict = {}
    for i, node in enumerate(nodes):
        for dep in node.read_writes.reads:
            if hasattr(dep, "name") and dep.name in buf_idx:
                src_idx = buf_idx[dep.name]
                if src_idx != i:
                    key = (src_idx, i)
                    if key not in merged:
                        merged[key] = {
                            "src": src_idx,
                            "dst": i,
                            "buffers": [],
                            "total_bytes": 0,
                        }
                    merged[key]["buffers"].append(dep.name)
                    try:
                        merged[key]["total_bytes"] += scheduler.dep_size_hint(dep)
                    except Exception:
                        pass
    return list(merged.values())


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════

def dump_json_graph(nodes, scheduler) -> None:
    """Dump structured features JSON for all pre-fusion nodes."""
    out_dir = _get_output_dir()

    features = _extract_structured_features(nodes, scheduler)
    feat_path = os.path.join(out_dir, "graph_features.json")
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, default=str)

    print(f"[x_dump_graph_json] saved: {feat_path}")
