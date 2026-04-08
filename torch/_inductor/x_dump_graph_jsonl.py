"""
Dump pre-fusion graph in JSONL format (compact, LLM-friendly).

Usage:
  X_DUMP_GRAPH=1 python ...

Env vars:
  X_DUMP_GRAPH         - set to "1" to enable (default: off)
  X_DUMP_GRAPH_DIR     - output base directory (default: <repo_root>/xfusion/x_dump_graph_jsonl)
"""

import itertools
import json
import os

_counter = itertools.count()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_DIR = os.path.join(_REPO_ROOT, "xfusion", "x_dump_graph_jsonl")


def _get_output_dir() -> str:
    base = os.environ.get("X_DUMP_GRAPH_DIR", _DEFAULT_DIR)
    for n in _counter:
        path = os.path.join(base, f"graph_{n:04d}")
        if os.path.exists(os.path.join(path, "graph.jsonl")):
            continue
        os.makedirs(path, exist_ok=True)
        return path
    raise RuntimeError("unreachable")  # pragma: no cover


def _to_int(x):
    try:
        return int(x)
    except Exception:
        return str(x)


# ── Extern op name extraction ──

_NOP_IR_TYPES = frozenset({"NopKernel", "ConcatKernel"})
_VIEW_IR_TYPES = frozenset({
    "BaseView", "ExpandView", "PermuteView", "SqueezeView",
    "GenericView", "View", "ReinterpretView",
})


def _get_extern_op_name(node) -> str:
    try:
        inner = getattr(node, "node", None) or getattr(node, "snodes", [None])[0]
        if inner is not None:
            raw = getattr(inner, "python_kernel_name", None)
            if raw is None:
                op_overload = getattr(inner, "op_overload", None)
                if op_overload is not None:
                    raw = str(op_overload)
            if raw:
                parts = raw.split(".")
                name = parts[-2] if raw.endswith(".default") and len(parts) >= 2 else parts[-1]
                return name
            # Fallback: origins
            if hasattr(inner, "get_origins"):
                for origin in inner.get_origins():
                    if origin.op == "call_function":
                        target = str(origin.target)
                        if "aten." in target:
                            return target.split("aten.")[-1].split(".")[0]
                        return target.split(".")[-1]
            # Fallback: IR type
            cls_name = type(inner).__name__
            if cls_name in _NOP_IR_TYPES:
                return "nop"
            if cls_name in _VIEW_IR_TYPES:
                return "view"
            for base in type(inner).__mro__:
                bname = base.__name__
                if bname in _NOP_IR_TYPES:
                    return "nop"
                if bname in _VIEW_IR_TYPES or bname == "BaseView":
                    return "view"
    except Exception:
        pass
    return "unknown"


def _get_origin_ops(node) -> list[str]:
    """Get aten origin ops (stripped of aten. prefix, prims excluded)."""
    ops = []
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


def _get_all_origin_targets(node) -> list[str]:
    """Get all origin op targets (raw strings, including prims)."""
    targets = []
    for snode in node.get_nodes():
        if hasattr(snode, "node") and snode.node and hasattr(snode.node, "get_origins"):
            for o in snode.node.get_origins():
                if o.op == "call_function":
                    targets.append(str(o.target))
    return targets


def _short_dtype(dtype_str: str) -> str:
    m = {
        "torch.float32": "f32", "torch.float16": "f16", "torch.bfloat16": "bf16",
        "torch.int64": "i64", "torch.int32": "i32", "torch.int8": "i8",
        "torch.bool": "b8", "torch.uint8": "u8", "torch.float64": "f64",
    }
    return m.get(dtype_str, dtype_str.replace("torch.", ""))


# ── Semantic type mapping ──
#
# All fusible node types (pw, red, tmpl) use the same generic format:
#   <category>(<sorted_aten_ops>)
# No hardcoded pattern matching — the raw ops are the most generic and
# accurate representation across arbitrary PyTorch models.


def _semantic_pw_type(node) -> str:
    """Determine semantic type string for a pointwise node."""
    ops = _get_origin_ops(node)

    # Include prims.iota if present (semantically meaningful index generation)
    all_targets = _get_all_origin_targets(node)
    if any("prims.iota" in t for t in all_targets):
        ops = sorted(set(ops) | {"iota"})

    if ops:
        return f"pw({','.join(ops)})"
    return "pw()"


def _semantic_ext_type(node) -> str:
    """Determine semantic type string for an extern node."""
    op_name = _get_extern_op_name(node)
    return f"ext.{op_name}"


# ── Shape / dtype extraction ──

def _node_shape_dtype(node) -> tuple[str, tuple]:
    for buf in node.get_outputs():
        if hasattr(buf.node, "layout") and hasattr(buf.node.layout, "size"):
            dtype = _short_dtype(str(buf.node.layout.dtype))
            shape = tuple(_to_int(s) for s in buf.node.layout.size)
            return dtype, shape
    return "", ()


def _extern_shape_dtype(node) -> tuple[str, tuple]:
    try:
        inner = getattr(node, "node", None)
        if inner is not None:
            dtype = ""
            if hasattr(inner, "layout") and hasattr(inner.layout, "dtype"):
                dtype = _short_dtype(str(inner.layout.dtype))
            elif hasattr(inner, "get_dtype"):
                dtype = _short_dtype(str(inner.get_dtype()))
            shape = ()
            if hasattr(inner, "get_size"):
                shape = tuple(_to_int(s) for s in inner.get_size())
            elif hasattr(inner, "layout") and hasattr(inner.layout, "size"):
                shape = tuple(_to_int(s) for s in inner.layout.size)
            return dtype, shape
    except Exception:
        pass
    return "", ()


# ── Dependency resolution ──

def _resolve_deps(node, buf_to_idx: dict, self_idx: int) -> list[int]:
    dep_ids = set()
    for dep in node.read_writes.reads:
        if hasattr(dep, "name") and dep.name in buf_to_idx:
            prod_idx = buf_to_idx[dep.name]
            if prod_idx != self_idx:
                dep_ids.add(prod_idx)
    return sorted(dep_ids)


def _is_residual(dep_ids: list[int], self_id: int, node) -> bool:
    """Residual flag heuristic.

    True when: (1) node is a pointwise add, AND (2) at least one dependency
    spans > 5 id positions.  The add check avoids false positives on non-add
    long-range deps (e.g. U-Net concat, DenseNet cat).
    """
    if self_id - min(dep_ids, default=self_id) <= 5:
        return False
    ops = set(_get_origin_ops(node))
    return "add" in ops


# ── Main entry point ──

def dump_jsonl_graph(nodes, scheduler) -> None:
    out_dir = _get_output_dir()

    jsonl_lines = _build_jsonl(nodes, scheduler)
    jsonl_path = os.path.join(out_dir, "graph.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in jsonl_lines:
            f.write(line + "\n")

    print(f"[x_dump_graph_jsonl] JSONL graph saved to: {jsonl_path}")


def _build_jsonl(nodes, scheduler) -> list[str]:
    # ── buf -> producer id mapping ──
    buf_to_idx: dict[str, int] = {}
    for i, node in enumerate(nodes):
        for buf in node.get_outputs():
            buf_to_idx[buf.get_name()] = i

    # ── Collect node info ──
    infos = []  # (id, type_str, dtype, shape_tuple, dep_ids, is_residual)

    for i, node in enumerate(nodes):
        if node.is_extern():
            type_str = _semantic_ext_type(node)
            dtype, shape = _extern_shape_dtype(node)
        elif node.is_reduction():
            ops = _get_origin_ops(node)
            type_str = f"red({','.join(ops)})" if ops else "red()"
            dtype, shape = _node_shape_dtype(node)
        elif node.is_template():
            ops = _get_origin_ops(node)
            type_str = f"tmpl({','.join(ops)})" if ops else "tmpl()"
            dtype, shape = _node_shape_dtype(node)
        else:
            type_str = _semantic_pw_type(node)
            dtype, shape = _node_shape_dtype(node)

        dep_ids = _resolve_deps(node, buf_to_idx, i)
        residual = _is_residual(dep_ids, i, node)
        infos.append((i, type_str, dtype, shape, dep_ids, residual))

    # ── Shape alias table (shapes appearing >= 3 times) ──
    shape_freq: dict[tuple, int] = {}
    shape_first: dict[tuple, int] = {}
    for idx, (_, _, _, shape, _, _) in enumerate(infos):
        if shape:
            shape_freq[shape] = shape_freq.get(shape, 0) + 1
            if shape not in shape_first:
                shape_first[shape] = idx

    # Sort by frequency desc, then by first occurrence asc
    eligible = [(s, f) for s, f in shape_freq.items() if f >= 3]
    eligible.sort(key=lambda x: (-x[1], shape_first[x[0]]))
    shape_alias: dict[tuple, str] = {}
    for alias_idx, (shape, _) in enumerate(eligible):
        shape_alias[shape] = f"S{alias_idx}"

    # ── Detect device and backend ──
    device = "unknown"
    for node in nodes:
        d = node.get_device()
        if d:
            device = str(d)
            break
    if "cuda" in device:
        backend = "triton"
    elif "cpu" in device:
        backend = "cpp"
    else:
        backend = "triton"

    # ── Max type string width for padding ──
    max_type_len = max((len(t) for _, t, _, _, _, _ in infos), default=0)

    # ── Build header ──
    header = {
        "type": "header",
        "nodes": len(nodes),
        "device": device,
        "backend": backend,
    }
    if shape_alias:
        header["shapes"] = {
            alias: list(shape) for shape, alias in
            sorted(shape_alias.items(), key=lambda x: int(x[1][1:]))
        }

    lines = [json.dumps(header, separators=(",", ":"))]

    # ── Build node lines ──
    for node_id, type_str, dtype, shape, dep_ids, residual in infos:
        padded_type = type_str + " " * (max_type_len - len(type_str))

        obj: dict = {"id": node_id, "t": padded_type}

        if dtype and shape:
            if shape in shape_alias:
                obj["s"] = f"{dtype} {shape_alias[shape]}"
            else:
                obj["s"] = f"{dtype} {list(shape)}"
        elif dtype:
            obj["s"] = dtype

        obj["d"] = dep_ids

        if residual:
            obj["r"] = True

        lines.append(json.dumps(obj, separators=(",", ":")))

    return lines
