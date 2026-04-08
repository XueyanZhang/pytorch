"""
Dump pre-fusion graph in adjacency-line format (compact, LLM-friendly).

Output:
  graph_adj.txt          — Layer 1: one-line-per-node adjacency overview
  graph_node_details.json — Layer 2: per-node detail for get_node_detail(id) queries

Env vars:
  X_DUMP_GRAPH     — set to "1" to enable (default: off)
  X_DUMP_GRAPH_DIR — output base directory
                     (default: <repo_root>/xfusion/x_dump_graph_adj)
"""

import itertools
import json
import os

_counter = itertools.count()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_DIR = os.path.join(_REPO_ROOT, "xfusion", "x_dump_graph_adj")


def _get_output_dir() -> str:
    base = os.environ.get("X_DUMP_GRAPH_DIR", _DEFAULT_DIR)
    for n in _counter:
        path = os.path.join(base, f"graph_{n:04d}")
        if os.path.exists(os.path.join(path, "graph_adj.txt")):
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


# ── Reduction type ──

def _get_reduction_type(node) -> str:
    for snode in node.get_nodes():
        if hasattr(snode, "node") and snode.node:
            data = getattr(snode.node, "data", None)
            if data and hasattr(data, "get_reduction_type"):
                return str(data.get_reduction_type())
    return "unknown"


# ── Origin ops ──

def _get_origin_ops(node) -> list[str]:
    ops: list[str] = []
    if hasattr(node, "node") and node.node and hasattr(node.node, "get_origins"):
        for o in node.node.get_origins():
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
    dep_ids: set[int] = set()
    for dep in node.read_writes.reads:
        if hasattr(dep, "name") and dep.name in buf_to_idx:
            prod = buf_to_idx[dep.name]
            if prod != self_idx:
                dep_ids.add(prod)
    return sorted(dep_ids)


def _buf_to_idx_map(nodes) -> dict[str, int]:
    m: dict[str, int] = {}
    for i, node in enumerate(nodes):
        for buf in node.get_outputs():
            m[buf.get_name()] = i
    return m


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════

def dump_adj_graph(nodes, scheduler) -> None:
    out_dir = _get_output_dir()
    buf_idx = _buf_to_idx_map(nodes)

    # Layer 1: adjacency-line text
    adj_path = os.path.join(out_dir, "graph_adj.txt")
    with open(adj_path, "w", encoding="utf-8") as f:
        f.write(_build_adj_text(nodes, buf_idx))

    # Layer 2: node detail JSON
    detail_path = os.path.join(out_dir, "graph_node_details.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(_build_node_details(nodes, buf_idx), f, indent=2, default=str)

    print(f"[x_dump_graph_adj] saved: {adj_path}")
    print(f"[x_dump_graph_adj] saved: {detail_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Layer 1: Adjacency-line text
# ═══════════════════════════════════════════════════════════════════════

def _collect_node_info(nodes, buf_idx):
    """Return list of (id, type_tag, ops_str, dtype, shape_tuple, dep_ids)."""
    infos = []
    for i, node in enumerate(nodes):
        deps = _resolve_deps(node, buf_idx, i)
        if node.is_extern():
            tag = f"ext.{_get_extern_op_name(node)}"
            ops = ""
            dtype, shape = _extern_shape_dtype(node)
        elif node.is_template():
            tag = "tmpl"
            ops = " ".join(_get_origin_ops(node))
            dtype, shape = _node_shape_dtype(node)
        elif node.is_reduction():
            rtype = _get_reduction_type(node)
            short = rtype.replace("welford_reduce", "welford")
            tag = f"red.{short}"
            ops = " ".join(_get_origin_ops(node))
            dtype, shape = _node_shape_dtype(node)
        else:
            tag = "pw"
            ops = " ".join(_get_origin_ops(node))
            dtype, shape = _node_shape_dtype(node)
        infos.append((i, tag, ops, dtype, shape, deps))
    return infos


def _build_shape_aliases(infos):
    """Build shape -> alias mapping for shapes appearing >= 2 times."""
    counts: dict[tuple, int] = {}
    for *_, shape, _ in infos:
        if shape:
            counts[shape] = counts.get(shape, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    aliases: dict[tuple, str] = {}
    idx = 0
    for shape, cnt in ranked:
        if cnt >= 2:
            aliases[shape] = f"S{idx}"
            idx += 1
    return aliases, ranked


def _build_adj_text(nodes, buf_idx) -> str:
    infos = _collect_node_info(nodes, buf_idx)
    aliases, ranked = _build_shape_aliases(infos)

    # Header
    device = "unknown"
    for n in nodes:
        d = n.get_device()
        if d:
            device = str(d)
            break

    num = {"pw": 0, "red": 0, "ext": 0, "tmpl": 0}
    for _, tag, *_ in infos:
        if tag == "pw":
            num["pw"] += 1
        elif tag.startswith("red"):
            num["red"] += 1
        elif tag.startswith("ext"):
            num["ext"] += 1
        elif tag == "tmpl":
            num["tmpl"] += 1

    lines = [f"# {len(nodes)} nodes | {device}"]
    type_parts = [f"{v} {k}" for k, v in num.items() if v]
    lines.append(f"# Types: {', '.join(type_parts)}")

    if aliases:
        alias_strs = [f"{aliases[s]}={list(s)}" for s, _ in ranked if s in aliases]
        lines.append(f"# Shapes: {' '.join(alias_strs)}")

    lines.append("")

    # Node lines
    for i, tag, ops, dtype, shape, deps in infos:
        type_col = f"{tag}({ops})" if ops else f"{tag}()"
        if shape and shape in aliases:
            shape_col = f"{dtype} {aliases[shape]}"
        elif shape:
            shape_col = f"{dtype} {list(shape)}"
        else:
            shape_col = ""
        deps_str = ",".join(str(d) for d in deps)
        line = f"{i:>3} | {type_col:<28} | {shape_col:<14} | <- [{deps_str}]"
        lines.append(line)

    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════
#  Layer 2: Node detail JSON
# ═══════════════════════════════════════════════════════════════════════

def _build_node_details(nodes, buf_idx) -> dict:
    details = {}
    for i, node in enumerate(nodes):
        d: dict = {"id": i, "name": node.get_name()}

        if node.is_extern():
            d["type"] = "extern"
            d["op_name"] = _get_extern_op_name(node)
            dtype, shape = _extern_shape_dtype(node)
            outs = [buf.get_name() for buf in node.get_outputs()]
            d["out"] = {"buf": outs[0] if outs else "", "dtype": dtype, "shape": list(shape)}
        elif node.is_template():
            d["type"] = "template"
            d["ops"] = _get_origin_ops(node)
            _fill_out(d, node)
        elif node.is_reduction():
            d["type"] = "reduction"
            d["reduction_type"] = _get_reduction_type(node)
            d["ops"] = _get_origin_ops(node)
            _fill_out(d, node)
            _fill_reduction_size(d, node)
        else:
            d["type"] = "pointwise"
            d["ops"] = _get_origin_ops(node)
            _fill_out(d, node)

        d["in"] = _build_inputs(node, buf_idx, i)

        # Iteration ranges
        if hasattr(node, "group") and node.group:
            _, iteration = node.group
            d["iteration_ranges"] = [_to_int(x) for x in iteration]

        d["estimated_bytes"] = _estimate_total_bytes(node)
        details[str(i)] = d

    return details


def _fill_out(d: dict, node) -> None:
    for buf in node.get_outputs():
        if hasattr(buf.node, "layout") and hasattr(buf.node.layout, "size"):
            d["out"] = {
                "buf": buf.get_name(),
                "dtype": _short_dtype(str(buf.node.layout.dtype)),
                "shape": [_to_int(s) for s in buf.node.layout.size],
            }
            return
    d["out"] = {"buf": "", "dtype": "", "shape": []}


def _fill_reduction_size(d: dict, node) -> None:
    for snode in node.get_nodes():
        if hasattr(snode, "node") and snode.node:
            data = getattr(snode.node, "data", None)
            if data and hasattr(data, "get_reduction_size"):
                try:
                    d["reduction_size"] = [_to_int(x) for x in data.get_reduction_size()]
                except Exception:
                    pass
                return


def _build_inputs(node, buf_idx: dict, self_idx: int) -> list[dict]:
    seen: dict = {}
    for dep in node.read_writes.reads:
        if hasattr(dep, "name") and dep.name not in seen:
            seen[dep.name] = dep

    inputs = []
    for buf_name in sorted(seen):
        dep = seen[buf_name]
        size = [_to_int(s) for s in dep.size] if hasattr(dep, "size") else []
        shared = _dep_bytes(dep)
        if buf_name in buf_idx:
            prod = buf_idx[buf_name]
            if prod == self_idx:
                continue
            inputs.append({"buf": buf_name, "from": prod, "shape": size, "shared_bytes": shared})
        else:
            inputs.append({"buf": buf_name, "from": "graph_input", "shape": size, "shared_bytes": shared})
    return inputs


def _dep_bytes(dep) -> int:
    try:
        if hasattr(dep, "size"):
            numel = 1
            for s in dep.size:
                numel *= int(s)
            return numel * 4  # default 4 bytes per element
    except Exception:
        pass
    return 0


def _estimate_total_bytes(node) -> int:
    total = 0
    for dep in node.read_writes.reads:
        total += _dep_bytes(dep)
    for dep in node.read_writes.writes:
        total += _dep_bytes(dep)
    return total
