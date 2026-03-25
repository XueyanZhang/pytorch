"""
Dump pre-fusion graph as raw IR text (node.debug_str()).

Usage:
  X_DUMP_GRAPH=1 python ...

Env vars:
  X_DUMP_GRAPH         - set to "1" to enable (default: off)
  X_DUMP_GRAPH_DIR     - output base directory
                         (default: /home/pdd/xyz/fusionr1/xfusion/x_dump_graph_raw)
"""

import itertools
import os

_counter = itertools.count()

_DEFAULT_DIR = "/home/pdd/xyz/fusionr1/xfusion/x_dump_graph_raw"


def _get_output_dir() -> str:
    base = os.environ.get("X_DUMP_GRAPH_DIR", _DEFAULT_DIR)
    for n in _counter:
        path = os.path.join(base, f"graph_{n:04d}")
        if os.path.exists(os.path.join(path, "graph_raw.txt")):
            continue
        os.makedirs(path, exist_ok=True)
        return path
    raise RuntimeError("unreachable")


def dump_raw_graph(nodes, scheduler) -> None:
    """Dump raw IR text for all pre-fusion nodes."""
    out_dir = _get_output_dir()

    raw_path = os.path.join(out_dir, "graph_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        for node in nodes:
            f.write(node.debug_str())
            f.write("\n\n\n")

    print(f"[x_dump_graph_raw] saved: {raw_path}")
