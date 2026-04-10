"""
Apply LLM-guided fusion decisions to the scheduler.

Reads a JSON file containing fusion groups (produced by LLM evaluation)
and applies those fusions to the scheduler node list, replacing the
default heuristic-driven fusion pass.

Env vars:
  X_LLM_FUSION          — set to "1" to enable (default: off)
  X_LLM_FUSION_LEGALITY_ONLY — "1" (default) skip heuristics; "0" use full can_fuse
"""

import logging
import os

from torch.utils._ordered_set import OrderedSet

fusion_log = logging.getLogger("torch._inductor.fusion")

# Fusion result constants (returned by _fuse_group)
FUSE_OK = "ok"
FUSE_REJECTED_LEGALITY = "legality"
FUSE_REJECTED_CYCLE = "cycle"


# ═══════════════════════════════════════════════════════════════════════
#  JSON loading and validation
# ═══════════════════════════════════════════════════════════════════════


def _validate_groups(groups: list[dict], num_nodes: int) -> list[dict]:
    """
    Validate fusion groups. Returns only valid groups.
    - Skips groups with <2 nodes
    - Skips groups with out-of-bounds indices
    - Skips ALL groups that share a node index with another group
    """
    # First pass: check bounds and size
    valid = []
    for i, g in enumerate(groups):
        nodes = g.get("nodes", [])
        if len(nodes) < 2:
            continue
        oob = [n for n in nodes if n < 0 or n >= num_nodes]
        if oob:
            fusion_log.warning(
                "[x_llm_fusion] Group %d: out-of-bounds indices %s (num_nodes=%d), skipping",
                i, oob, num_nodes,
            )
            continue
        valid.append(g)

    # Second pass: detect duplicate indices across groups
    index_to_groups: dict[int, list[int]] = {}
    for gi, g in enumerate(valid):
        for idx in g["nodes"]:
            index_to_groups.setdefault(idx, []).append(gi)

    conflicting_groups = set()
    for idx, gis in index_to_groups.items():
        if len(gis) > 1:
            fusion_log.warning(
                "[x_llm_fusion] Node %d appears in multiple groups %s, skipping all",
                idx, gis,
            )
            conflicting_groups.update(gis)

    return [g for gi, g in enumerate(valid) if gi not in conflicting_groups]


# ═══════════════════════════════════════════════════════════════════════
#  Topological ordering within a fusion group
# ═══════════════════════════════════════════════════════════════════════


def _topological_order_within_group(
    group_indices: list[int], nodes: list
) -> list[int]:
    """
    Sort group indices so that dependencies come first.
    For independent nodes (horizontal fusion), preserve original index order.
    """
    index_set = set(group_indices)
    # Map node name -> index for nodes in this group
    name_to_idx = {}
    for idx in group_indices:
        for name in nodes[idx].get_operation_names():
            name_to_idx[name] = idx

    # Build dependency edges within the group
    # edge: (a, b) means a must come before b
    adj: dict[int, list[int]] = {idx: [] for idx in group_indices}
    in_degree: dict[int, int] = {idx: 0 for idx in group_indices}

    for idx in group_indices:
        node = nodes[idx]
        for ancestor_name in node.ancestors:
            if ancestor_name in name_to_idx:
                dep_idx = name_to_idx[ancestor_name]
                if dep_idx != idx and dep_idx in index_set:
                    adj[dep_idx].append(idx)
                    in_degree[idx] += 1

    # Kahn's algorithm, breaking ties by original index order
    queue = sorted([idx for idx in group_indices if in_degree[idx] == 0])
    result = []
    while queue:
        idx = queue.pop(0)
        result.append(idx)
        for neighbor in sorted(adj[idx]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                queue.sort()

    if len(result) != len(group_indices):
        fusion_log.warning(
            "[x_llm_fusion] Cycle detected within group %s, using original order",
            group_indices,
        )
        return list(group_indices)

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Legality checking
# ═══════════════════════════════════════════════════════════════════════


def _can_fuse_legality_only(scheduler, node1, node2) -> bool:
    """
    Check only correctness-required fusion constraints, skipping heuristics.
    Extracted from Scheduler.can_fuse (lines 4958-5083 of scheduler.py).
    """
    from torch._inductor.scheduler import (
        ExternKernelSchedulerNode,
        FusedMixOrderReductions,
        GroupedSchedulerNode,
        NopKernelSchedulerNode,
        WhyNoFuse,
    )
    from . import config, ir
    from .virtualized import V

    if node1 is node2:
        return False

    if isinstance(node1, FusedMixOrderReductions):
        return node1.can_fuse_with(node2)
    if isinstance(node2, FusedMixOrderReductions):
        return False

    why = WhyNoFuse(node1, node2)

    # Multi-output template fast path
    if node1.is_template() and scheduler.get_backend(
        node1.get_device()
    ).can_fuse_multi_outputs_template(node1, node2):
        return True

    # Type checks
    if isinstance(node1, GroupedSchedulerNode) or isinstance(
        node2, GroupedSchedulerNode
    ):
        why("grouped node must not be fused with other nodes")
        return False
    if (
        isinstance(node1, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
        and not node1.is_template()
    ):
        why("node1 is extern or nop")
        return False
    if (
        isinstance(node2, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
        and not node2.is_template()
    ):
        why("node2 is extern or nop")
        return False

    # Dependency ordering
    if node2.get_operation_names() & node1.ancestors:
        why("node1 must go before node2")
        return False

    # Template prologue checks
    if node2.is_template():
        if not config.prologue_fusion:
            why("prologue fusion turned off")
            return False
        if node1.is_reduction() or node1.is_template():
            why("prologue fusion only supported for pointwise nodes")
            return False
        template = node2.get_template_node_or_throw()
        if not isinstance(template, ir.TritonTemplateBuffer):
            why("prologue fusion only supported for TritonTemplates")
            return False
        allowed_prologue_inps = template.get_allowed_prologue_inps()
        unsupported_prologue_args = (
            OrderedSet(inp.get_name() for inp in template.inputs)
            - allowed_prologue_inps
        )
        if node1.get_buffer_names() & unsupported_prologue_args:
            why("prologue fusion not implemented for kernel for these inputs")
            return False
        if node1.has_aliasing_or_mutation() or node1.has_aliasing_or_mutation():
            why("template prologue can only fuse functional pointwise nodes")
            return False
        prologue_nodes = node1.get_nodes()
        for node in prologue_nodes[:-1]:
            node_outs = node.get_outputs()
            for out in node_outs:
                if not all(user.node in prologue_nodes for user in out.users):
                    why("template prologue can only fuse nodes with a single use")
                    return False
        template_snodes = (
            [node2]
            if not hasattr(node2, "snodes")
            else [n for n in node2.snodes if n.is_template()]
        )
        assert len(template_snodes) == 1
        template_snode = template_snodes[0]
        if not (
            len(prologue_nodes[-1].outputs) == 1
            and len(prologue_nodes[-1].outputs[0].users) == 1
            and prologue_nodes[-1].outputs[0].users[0].node is template_snode
        ):
            why("template prologue can only fuse nodes with a single use into template")
            return False

    # Template epilogue checks
    if node1.is_template() and (
        node2.has_aliasing_or_mutation()
        or node2.is_reduction()
        or not config.epilogue_fusion
    ):
        why("template epilogue not satisfied")
        return False

    # Buffer exclusion
    if (node1.get_buffer_names() & V.graph.no_fuse_buffer_names) or (
        node2.get_buffer_names() & V.graph.no_fuse_buffer_names
    ):
        why("fusion for buffer explicit disabled")
        return False

    # Device check
    device = node1.get_device()
    device2 = node2.get_device()
    if device != device2:
        why("device mismatch (%s vs %s)", device, device2)
        return False

    # Vertical vs horizontal legality (skip heuristic scoring)
    if node1.get_operation_names() & node2.ancestors:
        # Vertical: node2 depends on node1
        return (
            scheduler.can_fuse_vertical(node1, node2)
            and scheduler.get_backend(device).can_fuse_vertical(node1, node2)
        )
    else:
        # Horizontal: independent or common reads
        return scheduler.get_backend(device).can_fuse_horizontal(node1, node2)


# ═══════════════════════════════════════════════════════════════════════
#  Fusion operations
# ═══════════════════════════════════════════════════════════════════════


def _fuse_two_nodes(scheduler, node1, node2, fused_nodes: OrderedSet):
    """
    Fuse two nodes. Mirrors the closure in fuse_nodes_once (scheduler.py:4127-4141).
    Returns the new fused node.
    """
    fusion_log.debug(
        "[x_llm_fusion] fusing %s with %s", node1.get_name(), node2.get_name()
    )
    device = node1.get_device()
    assert node2.get_device() == device
    node3 = scheduler.get_backend(device).fuse(node1, node2)
    fused_nodes.discard(node1)
    fused_nodes.discard(node2)
    fused_nodes.add(node3)
    scheduler.name_to_fused_node.update(
        {n.get_name(): node3 for n in node3.get_nodes()}
    )
    return node3


def _fuse_group(
    scheduler,
    group: dict,
    nodes: list,
    fused_nodes: OrderedSet,
    legality_only: bool,
) -> bool:
    """
    Attempt to fuse all nodes in a single fusion group.
    Returns FUSE_OK if fully fused, or FUSE_REJECTED_LEGALITY / FUSE_REJECTED_CYCLE.
    """
    group_indices = group["nodes"]
    reason = group.get("reason", "")
    ordered = _topological_order_within_group(group_indices, nodes)

    fusion_log.debug(
        "[x_llm_fusion] Processing group %s (reason: %s)", group_indices, reason
    )

    # Start with the first node, resolve to current fused version
    first_name = nodes[ordered[0]].get_name()
    accumulator = scheduler.name_to_fused_node.get(first_name, nodes[ordered[0]])

    result = FUSE_OK
    for idx in ordered[1:]:
        next_name = nodes[idx].get_name()
        next_node = scheduler.name_to_fused_node.get(next_name, nodes[idx])

        # Already fused together (shouldn't normally happen)
        if accumulator is next_node:
            continue

        # Legality check
        if legality_only:
            can_fuse = _can_fuse_legality_only(scheduler, accumulator, next_node)
        else:
            can_fuse = scheduler.can_fuse(accumulator, next_node)

        if not can_fuse:
            fusion_log.warning(
                "[x_llm_fusion] Legality check failed: cannot fuse %s with %s in group %s",
                accumulator.get_name(), next_node.get_name(), group_indices,
            )
            result = FUSE_REJECTED_LEGALITY
            break

        # Cycle check
        if scheduler.will_fusion_create_cycle(accumulator, next_node):
            fusion_log.warning(
                "[x_llm_fusion] Cycle detected: cannot fuse %s with %s in group %s",
                accumulator.get_name(), next_node.get_name(), group_indices,
            )
            result = FUSE_REJECTED_CYCLE
            break

        accumulator = _fuse_two_nodes(scheduler, accumulator, next_node, fused_nodes)

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════


def apply_llm_fusion(scheduler, nodes: list, groups: list[dict]) -> list:
    """
    Apply LLM-guided fusion decisions to the scheduler node list.

    Args:
        groups: fusion groups to apply.
    """
    from torch._inductor import metrics

    legality_only = os.environ.get("X_LLM_FUSION_LEGALITY_ONLY", "1") != "0"

    num_suggested = len(groups)
    groups = _validate_groups(groups, len(nodes))
    metrics.llm_groups_suggested += num_suggested
    metrics.llm_groups_rejected_validation += num_suggested - len(groups)

    if not groups:
        fusion_log.debug("===== llm fusion: no valid groups to apply =====")
        fusion_log.info("No valid fusion groups to apply")
        return nodes

    mode = "legality-only" if legality_only else "full can_fuse"
    fusion_log.debug(
        "===== llm fusion start: %d nodes, %d groups, mode=%s =====",
        len(nodes), len(groups), mode,
    )

    # Initialize tracking structures (mirrors fuse_nodes_once)
    fused_nodes = OrderedSet(nodes)
    scheduler.name_to_fused_node = {
        n.get_name(): n for n in nodes for n in n.get_nodes()
    }

    # Apply each group
    applied = 0
    skipped = 0
    total = len(groups)
    for gi, group in enumerate(groups):
        fusion_log.debug(
            "--- llm fusion group (%d/%d): nodes=%s reason=%s ---",
            gi + 1, total, group["nodes"], group.get("reason", ""),
        )
        old_count = len(fused_nodes)
        fuse_result = _fuse_group(scheduler, group, nodes, fused_nodes, legality_only)
        if fuse_result == FUSE_OK:
            applied += 1
            metrics.llm_groups_applied += 1
            fusion_log.debug(
                "--- llm fusion group (%d/%d): success, %d -> %d nodes ---",
                gi + 1, total, old_count, len(fused_nodes),
            )
        else:
            skipped += 1
            if fuse_result == FUSE_REJECTED_LEGALITY:
                metrics.llm_groups_rejected_legality += 1
            elif fuse_result == FUSE_REJECTED_CYCLE:
                metrics.llm_groups_rejected_cycle += 1
            fusion_log.debug(
                "--- llm fusion group (%d/%d): incomplete (%s), %d -> %d nodes ---",
                gi + 1, total, fuse_result, old_count, len(fused_nodes),
            )

    # Rebuild node list (mirrors fuse_nodes_once lines 4205-4206)
    result = sorted(fused_nodes, key=lambda x: x.min_order)
    result = scheduler.topological_sort_schedule(result)

    total_fused = len(nodes) - len(result)
    fusion_log.debug(
        "===== llm fusion complete: %d/%d groups applied, %d skipped, "
        "%d nodes fused (%d -> %d) =====",
        applied, total, skipped, total_fused, len(nodes), len(result),
    )
    fusion_log.info(
        "Applied %d/%d groups, %d skipped (%d nodes fused, %d -> %d)",
        applied, total, skipped, total_fused, len(nodes), len(result),
    )

    return result
