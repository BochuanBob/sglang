"""Topology preprocessing for graph-native LPLB solvers.

The replica placement changes only at startup or during EPLB rebalancing, so
coloring belongs on the host rather than in the per-forward CUDA path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ColoredReplicaTopology:
    eligible_ranks: torch.Tensor
    valid_copies: torch.Tensor
    colored_experts: torch.Tensor
    color_offsets: torch.Tensor
    num_colors: int


def build_colored_replica_topology(
    log2phy: torch.Tensor,
    replicated_logical: torch.Tensor,
    num_physical_experts: int,
    num_gpus: int,
    *,
    require_pairs: bool = False,
) -> ColoredReplicaTopology:
    """Build eligible-rank hyperedges and a greedy vertex-disjoint coloring.

    ``log2phy`` is the global logical-to-physical map with ``-1`` padding.
    Physical experts are assumed to be contiguous by rank, matching
    :class:`LPLBSolver`'s existing assignment matrices.
    """
    if num_physical_experts % num_gpus:
        raise ValueError("physical experts must be evenly partitioned across ranks")
    per_gpu = num_physical_experts // num_gpus
    rows = log2phy.detach().cpu().to(torch.int64)
    logical_ids = replicated_logical.detach().cpu().to(torch.int64).tolist()

    eligible: list[list[int]] = []
    width = log2phy.shape[1]
    for logical_id in logical_ids:
        physical = rows[logical_id]
        physical = physical[physical >= 0]
        ranks = (physical // per_gpu).tolist()
        if len(set(ranks)) != len(ranks):
            raise ValueError(
                f"logical expert {logical_id} has multiple replicas on one rank; "
                "graph-native LPLB currently requires distinct eligible ranks"
            )
        if require_pairs and len(ranks) != 2:
            raise ValueError("edge-balance requires exactly two replicas per expert")
        eligible.append(ranks)

    if not eligible:
        eligible_tensor = torch.empty((0, width), dtype=torch.int32)
        return ColoredReplicaTopology(
            eligible_tensor.to(log2phy.device),
            torch.empty(0, dtype=torch.int32, device=log2phy.device),
            torch.empty(0, dtype=torch.int32, device=log2phy.device),
            torch.tensor([0], dtype=torch.int32, device=log2phy.device),
            0,
        )

    colors: list[list[int]] = []
    occupied: list[set[int]] = []
    for expert, ranks in enumerate(eligible):
        vertices = set(ranks)
        for color, used in zip(colors, occupied):
            if vertices.isdisjoint(used):
                color.append(expert)
                used.update(vertices)
                break
        else:
            colors.append([expert])
            occupied.append(set(vertices))

    colored = [expert for color in colors for expert in color]
    offsets = [0]
    for color in colors:
        offsets.append(offsets[-1] + len(color))
    device = log2phy.device
    padded_eligible = [ranks + [-1] * (width - len(ranks)) for ranks in eligible]
    return ColoredReplicaTopology(
        eligible_ranks=torch.tensor(padded_eligible, dtype=torch.int32, device=device),
        valid_copies=torch.tensor(
            [len(ranks) for ranks in eligible], dtype=torch.int32, device=device
        ),
        colored_experts=torch.tensor(colored, dtype=torch.int32, device=device),
        color_offsets=torch.tensor(offsets, dtype=torch.int32, device=device),
        num_colors=len(colors),
    )
