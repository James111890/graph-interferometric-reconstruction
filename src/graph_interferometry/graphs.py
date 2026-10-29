"""Antenna-centred and baseline-centred graph construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .arrays import make_physical_pairs


@dataclass(frozen=True)
class GraphStructure:
    number_of_antennas: int
    physical_pairs: torch.Tensor
    antenna_directed_pairs: torch.Tensor
    line_undirected_pairs: torch.Tensor
    line_directed_pairs: torch.Tensor
    line_relation: torch.Tensor

    @property
    def number_of_baselines(self) -> int:
        return int(self.physical_pairs.shape[0])


def make_line_graph_pairs(
    number_of_antennas: int, physical_pairs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join baseline nodes exactly when their antenna pairs share an endpoint."""

    parts = []
    for antenna in range(number_of_antennas):
        incident = torch.where(
            (physical_pairs[:, 0] == antenna) | (physical_pairs[:, 1] == antenna)
        )[0]
        parts.append(torch.combinations(incident, r=2))
    undirected = torch.cat(parts, dim=0)
    reverse = undirected[:, [1, 0]]
    node_ids = torch.arange(physical_pairs.shape[0], dtype=torch.long)
    self_pairs = torch.stack([node_ids, node_ids], dim=1)
    directed = torch.cat([undirected, reverse, self_pairs], dim=0)
    return undirected, directed


def make_graph(number_of_antennas: int = 16) -> GraphStructure:
    physical = make_physical_pairs(number_of_antennas)
    antenna_directed = torch.stack([physical, physical[:, [1, 0]]], dim=1).reshape(
        -1, 2
    )
    line_undirected, line_directed = make_line_graph_pairs(
        number_of_antennas, physical
    )
    return GraphStructure(
        number_of_antennas=number_of_antennas,
        physical_pairs=physical,
        antenna_directed_pairs=antenna_directed,
        line_undirected_pairs=line_undirected,
        line_directed_pairs=line_directed,
        line_relation=torch.zeros(line_directed.shape[0], 1),
    )


def aggregate_mean_max(
    messages: torch.Tensor, pairs: torch.Tensor, number_of_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate incoming messages by componentwise mean and maximum."""

    batch_size, number_of_relations, feature_dim = messages.shape
    receiver = pairs[:, 1]
    scatter_index = receiver.view(1, -1, 1).expand(
        batch_size, number_of_relations, feature_dim
    )
    summed = messages.new_zeros((batch_size, number_of_nodes, feature_dim))
    summed.scatter_add_(dim=1, index=scatter_index, src=messages)
    counts = torch.bincount(receiver, minlength=number_of_nodes).to(
        device=messages.device, dtype=messages.dtype
    )
    counts = counts.view(1, number_of_nodes, 1)
    mean = summed / counts.clamp_min(1)
    maximum = messages.new_full(
        (batch_size, number_of_nodes, feature_dim), -torch.inf
    )
    maximum.scatter_reduce_(
        dim=1,
        index=scatter_index,
        src=messages,
        reduce="amax",
        include_self=True,
    )
    maximum = torch.where(counts > 0, maximum, torch.zeros_like(maximum))
    return mean, maximum

