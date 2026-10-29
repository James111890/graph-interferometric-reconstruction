"""Discrete visibility operator and graph input features."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .arrays import baseline_vectors
from .graphs import GraphStructure
from .sources import ImageGrid


@dataclass(frozen=True)
class FeatureStatistics:
    antenna_node_mean: torch.Tensor
    antenna_node_std: torch.Tensor
    antenna_edge_mean: torch.Tensor
    antenna_edge_std: torch.Tensor
    baseline_node_mean: torch.Tensor
    baseline_node_std: torch.Tensor


def make_visibility_matrix(uv: torch.Tensor, grid: ImageGrid) -> torch.Tensor:
    phase = -2.0 * torch.pi * (uv @ grid.lm_pixels.T)
    kernel = torch.complex(torch.cos(phase), torch.sin(phase))
    return float(grid.pixel_area) * kernel


def add_relative_complex_noise(
    clean: torch.Tensor, noise_fraction: float, seed: int
) -> torch.Tensor:
    if noise_fraction <= 0:
        return clean.clone()
    generator = torch.Generator().manual_seed(seed)
    rms = torch.sqrt(
        torch.mean(torch.abs(clean).square(), dim=1, keepdim=True).clamp_min(1e-12)
    )
    sigma = noise_fraction * rms / math.sqrt(2.0)
    real = torch.randn(clean.real.shape, generator=generator) * sigma
    imaginary = torch.randn(clean.real.shape, generator=generator) * sigma
    return clean + torch.complex(real, imaginary)


def make_physical_baseline_features(
    visibility: torch.Tensor, uv: torch.Tensor
) -> torch.Tensor:
    batch_size, edges = visibility.shape
    length = torch.linalg.vector_norm(uv, dim=1)
    angle = torch.atan2(uv[:, 1], uv[:, 0])
    return torch.stack(
        [
            uv[:, 0].view(1, edges).expand(batch_size, edges),
            uv[:, 1].view(1, edges).expand(batch_size, edges),
            length.view(1, edges).expand(batch_size, edges),
            angle.view(1, edges).expand(batch_size, edges),
            visibility.real,
            visibility.imag,
        ],
        dim=-1,
    ).float()


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def make_antenna_directed_edge_features(
    physical_features: torch.Tensor,
) -> torch.Tensor:
    forward = physical_features.clone()
    reverse = physical_features.clone()
    reverse[:, :, 0] = -forward[:, :, 0]
    reverse[:, :, 1] = -forward[:, :, 1]
    reverse[:, :, 2] = forward[:, :, 2]
    reverse[:, :, 3] = _wrap_to_pi(forward[:, :, 3] + math.pi)
    reverse[:, :, 4] = forward[:, :, 4]
    reverse[:, :, 5] = -forward[:, :, 5]
    return torch.stack([forward, reverse], dim=2).reshape(
        physical_features.shape[0],
        2 * physical_features.shape[1],
        physical_features.shape[2],
    )


def antenna_node_features(positions: torch.Tensor) -> torch.Tensor:
    number_of_antennas = int(positions.shape[0])
    degree = torch.full((number_of_antennas, 1), float(number_of_antennas - 1))
    return torch.cat([positions, degree], dim=1)


def build_raw_model_inputs(
    images: torch.Tensor,
    positions: torch.Tensor,
    graph: GraphStructure,
    grid: ImageGrid,
    noise_fraction: float = 0.0,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Generate visibilities and both graph representations for an image batch."""

    if images.ndim == 3:
        flat = images.flatten(start_dim=1)
    elif images.ndim == 2:
        flat = images
    else:
        raise ValueError("images must have shape [batch, N, N] or [batch, pixels]")
    uv = baseline_vectors(positions, graph.physical_pairs)
    phi = make_visibility_matrix(uv, grid)
    clean = flat.to(phi.dtype) @ phi.T
    observed = add_relative_complex_noise(clean, noise_fraction, seed)
    baseline_nodes = make_physical_baseline_features(observed, uv)
    antenna_edges = make_antenna_directed_edge_features(baseline_nodes)
    node_template = antenna_node_features(positions)
    antenna_nodes = node_template.unsqueeze(0).expand(flat.shape[0], -1, -1).clone()
    return {
        "antenna_node_raw": antenna_nodes,
        "antenna_edge_raw": antenna_edges,
        "baseline_node_raw": baseline_nodes,
        "target": flat.float(),
        "clean_visibility": clean,
        "observed_visibility": observed,
        "uv": uv,
        "phi": phi,
    }


def _feature_statistics(
    tensor: torch.Tensor, dimensions: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        tensor.mean(dim=dimensions, keepdim=True),
        tensor.std(dim=dimensions, keepdim=True).clamp_min(1e-6),
    )


def fit_feature_statistics(raw: dict[str, torch.Tensor]) -> FeatureStatistics:
    antenna_node_mean, antenna_node_std = _feature_statistics(
        raw["antenna_node_raw"], (0, 1)
    )
    antenna_edge_mean, antenna_edge_std = _feature_statistics(
        raw["antenna_edge_raw"], (0, 1)
    )
    baseline_node_mean, baseline_node_std = _feature_statistics(
        raw["baseline_node_raw"], (0, 1)
    )
    return FeatureStatistics(
        antenna_node_mean,
        antenna_node_std,
        antenna_edge_mean,
        antenna_edge_std,
        baseline_node_mean,
        baseline_node_std,
    )


def standardise_model_inputs(
    raw: dict[str, torch.Tensor], statistics: FeatureStatistics
) -> dict[str, torch.Tensor]:
    return {
        **raw,
        "antenna_nodes": (
            raw["antenna_node_raw"] - statistics.antenna_node_mean
        )
        / statistics.antenna_node_std,
        "antenna_edges": (
            raw["antenna_edge_raw"] - statistics.antenna_edge_mean
        )
        / statistics.antenna_edge_std,
        "baseline_nodes": (
            raw["baseline_node_raw"] - statistics.baseline_node_mean
        )
        / statistics.baseline_node_std,
    }

