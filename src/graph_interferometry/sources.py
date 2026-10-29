"""Synthetic source families used in the controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

FAMILIES = ("point", "blob", "ring", "jet", "double_lobed")


@dataclass(frozen=True)
class ImageGrid:
    """Square image grid on ``[-extent, extent]^2`` including endpoints."""

    image_size: int
    extent: float
    axis: torch.Tensor
    l_grid: torch.Tensor
    m_grid: torch.Tensor
    lm_pixels: torch.Tensor
    pixel_spacing: float
    pixel_area: float


def make_grid(image_size: int, extent: float = 1.0) -> ImageGrid:
    if image_size < 2:
        raise ValueError("image_size must be at least 2")
    axis = torch.linspace(-extent, extent, image_size)
    l_grid, m_grid = torch.meshgrid(axis, axis, indexing="xy")
    pixel_spacing = float(axis[1] - axis[0])
    return ImageGrid(
        image_size=image_size,
        extent=extent,
        axis=axis,
        l_grid=l_grid,
        m_grid=m_grid,
        lm_pixels=torch.stack([l_grid.flatten(), m_grid.flatten()], dim=1),
        pixel_spacing=pixel_spacing,
        pixel_area=pixel_spacing**2,
    )


def _uniform(low: float, high: float, generator: torch.Generator) -> float:
    value = low + (high - low) * torch.rand(1, generator=generator)
    return float(value.item())


def sample_source_specs(
    family: str,
    count: int,
    seed: int,
    extent: float = 1.0,
) -> list[dict[str, float | str]]:
    """Sample parameter dictionaries before rasterisation.

    The ranges match the revised source generator used in Experiments B-E.
    """

    if family not in FAMILIES:
        raise ValueError(f"Unknown source family: {family}")
    generator = torch.Generator().manual_seed(seed)
    specs: list[dict[str, float | str]] = []

    for _ in range(count):
        spec: dict[str, float | str] = {
            "family": family,
            "l0": _uniform(-0.55 * extent, 0.55 * extent, generator),
            "m0": _uniform(-0.55 * extent, 0.55 * extent, generator),
            "amplitude": _uniform(0.6, 1.0, generator),
        }
        if family == "blob":
            spec["sigma"] = _uniform(0.08, 0.28, generator)
        elif family == "ring":
            spec["radius"] = _uniform(0.18, 0.52, generator)
            spec["width"] = _uniform(0.04, 0.12, generator)
        elif family == "jet":
            spec["angle"] = _uniform(0.0, math.pi, generator)
            spec["length"] = _uniform(0.25, 0.58, generator)
            spec["width"] = _uniform(0.04, 0.14, generator)
        elif family == "double_lobed":
            spec["angle"] = _uniform(0.0, math.pi, generator)
            spec["separation"] = _uniform(0.30, 0.68, generator)
            spec["sigma"] = _uniform(0.07, 0.15, generator)
            spec["core_amplitude"] = _uniform(0.25, 0.55, generator)
            spec["core_sigma"] = _uniform(0.04, 0.08, generator)
            spec["bridge_amplitude"] = _uniform(0.10, 0.25, generator)
            spec["bridge_width"] = _uniform(0.04, 0.09, generator)
            spec["bridge_length_fraction"] = _uniform(0.45, 0.70, generator)
        specs.append(spec)
    return specs


def _normalise_nonnegative(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp_min(0.0)
    peak = image.max()
    return image / peak if peak > 0 else image


def rasterise_source(
    spec: Mapping[str, float | str], grid: ImageGrid
) -> tuple[torch.Tensor, dict[str, object]]:
    """Evaluate one parameterised source on an image grid."""

    family = str(spec["family"])
    l0, m0 = float(spec["l0"]), float(spec["m0"])
    amplitude = float(spec["amplitude"])

    if family == "point":
        image = torch.zeros(grid.image_size, grid.image_size, dtype=torch.float32)
        column = int(torch.argmin(torch.abs(grid.axis - l0)).item())
        row = int(torch.argmin(torch.abs(grid.axis - m0)).item())
        image[row, column] = amplitude
        metadata = dict(spec)
        metadata.update(
            {
                "point_pixel": (row, column),
                "raster_l": float(grid.axis[column]),
                "raster_m": float(grid.axis[row]),
            }
        )
        return _normalise_nonnegative(image), metadata

    dl = grid.l_grid - l0
    dm = grid.m_grid - m0

    if family == "blob":
        sigma = float(spec["sigma"])
        image = amplitude * torch.exp(-(dl.square() + dm.square()) / (2 * sigma**2))
    elif family == "ring":
        radius, width = float(spec["radius"]), float(spec["width"])
        distance = torch.sqrt(dl.square() + dm.square())
        image = amplitude * torch.exp(-(distance - radius).square() / (2 * width**2))
    elif family == "jet":
        angle = float(spec["angle"])
        length, width = float(spec["length"]), float(spec["width"])
        major = dl * math.cos(angle) + dm * math.sin(angle)
        minor = -dl * math.sin(angle) + dm * math.cos(angle)
        image = amplitude * torch.exp(
            -major.square() / (2 * length**2) - minor.square() / (2 * width**2)
        )
    elif family == "double_lobed":
        angle = float(spec["angle"])
        separation = float(spec["separation"])
        sigma = float(spec["sigma"])
        core_amplitude = float(spec["core_amplitude"])
        core_sigma = float(spec["core_sigma"])
        bridge_amplitude = float(spec["bridge_amplitude"])
        bridge_width = float(spec["bridge_width"])
        bridge_length = float(spec["bridge_length_fraction"]) * separation
        half = separation / 2.0
        dl_axis, dm_axis = math.cos(angle), math.sin(angle)
        l1, m1 = l0 + half * dl_axis, m0 + half * dm_axis
        l2, m2 = l0 - half * dl_axis, m0 - half * dm_axis
        first = torch.exp(
            -((grid.l_grid - l1).square() + (grid.m_grid - m1).square())
            / (2 * sigma**2)
        )
        second = torch.exp(
            -((grid.l_grid - l2).square() + (grid.m_grid - m2).square())
            / (2 * sigma**2)
        )
        core = torch.exp(-(dl.square() + dm.square()) / (2 * core_sigma**2))
        major = dl * math.cos(angle) + dm * math.sin(angle)
        minor = -dl * math.sin(angle) + dm * math.cos(angle)
        bridge = torch.exp(
            -major.square() / (2 * bridge_length**2)
            - minor.square() / (2 * bridge_width**2)
        )
        image = (
            amplitude * (first + second)
            + core_amplitude * core
            + bridge_amplitude * bridge
        )
    else:
        raise ValueError(f"Unknown source family: {family}")

    return _normalise_nonnegative(image.float()), dict(spec)


def rasterise_sources(
    specs: Sequence[Mapping[str, float | str]], grid: ImageGrid
) -> dict[str, object]:
    examples = [rasterise_source(spec, grid) for spec in specs]
    images = torch.stack([image for image, _ in examples])
    metadata = [item for _, item in examples]
    return {
        "image": images,
        "flat": images.flatten(start_dim=1),
        "metadata": metadata,
        "family": [str(item["family"]) for item in metadata],
    }

