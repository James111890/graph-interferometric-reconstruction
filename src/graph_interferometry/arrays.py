"""Controlled synthetic interferometer array layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

DEFAULT_NORMALISED_MAX_BASELINE = 2.1269912719726562
DEFAULT_PHYSICAL_MAX_BASELINE_KM = 74.0
SKA_CORE_FRACTION = 224.0 / 512.0


@dataclass(frozen=True)
class ArrayLayout:
    name: str
    positions: torch.Tensor
    positions_km: torch.Tensor
    maximum_baseline: float
    metadata: dict[str, object] = field(default_factory=dict)


def make_physical_pairs(number_of_antennas: int) -> torch.Tensor:
    if number_of_antennas < 2:
        raise ValueError("At least two antennas are required")
    return torch.tensor(
        [
            (i, j)
            for i in range(number_of_antennas)
            for j in range(i + 1, number_of_antennas)
        ],
        dtype=torch.long,
    )


def centre_positions(positions: torch.Tensor) -> torch.Tensor:
    return positions - positions.mean(dim=0, keepdim=True)


def baseline_vectors(
    positions: torch.Tensor, physical_pairs: torch.Tensor | None = None
) -> torch.Tensor:
    if physical_pairs is None:
        physical_pairs = make_physical_pairs(int(positions.shape[0]))
    return positions[physical_pairs[:, 1]] - positions[physical_pairs[:, 0]]


def maximum_baseline(
    positions: torch.Tensor, physical_pairs: torch.Tensor | None = None
) -> float:
    vectors = baseline_vectors(positions, physical_pairs)
    return float(torch.linalg.vector_norm(vectors, dim=1).max().item())


def rescale_to_maximum_baseline(
    positions: torch.Tensor, target: float
) -> torch.Tensor:
    positions = centre_positions(positions)
    current = maximum_baseline(positions)
    if current <= 0:
        raise ValueError("Degenerate array geometry")
    return positions * (target / current)


def _distribute_count(total: int, groups: int = 3) -> list[int]:
    counts = [total // groups] * groups
    for index in range(total % groups):
        counts[index] += 1
    return counts


def jittered_circle_positions(number_of_antennas: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    base_angles = (
        torch.arange(number_of_antennas, dtype=torch.float32)
        * (2 * math.pi / number_of_antennas)
    )
    angles = base_angles + 0.08 * torch.randn(number_of_antennas, generator=generator)
    radii = (1.0 + 0.08 * torch.randn(number_of_antennas, generator=generator)).clamp_min(0.25)
    return centre_positions(
        torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)
    )


def irregular_disk_positions(number_of_antennas: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    radii = torch.sqrt(torch.rand(number_of_antennas, generator=generator))
    angles = 2 * math.pi * torch.rand(number_of_antennas, generator=generator)
    return centre_positions(
        torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)
    )


def three_arm_y_positions(number_of_antennas: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    positions = [torch.tensor([0.0, 0.0])]
    for arm, count in enumerate(_distribute_count(number_of_antennas - 1)):
        if count == 0:
            continue
        base_angle = 2 * math.pi * arm / 3
        for radius in torch.linspace(0.08, 1.0, count):
            angle = base_angle + float(0.012 * torch.randn(1, generator=generator))
            adjusted_radius = float(radius) * (
                1.0 + float(0.015 * torch.randn(1, generator=generator))
            )
            positions.append(
                torch.tensor(
                    [
                        adjusted_radius * math.cos(angle),
                        adjusted_radius * math.sin(angle),
                    ]
                )
            )
    return centre_positions(torch.stack(positions).float())


def two_cluster_positions(number_of_antennas: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    left_count = number_of_antennas // 2
    right_count = number_of_antennas - left_count
    left = 0.11 * torch.randn((left_count, 2), generator=generator)
    right = 0.11 * torch.randn((right_count, 2), generator=generator)
    left[:, 0] -= 1.0
    right[:, 0] += 1.0
    left[:, 1] -= 0.12
    right[:, 1] += 0.12
    return centre_positions(torch.cat([left, right]))


def ska_low_design_analogue_positions(
    number_of_antennas: int,
    seed: int,
    target_maximum_baseline_km: float = DEFAULT_PHYSICAL_MAX_BASELINE_KM,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Generate the thesis' structural SKA-Low analogue.

    This is not an official SKA-Low station layout. Approximately 44 percent
    of the points are sampled in a one-kilometre-diameter disk. The remaining
    points are divided as evenly as possible among three winding arms.
    """

    generator = torch.Generator().manual_seed(seed)
    core_count = int(round(SKA_CORE_FRACTION * number_of_antennas))
    remote_count = number_of_antennas - core_count

    core_angles = 2 * math.pi * torch.rand(core_count, generator=generator)
    core_radii = 0.5 * torch.sqrt(torch.rand(core_count, generator=generator))
    core = torch.stack(
        [core_radii * torch.cos(core_angles), core_radii * torch.sin(core_angles)],
        dim=1,
    )

    remote: list[torch.Tensor] = []
    remote_by_arm = _distribute_count(remote_count)
    inner_km, outer_km, spiral_strength = 1.2, 42.75, 0.26
    for arm, count in enumerate(remote_by_arm):
        if count == 0:
            continue
        radii = torch.logspace(math.log10(inner_km), math.log10(outer_km), count)
        for index, radius in enumerate(radii):
            radial_distance = float(radius)
            angle = (
                2 * math.pi * arm / 3
                + spiral_strength * math.log(radial_distance / inner_km)
            )
            if index not in (0, count - 1):
                angle += float(0.012 * torch.randn(1, generator=generator))
                radial_distance *= 1.0 + float(
                    0.015 * torch.randn(1, generator=generator)
                )
            remote.append(
                torch.tensor(
                    [radial_distance * math.cos(angle), radial_distance * math.sin(angle)]
                )
            )

    remote_tensor = torch.stack(remote).float()
    pairs = make_physical_pairs(number_of_antennas)
    low, high = 0.5, 1.5
    for _ in range(50):
        scale = 0.5 * (low + high)
        trial = torch.cat([core, remote_tensor * scale])
        if maximum_baseline(trial, pairs) < target_maximum_baseline_km:
            low = scale
        else:
            high = scale
    positions = centre_positions(torch.cat([core, remote_tensor * (0.5 * (low + high))]))
    return positions, {
        "core_nodes": core_count,
        "remote_nodes": remote_count,
        "remote_nodes_by_arm": tuple(remote_by_arm),
        "structural_analogue_only": True,
    }


def generate_array(
    name: str = "jittered_circle",
    number_of_antennas: int = 16,
    seed: int = 84221,
    physical_maximum_baseline_km: float = DEFAULT_PHYSICAL_MAX_BASELINE_KM,
    normalised_maximum_baseline: float = DEFAULT_NORMALISED_MAX_BASELINE,
) -> ArrayLayout:
    """Create one controlled layout and its dimensionless simulation coordinates."""

    metadata: dict[str, object] = {}
    if name == "jittered_circle":
        raw = jittered_circle_positions(number_of_antennas, seed)
        positions_km = rescale_to_maximum_baseline(raw, physical_maximum_baseline_km)
    elif name == "irregular_disk":
        raw = irregular_disk_positions(number_of_antennas, seed)
        positions_km = rescale_to_maximum_baseline(raw, physical_maximum_baseline_km)
    elif name == "three_arm_y":
        raw = three_arm_y_positions(number_of_antennas, seed)
        positions_km = rescale_to_maximum_baseline(raw, physical_maximum_baseline_km)
    elif name == "two_cluster":
        raw = two_cluster_positions(number_of_antennas, seed)
        positions_km = rescale_to_maximum_baseline(raw, physical_maximum_baseline_km)
    elif name == "ska_low_design_analogue":
        positions_km, metadata = ska_low_design_analogue_positions(
            number_of_antennas, seed, physical_maximum_baseline_km
        )
    else:
        raise ValueError(f"Unknown array family: {name}")

    physical_max = maximum_baseline(positions_km)
    positions = positions_km * (normalised_maximum_baseline / physical_max)
    return ArrayLayout(
        name=name,
        positions=positions,
        positions_km=positions_km,
        maximum_baseline=maximum_baseline(positions),
        metadata=metadata,
    )

