"""Loss terms retained in the thesis experiments."""

from __future__ import annotations

import torch


def image_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target).square())


def data_fidelity_loss(
    prediction: torch.Tensor,
    observed_visibility: torch.Tensor,
    measurement_operator: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    operator = measurement_operator.to(prediction.device)
    predicted_visibility = prediction.to(operator.dtype) @ operator.T
    numerator = torch.mean(
        torch.abs(predicted_visibility - observed_visibility).square(), dim=1
    )
    denominator = torch.mean(
        torch.abs(observed_visibility).square(), dim=1
    ).clamp_min(epsilon)
    return torch.mean(numerator / denominator)


def combined_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed_visibility: torch.Tensor,
    measurement_operator: torch.Tensor,
    data_fidelity_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    image = image_mse_loss(prediction, target)
    fidelity = data_fidelity_loss(
        prediction, observed_visibility, measurement_operator
    )
    return {
        "total": image + data_fidelity_weight * fidelity,
        "image_mse": image,
        "data_fidelity": fidelity,
    }

