"""Plotting helpers for source and reconstruction galleries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _as_numpy(images: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(images, torch.Tensor):
        return images.detach().cpu().numpy()
    return np.asarray(images)


def plot_reconstruction_gallery(
    targets: torch.Tensor | np.ndarray,
    predictions: Mapping[str, torch.Tensor | np.ndarray],
    row_labels: Sequence[str] | None = None,
    save_path: str | Path | None = None,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> plt.Figure:
    """Plot matched targets and one or more reconstruction methods by row."""

    target_array = _as_numpy(targets)
    if target_array.ndim != 3:
        raise ValueError("targets must have shape [examples, height, width]")
    prediction_arrays = {name: _as_numpy(value) for name, value in predictions.items()}
    for name, array in prediction_arrays.items():
        if array.shape != target_array.shape:
            raise ValueError(f"Prediction shape for {name!r} does not match targets")

    rows = target_array.shape[0]
    columns = 1 + len(prediction_arrays)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.4 * columns, 2.35 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    column_titles = ["Target", *prediction_arrays.keys()]
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)
    for row in range(rows):
        images = [target_array[row], *[array[row] for array in prediction_arrays.values()]]
        for column, image in enumerate(images):
            axes[row, column].imshow(
                image,
                cmap="gray",
                origin="lower",
                vmin=value_range[0],
                vmax=value_range[1],
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        if row_labels is not None:
            axes[row, 0].set_ylabel(row_labels[row], rotation=0, ha="right", va="center")
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=220, bbox_inches="tight")
    return figure

