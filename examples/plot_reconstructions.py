"""Create a reconstruction gallery from a saved NumPy archive."""

from argparse import ArgumentParser

import numpy as np

from graph_interferometry.plotting import plot_reconstruction_gallery


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("archive", help="NPZ containing targets and prediction arrays")
    parser.add_argument("output", help="Output PNG path")
    args = parser.parse_args()

    data = np.load(args.archive, allow_pickle=True)
    targets = data["targets"]
    row_labels = data["row_labels"].tolist() if "row_labels" in data else None
    predictions = {
        key: data[key]
        for key in data.files
        if key not in {"targets", "row_labels"}
    }
    plot_reconstruction_gallery(
        targets, predictions, row_labels=row_labels, save_path=args.output
    )


if __name__ == "__main__":
    main()

