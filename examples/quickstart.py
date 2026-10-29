"""Generate a small batch and run every retained architecture once."""

import torch

from graph_interferometry import (
    AntennaGraphMPNN,
    BaselineLineGraphMPNN,
    FittedOutputBlend,
    build_raw_model_inputs,
    build_scratch_hybrid,
    fit_feature_statistics,
    generate_array,
    make_graph,
    make_grid,
    rasterise_sources,
    sample_source_specs,
    standardise_model_inputs,
)


def main() -> None:
    grid = make_grid(32)
    specs = []
    for family_index, family in enumerate(
        ("point", "blob", "ring", "jet", "double_lobed")
    ):
        specs.extend(sample_source_specs(family, 2, 93117 + 1000 * family_index))
    sky = rasterise_sources(specs, grid)

    layout = generate_array("jittered_circle", number_of_antennas=16, seed=84221)
    graph = make_graph(16)
    raw = build_raw_model_inputs(sky["image"], layout.positions, graph, grid)
    inputs = standardise_model_inputs(raw, fit_feature_statistics(raw))

    antenna = AntennaGraphMPNN(graph, image_size=32)
    baseline = BaselineLineGraphMPNN(graph, image_size=32)
    scratch_hybrid = build_scratch_hybrid(graph, image_size=32)

    with torch.no_grad():
        antenna_prediction = antenna(inputs["antenna_nodes"], inputs["antenna_edges"])
        baseline_prediction = baseline(inputs["baseline_nodes"])
        hybrid_prediction = scratch_hybrid(
            inputs["antenna_nodes"], inputs["antenna_edges"], inputs["baseline_nodes"]
        )
        blend = FittedOutputBlend.fit(
            antenna_prediction, baseline_prediction, inputs["target"]
        )
        blended_prediction = blend(antenna_prediction, baseline_prediction)

    print("Antenna:", tuple(antenna_prediction.shape))
    print("Baseline:", tuple(baseline_prediction.shape))
    print("Scratch hybrid:", tuple(hybrid_prediction.shape))
    print("Fitted output blend:", tuple(blended_prediction.shape))
    print("Demonstration alpha:", float(blend.alpha))
    print("These are untrained shape checks, not scientific reconstructions.")


if __name__ == "__main__":
    main()

