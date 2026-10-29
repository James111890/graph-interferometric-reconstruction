import torch

from graph_interferometry import (
    AntennaGraphMPNN,
    BaselineLineGraphMPNN,
    FittedOutputBlend,
    build_finetuned_hybrid,
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
from graph_interferometry.models import count_trainable_parameters
from graph_interferometry.training import combined_loss


def _small_problem():
    grid = make_grid(32)
    specs = []
    for index, family in enumerate(("point", "blob", "ring", "jet", "double_lobed")):
        specs.extend(sample_source_specs(family, 1, 100 + index))
    sky = rasterise_sources(specs, grid)
    layout = generate_array("jittered_circle", 16, 84221)
    graph = make_graph(16)
    raw = build_raw_model_inputs(sky["image"], layout.positions, graph, grid)
    processed = standardise_model_inputs(raw, fit_feature_statistics(raw))
    return graph, processed


def test_source_and_graph_shapes():
    graph, inputs = _small_problem()
    assert graph.physical_pairs.shape == (120, 2)
    assert graph.antenna_directed_pairs.shape == (240, 2)
    assert graph.line_directed_pairs.shape == (3480, 2)
    assert inputs["antenna_nodes"].shape == (5, 16, 3)
    assert inputs["antenna_edges"].shape == (5, 240, 6)
    assert inputs["baseline_nodes"].shape == (5, 120, 6)
    assert torch.allclose(inputs["target"].amax(dim=1), torch.ones(5))


def test_model_outputs_and_parameter_counts():
    graph, inputs = _small_problem()
    antenna = AntennaGraphMPNN(graph, 32)
    baseline = BaselineLineGraphMPNN(graph, 32)
    scratch = build_scratch_hybrid(graph, 32)
    fine_tuned = build_finetuned_hybrid(graph, antenna, baseline, 32)

    antenna_output = antenna(inputs["antenna_nodes"], inputs["antenna_edges"])
    baseline_output = baseline(inputs["baseline_nodes"])
    scratch_output = scratch(
        inputs["antenna_nodes"], inputs["antenna_edges"], inputs["baseline_nodes"]
    )
    fine_tuned_output = fine_tuned(
        inputs["antenna_nodes"], inputs["antenna_edges"], inputs["baseline_nodes"]
    )
    assert antenna_output.shape == baseline_output.shape == (5, 1024)
    assert scratch_output.shape == fine_tuned_output.shape == (5, 1024)
    assert count_trainable_parameters(antenna) == 121728
    assert count_trainable_parameters(baseline) == 122816
    assert count_trainable_parameters(scratch) == 177920


def test_output_blend_and_loss_are_finite():
    graph, inputs = _small_problem()
    antenna = AntennaGraphMPNN(graph, 32)
    baseline = BaselineLineGraphMPNN(graph, 32)
    antenna_output = antenna(inputs["antenna_nodes"], inputs["antenna_edges"])
    baseline_output = baseline(inputs["baseline_nodes"])
    blend = FittedOutputBlend.fit(
        antenna_output, baseline_output, inputs["target"]
    )
    prediction = blend(antenna_output, baseline_output)
    losses = combined_loss(
        prediction,
        inputs["target"],
        inputs["observed_visibility"],
        inputs["phi"],
    )
    assert 0.0 <= float(blend.alpha) <= 1.0
    assert all(torch.isfinite(value) for value in losses.values())

