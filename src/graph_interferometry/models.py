"""Standalone and retained hybrid reconstruction models."""

from __future__ import annotations

import torch
import torch.nn as nn

from .graphs import GraphStructure, aggregate_mean_max


def graph_mean_max_readout(node_states: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [node_states.mean(dim=1), node_states.max(dim=1).values], dim=1
    )


class EdgeAwareMessageLayer(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        edge_dim: int,
        message_dim: int = 64,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_input_dim + edge_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_input_dim + 2 * message_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self, node_state: torch.Tensor, edge_features: torch.Tensor, pairs: torch.Tensor
    ) -> torch.Tensor:
        sender, receiver = pairs[:, 0], pairs[:, 1]
        messages = self.message_mlp(
            torch.cat(
                [
                    node_state[:, receiver, :],
                    node_state[:, sender, :],
                    edge_features,
                ],
                dim=-1,
            )
        )
        mean, maximum = aggregate_mean_max(messages, pairs, node_state.shape[1])
        return self.update_mlp(torch.cat([node_state, mean, maximum], dim=-1))


class AntennaEncoder(nn.Module):
    def __init__(
        self, graph: GraphStructure, hidden_dim: int = 64, message_dim: int = 64
    ) -> None:
        super().__init__()
        self.register_buffer(
            "pairs", graph.antenna_directed_pairs.clone(), persistent=False
        )
        self.layer1 = EdgeAwareMessageLayer(3, 6, message_dim, hidden_dim)
        self.layer2 = EdgeAwareMessageLayer(
            hidden_dim, 6, message_dim, hidden_dim
        )

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        first = torch.relu(self.layer1(nodes, edges, self.pairs))
        second = torch.relu(self.layer2(first, edges, self.pairs))
        return graph_mean_max_readout(second)


class LineRawLayer1(nn.Module):
    def __init__(
        self, raw_dim: int = 6, hidden_dim: int = 64, message_dim: int = 64
    ) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * raw_dim + 1, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(raw_dim + 2 * message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, raw: torch.Tensor, pairs: torch.Tensor, relation: torch.Tensor
    ) -> torch.Tensor:
        sender, receiver = pairs[:, 0], pairs[:, 1]
        expanded_relation = relation.unsqueeze(0).expand(raw.shape[0], -1, -1)
        messages = self.message_mlp(
            torch.cat(
                [raw[:, receiver, :], raw[:, sender, :], expanded_relation], dim=-1
            )
        )
        mean, maximum = aggregate_mean_max(messages, pairs, raw.shape[1])
        return self.update_mlp(torch.cat([raw, mean, maximum], dim=-1))


class LineRawLayer2(nn.Module):
    def __init__(
        self, raw_dim: int = 6, hidden_dim: int = 64, message_dim: int = 64
    ) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2 * raw_dim + 1, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2 * message_dim + raw_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        raw: torch.Tensor,
        pairs: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        sender, receiver = pairs[:, 0], pairs[:, 1]
        expanded_relation = relation.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        messages = self.message_mlp(
            torch.cat(
                [
                    hidden[:, receiver, :],
                    hidden[:, sender, :],
                    raw[:, receiver, :],
                    raw[:, sender, :],
                    expanded_relation,
                ],
                dim=-1,
            )
        )
        mean, maximum = aggregate_mean_max(messages, pairs, hidden.shape[1])
        return self.update_mlp(torch.cat([hidden, mean, maximum, raw], dim=-1))


class BaselineLineGraphEncoder(nn.Module):
    def __init__(
        self, graph: GraphStructure, hidden_dim: int = 64, message_dim: int = 64
    ) -> None:
        super().__init__()
        self.register_buffer("pairs", graph.line_directed_pairs.clone(), persistent=False)
        self.register_buffer("relation", graph.line_relation.clone(), persistent=False)
        self.layer1 = LineRawLayer1(6, hidden_dim, message_dim)
        self.layer2 = LineRawLayer2(6, hidden_dim, message_dim)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        first = torch.relu(self.layer1(raw, self.pairs, self.relation))
        second = torch.relu(self.layer2(first, raw, self.pairs, self.relation))
        return graph_mean_max_readout(second)


def _decoder(input_dim: int, image_size: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, image_size**2),
    )


class AntennaGraphMPNN(nn.Module):
    def __init__(
        self,
        graph: GraphStructure,
        image_size: int = 32,
        hidden_dim: int = 64,
        message_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = AntennaEncoder(graph, hidden_dim, message_dim)
        self.decoder = _decoder(2 * hidden_dim, image_size, hidden_dim)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(nodes, edges))


class BaselineLineGraphMPNN(nn.Module):
    """Two-layer line graph with raw baseline-feature reinjection."""

    def __init__(
        self,
        graph: GraphStructure,
        image_size: int = 32,
        hidden_dim: int = 64,
        message_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = BaselineLineGraphEncoder(graph, hidden_dim, message_dim)
        self.decoder = _decoder(2 * hidden_dim, image_size, hidden_dim)

    def forward(self, baseline_nodes: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(baseline_nodes))


class AntennaBaselineHybrid(nn.Module):
    """Concatenate antenna and baseline embeddings before image decoding."""

    def __init__(
        self,
        graph: GraphStructure,
        image_size: int = 32,
        hidden_dim: int = 64,
        message_dim: int = 64,
    ) -> None:
        super().__init__()
        self.antenna_encoder = AntennaEncoder(graph, hidden_dim, message_dim)
        self.baseline_encoder = BaselineLineGraphEncoder(
            graph, hidden_dim, message_dim
        )
        self.decoder = _decoder(4 * hidden_dim, image_size, hidden_dim)

    def forward(
        self,
        antenna_nodes: torch.Tensor,
        antenna_edges: torch.Tensor,
        baseline_nodes: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        antenna_embedding = self.antenna_encoder(antenna_nodes, antenna_edges)
        baseline_embedding = self.baseline_encoder(baseline_nodes)
        reconstruction = self.decoder(
            torch.cat([antenna_embedding, baseline_embedding], dim=1)
        )
        if return_embeddings:
            return reconstruction, antenna_embedding, baseline_embedding
        return reconstruction

    def initialise_from_standalone(
        self,
        antenna_model: AntennaGraphMPNN,
        baseline_model: BaselineLineGraphMPNN,
    ) -> None:
        """Load selected standalone encoders and leave all parameters trainable."""

        self.antenna_encoder.load_state_dict(antenna_model.encoder.state_dict())
        self.baseline_encoder.load_state_dict(baseline_model.encoder.state_dict())


class FittedOutputBlend(nn.Module):
    """Checkpoint-fitted scalar blend of completed A and B reconstructions."""

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    @classmethod
    def fit(
        cls,
        antenna_prediction: torch.Tensor,
        baseline_prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> "FittedOutputBlend":
        direction = baseline_prediction - antenna_prediction
        residual = target - antenna_prediction
        numerator = torch.sum(direction * residual)
        denominator = torch.sum(direction.square()).clamp_min(1e-12)
        alpha = float((numerator / denominator).clamp(0.0, 1.0).item())
        return cls(alpha)

    def forward(
        self, antenna_prediction: torch.Tensor, baseline_prediction: torch.Tensor
    ) -> torch.Tensor:
        return (1.0 - self.alpha) * antenna_prediction + self.alpha * baseline_prediction


def build_scratch_hybrid(
    graph: GraphStructure,
    image_size: int = 32,
    hidden_dim: int = 64,
    message_dim: int = 64,
) -> AntennaBaselineHybrid:
    """Construct the randomly initialised A-B concatenation model."""

    return AntennaBaselineHybrid(graph, image_size, hidden_dim, message_dim)


def build_finetuned_hybrid(
    graph: GraphStructure,
    antenna_model: AntennaGraphMPNN,
    baseline_model: BaselineLineGraphMPNN,
    image_size: int = 32,
    hidden_dim: int = 64,
    message_dim: int = 64,
) -> AntennaBaselineHybrid:
    """Initialise an A-B hybrid from selected standalone encoders."""

    model = AntennaBaselineHybrid(graph, image_size, hidden_dim, message_dim)
    model.initialise_from_standalone(antenna_model, baseline_model)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

