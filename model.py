"""SPIN: Spatiotemporal Physics-Inspired Inference Network.

Local temporal encoding provides a PM2.5 initialization and a readout gate.
Physics-inspired graph propagation combines a self channel, two diffusion
operators, and an optional wind-driven advection channel. A sigmoid-gated
readout fuses the graph prediction with the local temporal estimate.

Grid inputs are flattened to N = H * W nodes before the forward pass.
"""

from __future__ import annotations

import math

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class TemporalConvBlock(nn.Module):
    """Temporal convolution block for local feature encoding at each node."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply three temporal convolutions with an additive sigmoid branch.

        Args:
            x: Tensor of shape ``(B, N, T, C_in)``.

        Returns:
            Tensor of shape ``(B, N, T - kernel_size + 1, C_out)``.
        """
        # (B, N, T, C) -> (B, C, N, T) for Conv2d over the time axis
        x = einops.rearrange(x, "b n t c -> b c n t")
        temp = self.conv1(x) + torch.sigmoid(self.conv2(x))
        out = torch.relu(temp + self.conv3(x))
        return einops.rearrange(out, "b c n t -> b n t c")


class PhysicsInspiredGraphConv(nn.Module):
    """Physics-inspired graph propagation with diffusion and advection channels.

    Concatenates self features, two diffusion responses, and an optional
    wind-driven advection response, then applies a learned linear mixing.
    The diffusion operators L^(D_1) and L^(D_2) are the backward random-walk
    and symmetric normalized Laplacians. The advection kernel is aggregated over the input
    time window and normalized over each source node's outgoing edges.
    The advection response sums incoming source features using A^(A)[u, v].
    It is layer-normalized before channel mixing.

    Graph operators are supplied explicitly; this layer learns the channel
    mixing weights, not an adaptive adjacency matrix.
    """

    def __init__(
        self, in_channels: int, out_channels: int, use_advection: bool, activation: str
    ) -> None:
        super().__init__()
        self.activation = activation
        self.use_advection = use_advection
        self.num_channels = 3 + (
            1 if use_advection else 0
        )  # self + random_walk_laplacian + symmetric_laplacian (+ advection_kernel)

        self.mixing_weight = nn.Parameter(
            torch.empty(in_channels * self.num_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

        if self.use_advection:
            self.advection_norm = nn.LayerNorm(in_channels)

    def reset_parameters(self) -> None:
        """Initialize the channel mixing weights and bias uniformly."""
        stdv = 1.0 / math.sqrt(self.mixing_weight.shape[1])
        self.mixing_weight.data.uniform_(-stdv, stdv)
        stdv_b = 1.0 / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv_b, stdv_b)

    def forward(
        self,
        x: torch.Tensor,
        random_walk_laplacian: torch.Tensor,
        symmetric_laplacian: torch.Tensor,
        advection_kernel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mix self, diffusion and optional advection feature channels.

        Args:
            x: Node features, shape ``(B, N, C_in)``.
            random_walk_laplacian: Backward random-walk Laplacian, shape ``(N, N)``.
            symmetric_laplacian: Symmetric Laplacian, shape ``(N, N)``.
            advection_kernel: Optional advection kernel, shape ``(B, N, N)``.

        Returns:
            Node features, shape ``(B, N, C_out)``.
        """
        batch_size, num_node, input_size = x.shape

        # diffusion channels: shared (N, N) kernels applied to all batch/feature columns
        x0 = einops.rearrange(x, "b n c -> n (c b)")  # (N, C*B)
        supports = [torch.unsqueeze(x0, 0)]  # identity (self) channel
        supports.append(torch.mm(random_walk_laplacian, x0).unsqueeze(0))
        supports.append(torch.mm(symmetric_laplacian, x0).unsqueeze(0))
        stacked = torch.cat(supports, dim=0)  # (3, N, C*B)

        if self.use_advection:
            assert advection_kernel is not None, "advection kernel required when use_advection=True"
            # per-batch advection: (B, N, C) propagated by (B, N, N)
            x_wind = einops.rearrange(x, "b n c -> b c n")
            x_wind = torch.einsum("bcn,bnm->bcm", x_wind, advection_kernel)  # (B, C, N)
            x_wind = self.advection_norm(einops.rearrange(x_wind, "b c n -> b n c"))
            x_wind = einops.rearrange(x_wind, "b n c -> n (c b)").unsqueeze(0)
            stacked = torch.cat([stacked, x_wind], dim=0)  # (M, N, C*B)

        # (M, N, C, B) -> (B, N, C*M) then linear mix
        stacked = einops.rearrange(
            stacked, "m n (c b) -> b n (c m)", b=batch_size, c=input_size, m=self.num_channels
        )
        out = torch.matmul(stacked, self.mixing_weight) + self.bias

        if self.activation == "relu":
            out = F.relu(out)
        elif self.activation == "selu":
            out = F.selu(out)
        return out


class SPIN(nn.Module):
    """Physics-inspired GNN for PM2.5 spatiotemporal kriging.

    ``forward((data, graph))`` uses these standardized input tensors:

      - ``data["pm25"]``: ``(B,T,N,pm25_dim)`` concentration field. Values at
        unobserved nodes are replaced internally by temporal estimates.
      - ``data["meteorology"]``: ``(B,T,N,meteorology_dim)`` meteorology.
      - ``data["emissions"]``: ``(B,T,N,emission_dim)``, when enabled.
      - ``graph["random_walk_laplacian"]``: ``(N,N)`` backward random-walk
        Laplacian, corresponding to L^(D_1) in the paper.
      - ``graph["symmetric_laplacian"]``: ``(N,N)`` normalized Laplacian,
        corresponding to L^(D_2).
      - ``graph["advection_kernel"]``: ``(B,N,N)`` temporally aggregated
        kernel, with source rows and target columns, when enabled.
      - ``graph["observed_mask"]``: ``(N,)`` or ``(B,N)``, 1 for observed nodes.

    The graph input is the observed or initialized PM2.5 history. Temporal
    features feed the initialization and fusion gate, without direct
    concatenation into the graph input. The final graph convolution performs
    the shared linear channel mixing that produces the graph prediction.

    Returns ``(prediction, initial_prediction)``: the gated graph readout
    and the temporal-only estimate, both shaped ``(B,T,N,pm25_dim)``.
    """

    def __init__(
        self,
        meteorology_dim: int,
        emission_dim: int,
        pm25_dim: int,
        sequence_length: int,
        graph_hidden_dim: int,
        temporal_hidden_dim: int,
        temporal_kernel_size: int,
        use_emissions: bool,
        use_advection: bool,
        use_gated_readout: bool,
    ) -> None:
        super().__init__()

        dimensions = (
            meteorology_dim,
            pm25_dim,
            sequence_length,
            graph_hidden_dim,
            temporal_hidden_dim,
            temporal_kernel_size,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in dimensions
        ):
            raise ValueError("Model dimensions must be positive integers")
        if not isinstance(emission_dim, int) or isinstance(emission_dim, bool) or emission_dim < 0:
            raise ValueError("emission_dim must be a nonnegative integer")
        if use_emissions and emission_dim == 0:
            raise ValueError("emission_dim must be positive when emissions are enabled")
        if sequence_length < 2 * temporal_kernel_size - 1:
            raise ValueError("sequence_length must be at least 2*temporal_kernel_size-1")
        self.meteorology_dim = meteorology_dim
        self.emission_dim = emission_dim
        self.pm25_dim = pm25_dim
        self.sequence_length = sequence_length
        self.use_emissions = use_emissions
        self.use_advection = use_advection
        self.use_gated_readout = use_gated_readout

        in_dim = meteorology_dim + (emission_dim if use_emissions else 0)
        self.in_dim = in_dim

        # Local temporal encoding with two convolution blocks.
        self.temporal_block1 = TemporalConvBlock(
            in_channels=in_dim, out_channels=temporal_hidden_dim, kernel_size=temporal_kernel_size
        )
        self.temporal_block2 = TemporalConvBlock(
            in_channels=temporal_hidden_dim,
            out_channels=temporal_hidden_dim,
            kernel_size=temporal_kernel_size,
        )

        # Two temporal blocks reduce the time axis by 2*(temporal_kernel_size - 1).
        self.temporal_feature_dim = temporal_hidden_dim * (
            sequence_length - 2 * temporal_kernel_size + 2
        )
        self.initialization_readout = nn.Linear(
            self.temporal_feature_dim, sequence_length * pm25_dim
        )  # local temporal estimate
        self.fusion_gate = nn.Linear(
            self.temporal_feature_dim, sequence_length * pm25_dim
        )  # sigmoid readout gate

        # Physics-inspired propagation; the middle layer uses a residual update.
        self.graph_conv1 = PhysicsInspiredGraphConv(
            sequence_length * pm25_dim, graph_hidden_dim, use_advection, activation="relu"
        )
        self.graph_conv2 = PhysicsInspiredGraphConv(
            graph_hidden_dim, graph_hidden_dim, use_advection, activation="relu"
        )
        self.graph_readout = PhysicsInspiredGraphConv(
            graph_hidden_dim, sequence_length * pm25_dim, use_advection, activation="linear"
        )

    def forward(
        self, model_in: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the SPIN forward pass.

        Args:
            model_in: ``(data, graph)`` dicts as documented on the class.

        Returns:
            Tuple ``(prediction, initial_prediction)`` each of shape ``(B, T, N, pm25_dim)``.
        """
        data, graph = model_in
        pm25 = data["pm25"]
        if pm25.ndim != 4:
            raise ValueError("pm25 must have shape (B,T,N,F)")
        bs, ts, n, f = pm25.shape
        if bs < 1 or n < 1 or ts != self.sequence_length or f != self.pm25_dim:
            raise ValueError("pm25 shape is inconsistent with model dimensions")
        if data["meteorology"].shape != (bs, ts, n, self.meteorology_dim):
            raise ValueError("meteorology shape is inconsistent with model dimensions")
        if self.use_emissions and data["emissions"].shape != (bs, ts, n, self.emission_dim):
            raise ValueError("emissions shape is inconsistent with model dimensions")
        if graph["random_walk_laplacian"].shape != (n, n) or graph["symmetric_laplacian"].shape != (
            n,
            n,
        ):
            raise ValueError("Diffusion operators must have shape (N,N)")
        if self.use_advection and graph["advection_kernel"].shape != (bs, n, n):
            raise ValueError("advection_kernel must have shape (B,N,N)")
        observed = graph["observed_mask"]
        if observed.shape not in {(n,), (bs, n)} or not torch.all(
            (observed == 0) | (observed == 1)
        ):
            raise ValueError("observed_mask must be binary with shape (N,) or (B,N)")

        random_walk_laplacian, symmetric_laplacian = (
            graph["random_walk_laplacian"],
            graph["symmetric_laplacian"],
        )
        advection_kernel = graph["advection_kernel"] if self.use_advection else None

        # (B, T, N, pm25_dim) -> (B, N, T*pm25_dim)
        x = einops.rearrange(pm25, "b t n f -> b n (t f)")

        # Local temporal encoding of meteorology and emissions.
        temporal_features = data["meteorology"]
        if self.use_emissions:
            temporal_features = torch.cat([temporal_features, data["emissions"]], dim=-1)
        temporal_features = einops.rearrange(temporal_features, "b t n c -> b n t c")

        temporal_features = torch.relu(self.temporal_block1(temporal_features))
        temporal_features = torch.relu(self.temporal_block2(temporal_features))
        temporal_features = einops.rearrange(temporal_features, "b n t c -> b n (t c)")
        initial_prediction = self.initialization_readout(temporal_features).type_as(
            pm25
        )  # (B, N, T*pm25_dim)
        fusion_weight = torch.sigmoid(self.fusion_gate(temporal_features))  # (B, N, T*pm25_dim)

        # Initialize unobserved nodes with the local estimate X_init.
        observed_mask = graph["observed_mask"]
        target_nodes = observed_mask == 0
        x = x.clone()
        if target_nodes.ndim == 1:
            target_nodes = target_nodes.unsqueeze(0)
        x = torch.where(target_nodes[..., None], initial_prediction, x)

        # Diffusion and advection propagation, followed by the graph readout.
        hidden1 = self.graph_conv1(x, random_walk_laplacian, symmetric_laplacian, advection_kernel)
        hidden2 = (
            self.graph_conv2(hidden1, random_walk_laplacian, symmetric_laplacian, advection_kernel)
            + hidden1
        )  # residual update
        graph_prediction = self.graph_readout(
            hidden2, random_walk_laplacian, symmetric_laplacian, advection_kernel
        )

        graph_prediction = einops.rearrange(
            graph_prediction, "b n (t f) -> b t n f", t=ts, f=self.pm25_dim
        )
        initial_prediction = einops.rearrange(
            initial_prediction, "b n (t f) -> b t n f", t=ts, f=self.pm25_dim
        )
        fusion_weight = einops.rearrange(
            fusion_weight, "b n (t f) -> b t n f", t=ts, f=self.pm25_dim
        )

        # Paper readout: X_hat = (1 - pi) * X_graph + pi * X_init.
        if self.use_gated_readout:
            prediction = (1 - fusion_weight) * graph_prediction + fusion_weight * initial_prediction
        else:
            prediction = graph_prediction

        return prediction, initial_prediction


def _normalized_laplacian(adj: np.ndarray, norm: str) -> np.ndarray:
    """Dense normalized Laplacian with a zero-row isolated-node convention.

    Input adjacency has no self-loops. Degree-zero rows receive no identity
    entry; other rows receive an identity term minus normalized adjacency.
    This defines this implementation's convention rather than asserting
    equivalence to any external graph-library version.

    Args:
        adj: Nonnegative, finite, zero-diagonal adjacency, shape ``(N, N)``.
        norm: ``"rw"`` for ``I - D^{-1} A`` or ``"sym"`` for
            ``I - D^{-1/2} A D^{-1/2}``.

    Returns:
        Dense Laplacian, shape ``(N, N)``.
    """
    adj = np.asarray(adj, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or adj.shape[0] == 0:
        raise ValueError("Adjacency must be a nonempty square matrix")
    if not np.all(np.isfinite(adj)) or np.any(adj < 0):
        raise ValueError("Adjacency must be finite and nonnegative")
    if np.any(np.diag(adj) != 0):
        raise ValueError("Adjacency must have a zero diagonal")
    deg = adj.sum(axis=1)  # weighted out-degree
    active = deg > 0
    eye = np.diag(active.astype(np.float64))  # self-loop only for non-isolated nodes

    if norm == "rw":
        d_inv = np.zeros_like(deg)
        d_inv[active] = 1.0 / deg[active]
        lap = eye - d_inv[:, None] * adj
    elif norm == "sym":
        d_inv_sqrt = np.zeros_like(deg)
        d_inv_sqrt[active] = 1.0 / np.sqrt(deg[active])
        lap = eye - d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    else:
        raise ValueError(f"unknown norm: {norm}")

    return lap.astype(np.float32)


def build_diffusion_operators(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the two diffusion operators used in physics-inspired propagation.

    Returns L^(D_1), the random-walk Laplacian of transposed adjacency,
    and L^(D_2), the symmetric normalized Laplacian of the adjacency.

    Args:
        adj: Nonnegative, finite, zero-diagonal adjacency, shape ``(N, N)``.

    Returns:
        Tuple ``(random_walk_laplacian, symmetric_laplacian)`` each of shape ``(N, N)``, float32.
    """
    random_walk_laplacian = _normalized_laplacian(np.asarray(adj).T, norm="rw")
    symmetric_laplacian = _normalized_laplacian(np.asarray(adj), norm="sym")
    return random_walk_laplacian, symmetric_laplacian


def build_advection_kernel(
    wind_speed: torch.Tensor,
    wind_direction: torch.Tensor,
    edge_distance: torch.Tensor,
    edge_bearing: torch.Tensor,
    edge_adjacency: torch.Tensor,
    temporal_topk: int,
) -> torch.Tensor:
    """Build the wind-projected, directional advection adjacency.

    For every directed edge ``(source -> target)`` the wind speed/direction at the *source*
    node is projected onto the edge bearing; the positive projection divided by
    edge distance gives the advective coupling. The per-edge values are then
    placed into a dense ``(B, T, N, N)`` tensor, the top-``k`` over time is taken
    and averaged, yielding a ``(B, N, N)`` advection kernel.

    Args:
        wind_speed: Wind speed in km/h, shape ``(B, T, N)``.
        wind_direction: Wind travel direction in degrees, using the edge-bearing convention, shape ``(B, T, N)``.
        edge_distance: Edge distance matrix (km), shape ``(N, N)``.
        edge_bearing: Edge bearing matrix (deg), shape ``(N, N)``.
        edge_adjacency: Adjacency with the diagonal removed, shape ``(N, N)``; its
            non-zero pattern defines the active directed edges.
        temporal_topk: Number of strongest timesteps to average over.

    Returns:
        Advection kernel, shape ``(B, N, N)``.
    """
    if (
        wind_speed.ndim != 3
        or wind_direction.shape != wind_speed.shape
        or min(wind_speed.shape) < 1
    ):
        raise ValueError("Wind tensors must have matching nonempty (B,T,N) shapes")
    if not isinstance(temporal_topk, int) or isinstance(temporal_topk, bool) or temporal_topk <= 0:
        raise ValueError("temporal_topk must be a positive integer")
    n = wind_speed.shape[-1]
    if any(matrix.shape != (n, n) for matrix in (edge_distance, edge_bearing, edge_adjacency)):
        raise ValueError("Graph matrices must have shape (N,N)")
    if any(
        tensor.device != wind_speed.device
        for tensor in (wind_direction, edge_distance, edge_bearing, edge_adjacency)
    ):
        raise ValueError("Wind and graph tensors must be on the same device")
    if not torch.all(torch.isfinite(wind_speed) & (wind_speed >= 0)) or not torch.all(
        torch.isfinite(wind_direction)
    ):
        raise ValueError("Wind values must be finite and speeds nonnegative")
    if not torch.all(torch.isfinite(edge_adjacency)) or torch.any(edge_adjacency.diagonal() != 0):
        raise ValueError("Adjacency must be finite with a zero diagonal")
    active = edge_adjacency != 0
    if not torch.all(torch.isfinite(edge_distance[active]) & (edge_distance[active] > 0)):
        raise ValueError("Active edge distances must be finite and positive")
    if not torch.all(torch.isfinite(edge_bearing[active])):
        raise ValueError("Active edge bearings must be finite")
    if not wind_speed.is_floating_point() or not wind_direction.is_floating_point():
        raise ValueError("Wind tensors must be floating-point")
    device = wind_speed.device
    output_dtype = wind_speed.dtype
    dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype
    wind_speed = wind_speed.to(dtype)
    wind_direction = wind_direction.to(dtype)
    b, t, n = wind_speed.shape

    edge_mask = edge_adjacency != 0  # (N, N) boolean, directed edges
    source_index, target_index = torch.nonzero(edge_mask, as_tuple=True)  # source, target

    # wind state at the source node of each edge
    source_wind_speed = wind_speed[:, :, source_index]  # (B, T, E)
    source_wind_direction = wind_direction[:, :, source_index]  # (B, T, E)

    dist = edge_distance[edge_mask].to(device=device, dtype=dtype)  # (E,)
    bearing = edge_bearing[edge_mask].to(device=device, dtype=dtype)  # (E,)

    dist_b = dist[None, None, :]
    bearing_b = bearing[None, None, :]

    # cosine of the angle between wind direction and edge bearing, clamped to >=0
    alpha = torch.deg2rad(torch.abs(bearing_b - source_wind_direction))
    advec = torch.relu(torch.cos(alpha)) * source_wind_speed / dist_b  # (B, T, E)

    a_wind = torch.zeros((b, t, n, n), dtype=dtype, device=device)
    a_wind[:, :, source_index, target_index] = advec

    k = min(temporal_topk, t)
    a_wind = torch.topk(a_wind, k=k, dim=1).values.mean(dim=1)  # (B, N, N)

    # row-normalize so each source node's outgoing advection sums approximately to 1 (advection kernel)
    row_sum = a_wind.sum(dim=-1, keepdim=True)
    a_wind = a_wind / (row_sum + 1e-8)
    return a_wind.to(output_dtype)
