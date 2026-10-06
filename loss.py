"""SPIN inference, initialization and AOD spatial-gradient objectives."""

from __future__ import annotations

import math

import einops
import torch


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    time_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute error over selected nodes, with optional time weights.

    Args:
        prediction: Standardized predictions, shape ``(B, T, N, F)``.
        target: Standardized targets with the same shape.
        node_mask: Binary selection mask, shape ``(N,)`` or ``(B, N)``.
        time_weights: Optional nonnegative weights, shape ``(T,)``. Reduction
            divides by the selected element count, not the sum of weights.

    Returns:
        Scalar loss. An empty selection gives a differentiable zero.
    """
    if prediction.ndim != 4 or target.shape != prediction.shape:
        raise ValueError("prediction and target must have matching (B,T,N,F) shapes")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("prediction and target must be floating-point tensors")
    dtype = torch.promote_types(prediction.dtype, target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    prediction = prediction.to(dtype)
    target = target.to(dtype)
    b, t, n, _ = prediction.shape
    if node_mask.shape not in {(n,), (b, n)}:
        raise ValueError("node_mask must have shape (N,) or (B,N)")
    if not torch.all((node_mask == 0) | (node_mask == 1)):
        raise ValueError("node_mask must be binary")
    if time_weights is not None:
        if time_weights.shape != (t,):
            raise ValueError("time_weights must have shape (T,)")
        if not torch.all(torch.isfinite(time_weights) & (time_weights >= 0)):
            raise ValueError("time_weights must be finite and nonnegative")
    selected = node_mask == 1
    if selected.ndim == 1:
        selected = selected.unsqueeze(0).expand(b, -1)
    selected = selected[:, None, :, None].expand_as(prediction)
    # Select before arithmetic so missing values outside the mask cannot leak.
    errors = (prediction[selected] - target[selected]).abs()
    if errors.numel() == 0:
        return errors.sum()
    if time_weights is not None:
        weights = time_weights[None, :, None, None].expand_as(prediction)
        errors = errors * weights[selected]
    return errors.mean()


def aod_spatial_gradient_loss(
    prediction: torch.Tensor,
    aod: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked AOD spatial-gradient constraint, L_AOD in the paper.

    Compare standardized finite differences along horizontal, vertical and
    both diagonal grid edges. Each edge requires two valid AOD endpoints,
    corresponding to Omega_ij = omega_i * omega_j in the paper.

    Args:
        prediction: Standardized PM2.5 field, shape ``(B,T,H,W)``.
        aod: Standardized AOD field with the same shape. This function does
            not estimate normalization statistics or standardize the fields.
        valid_mask: Binary AOD availability mask with the same shape.
            Non-finite AOD is always excluded, even if marked valid.

    Returns:
        Mean absolute difference of edge differences across all valid edges.
        Edges need two valid endpoints. No valid edges gives differentiable zero.
        Differences are not divided by distance; diagonals have equal weight.
        Half-precision operands are accumulated in float32 to avoid overflow.
    """
    if (
        prediction.ndim != 4
        or aod.shape != prediction.shape
        or valid_mask.shape != prediction.shape
    ):
        raise ValueError("prediction, aod and valid_mask must have matching (B,T,H,W) shapes")
    if not torch.all((valid_mask == 0) | (valid_mask == 1)):
        raise ValueError("valid_mask must be binary")
    if not prediction.is_floating_point() or not aod.is_floating_point():
        raise ValueError("prediction and aod must be floating-point tensors")
    dtype = torch.promote_types(prediction.dtype, aod.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    prediction = prediction.to(dtype)
    aod = aod.to(dtype)
    valid = (valid_mask == 1) & torch.isfinite(aod)
    total = prediction[..., :0].sum()
    count = torch.zeros((), dtype=torch.int64, device=prediction.device)
    directions = (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None))),
        ((slice(1, None), slice(None, -1)), (slice(None, -1), slice(1, None))),
    )
    for start, end in directions:
        left = (slice(None), slice(None), *start)
        right = (slice(None), slice(None), *end)
        edge = valid[left] & valid[right]
        pred_difference = prediction[left][edge] - prediction[right][edge]
        aod_difference = aod[left][edge] - aod[right][edge]
        total = total + (pred_difference - aod_difference).abs().sum()
        count = count + edge.sum()
    return total / count.clamp_min(1)


def spin_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    inference_mask: torch.Tensor,
    *,
    initial_prediction: torch.Tensor | None = None,
    initialization_mask: torch.Tensor | None = None,
    time_weights: torch.Tensor | None = None,
    aod: torch.Tensor | None = None,
    aod_mask: torch.Tensor | None = None,
    grid_shape: tuple[int, int] | None = None,
    initialization_weight: float,
    aod_weight: float,
) -> dict[str, torch.Tensor]:
    """Compose L_infer + lambda_1 * L_init + lambda_2 * L_AOD.

    initialization_weight and aod_weight correspond to lambda_1 and lambda_2.

    Prediction/target use ``(B,T,N,F)``. AOD uses ``(B,T,H,W)`` and requires
    ``N=H*W`` and a single PM2.5 feature. Callers supply masks, coefficients
    and any temporal weights explicitly; this API contains no experiment recipe.
    For the paper objective, inference_mask selects masked target stations
    (V_target), and initialization_mask selects observed trainable stations
    (V_obs). Exclude held-out stations and stationless cells from both masks.
    Use time_weights=None for the unweighted objective in the paper.
    Initialization uses its own caller-supplied mask and the same target.
    A positive coefficient requires the associated tensors and mask.

    Returns:
        Dictionary with scalar ``total``, ``inference``, ``initialization``
        and ``aod`` terms. Component values are unweighted.
    """
    if any(
        not math.isfinite(weight) or weight < 0 for weight in (initialization_weight, aod_weight)
    ):
        raise ValueError("Loss coefficients must be finite and nonnegative")
    inference = masked_l1_loss(prediction, target, inference_mask, time_weights)
    initialization = prediction[..., :0].sum()
    gradient = prediction[..., :0].sum()
    if initialization_weight > 0:
        if initial_prediction is None or initialization_mask is None:
            raise ValueError(
                "Initialization loss requires initial_prediction and initialization_mask"
            )
        initialization = masked_l1_loss(
            initial_prediction, target, initialization_mask, time_weights
        )
    if aod_weight > 0:
        if aod is None or aod_mask is None or grid_shape is None:
            raise ValueError("AOD loss requires aod, aod_mask and grid_shape")
        h, w = grid_shape
        if h <= 0 or w <= 0 or prediction.shape[2:] != (h * w, 1):
            raise ValueError("AOD loss requires N=H*W and one prediction feature")
        field = einops.rearrange(prediction[..., 0], "b t (h w) -> b t h w", h=h, w=w)
        gradient = aod_spatial_gradient_loss(field, aod, aod_mask)
    return {
        "total": inference + initialization_weight * initialization + aod_weight * gradient,
        "inference": inference,
        "initialization": initialization,
        "aod": gradient,
    }
