"""CORAL v4.2 — Precision-weighted predictive coding.

v4.2 design change: learned PrecisionNetwork replaced by RunningPrecision.
Eight failed attempts (see handoff note) established that learned precision
via backprop collapses to uniform output and adds training instability.
RunningPrecision uses running EMA statistics of the prediction error variance
and has ZERO learnable parameters — precision is a constant multiplier for
gradient purposes.

Implements the core free energy mechanism:
  1. PredictionNetwork:  top-down prediction  μ_l = f(z_{l+1})
  2. RunningPrecision:   per-dim precision    π_l = 1 / (EMA_var + eps)  [no params]
  3. PredictiveCodingModule: combines prediction, error, precision

CRITICAL: precision is treated as a constant in the gradient graph.
RunningPrecision.update() is decorated with @torch.no_grad() and the
ema_var buffer does not have requires_grad=True.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from coral.config import ModelConfig


# ---------------------------------------------------------------------------
# Prediction network: top-down prediction of level l from level l+1
# ---------------------------------------------------------------------------


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm without learnable affine parameters.

    Normalizes each token/sample so that its root-mean-square magnitude is 1.
    Scale-invariant: output is unaffected by the absolute scale of the input.

    Applied symmetrically to both prediction and target before computing the
    PC prediction error, to keep the loss and its gradient bounded regardless
    of state norms (which can reach hundreds after many backbone applications).

    Args:
        x:   [..., d] — any tensor; normalization is along the last dimension.
        eps: Small floor to prevent division by zero.

    Returns:
        Tensor with same shape as x, RMS ≈ 1 along the last dimension.
    """
    return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps)


class PredictionNetwork(nn.Module):
    """Top-down prediction: level l+1 predicts level l's state.

    Architecture: 2-layer MLP with GELU activation.
        d_{l+1} → 2*d_l → d_l

    The hidden dim is 2×d_l to provide nonlinear capacity while keeping
    the projection small relative to d_l.
    """

    def __init__(self, dim_upper: int, dim_lower: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_upper, 2 * dim_lower, bias=True),
            nn.GELU(),
            nn.Linear(2 * dim_lower, dim_lower, bias=True),
        )
        # Small random init on output layer; near-zero but not dead
        nn.init.normal_(self.net[0].weight, std=0.01)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, std=0.01)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, z_upper: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_upper: [B, L, d_{l+1}]

        Returns:
            mu_lower: [B, L, d_l] — top-down prediction of level l's state.
        """
        return self.net(z_upper)


# ---------------------------------------------------------------------------
# Running-statistics precision (ZERO learnable parameters)
# ---------------------------------------------------------------------------


class RunningPrecision(nn.Module):
    """Per-dimension precision via running EMA of prediction error variance.

    v4.2 replacement for the learned PrecisionNetwork.  Precision is
    estimated as the reciprocal of the exponential-moving-average variance
    of the prediction error, tracked independently per feature dimension.

    Key properties:
    - Zero learnable parameters (ema_var is a buffer, not a Parameter).
    - Precision is a constant for gradient purposes (update is @no_grad).
    - Initialised to ones → precision ≈ 1/(1+eps) ≈ 0.99 everywhere.
    - High-error dimensions get lower precision (higher variance → lower 1/var).

    Args:
        dim:      Feature dimension d_l.
        momentum: EMA smoothing factor (0.99 → slow update, stable statistics).
        eps:      Minimum variance floor (prevents precision from blowing up).
    """

    def __init__(self, dim: int, momentum: float = 0.99, eps: float = 0.01) -> None:
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        # Buffer: not a learnable parameter; survives model.state_dict() checkpointing.
        self.register_buffer("ema_var", torch.ones(dim))

    @torch.no_grad()
    def update(self, prediction_error: torch.Tensor) -> None:
        """Update per-dim running variance from the current prediction error.

        Called once per segment after the bottom-up pass.  The gradient
        graph is severed by @no_grad so this never enters the backward pass.

        Args:
            prediction_error: [B, L, dim] — current ε = z_lower - μ_lower.
        """
        # Per-dim variance across batch (B) and sequence (L) dims
        var = prediction_error.detach().pow(2).mean(dim=(0, 1))  # [dim]
        self.ema_var.mul_(self.momentum).add_(var * (1.0 - self.momentum))

    @property
    def precision(self) -> torch.Tensor:
        """Per-dim precision: 1 / (ema_var + eps).

        Returns:
            [dim] — never in the computation graph (buffer, no_grad update).
        """
        return 1.0 / (self.ema_var + self.eps)

    def per_head_precision(self, n_heads: int = 8) -> torch.Tensor:
        """Per-attention-head precision by averaging over head dimensions.

        Groups the dim features into n_heads equal chunks, averages the
        EMA variance within each chunk, then returns per-head precision.
        Useful for monitoring codebook health and crystallisation decisions.

        Args:
            n_heads: Number of attention heads (must divide dim evenly).

        Returns:
            [n_heads] precision values.
        """
        head_size = self.dim // n_heads
        head_vars = self.ema_var.view(n_heads, head_size).mean(dim=1)  # [n_heads]
        return 1.0 / (head_vars + self.eps)


# ---------------------------------------------------------------------------
# Error upward projection (bias-free)
# ---------------------------------------------------------------------------


class ErrorUpProjection(nn.Module):
    """Projects precision-weighted error from level l up to level l+1's space.

    Bias-free: zero error maps to zero update. This is important for the
    gradient flow — level l+1 should only receive signal when there is
    genuine prediction error.
    """

    def __init__(self, dim_lower: int, dim_upper: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_lower, dim_upper, bias=False)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xi: [B, L, d_l] — precision-weighted error.

        Returns:
            xi_up: [B, L, d_{l+1}] — projected error in upper level's space.
        """
        return self.proj(xi)


# ---------------------------------------------------------------------------
# Predictive coding module: combines all components for one level pair
# ---------------------------------------------------------------------------


class PredictiveCodingModule(nn.Module):
    """Precision-weighted predictive coding for one adjacent level pair (l, l+1).

    Computes:
        μ_l  = prediction_net(z_{l+1})             # top-down prediction
        ε_l  = z_l - μ_l                            # prediction error
        π_l  = running_precision.precision          # [dim] — EMA-based constant
        ξ_l  = π_l ⊙ ε_l                            # precision-weighted error
        ξ_up = error_up_proj(ξ_l)                   # projected for level l+1

    v4.2: precision is computed from running EMA statistics, not a learned net.
    running_precision.update(ε) is called inside forward() to track variance.

    Args:
        dim_lower: Dimension d_l of the lower (faster) level.
        dim_upper: Dimension d_{l+1} of the upper (slower) level.
        eps_min:   Minimum variance floor (prevents precision explosion).
        momentum:  EMA momentum for running precision (default 0.99).
    """

    def __init__(
        self,
        dim_lower: int,
        dim_upper: int,
        eps_min: float = 0.01,
        momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.dim_lower = dim_lower
        self.dim_upper = dim_upper

        self.prediction_net = PredictionNetwork(dim_upper, dim_lower)
        self.running_precision = RunningPrecision(dim_lower, momentum=momentum, eps=eps_min)
        self.error_up_proj = ErrorUpProjection(dim_lower, dim_upper)

    def predict(self, z_upper: torch.Tensor) -> torch.Tensor:
        """Compute top-down prediction μ_l from level l+1 state.

        Args:
            z_upper: [B, L, d_{l+1}]

        Returns:
            mu: [B, L, d_l]
        """
        return self.prediction_net(z_upper)

    def forward(
        self, z_lower: torch.Tensor, z_upper: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute full predictive coding pass for one level pair.

        Option 2 — single normalised pathway: RMSNorm is applied symmetrically
        to both the prediction (mu) and the target (z_lower) before eps, the EMA
        update, and the conditioning signal are computed.  Everything downstream
        operates on the normalised scale, so precision stays bounded near 1.

        Args:
            z_lower: [B, L, d_l]     — lower level state
            z_upper: [B, L, d_{l+1}] — upper level state

        Returns:
            mu:    [B, L, d_l]       — top-down prediction (raw, pre-normalisation)
            eps:   [B, L, d_l]       — normalised prediction error
                                       rms_normalize(z_lower) - rms_normalize(mu)
            pi:    [d_l]             — precision vector (constant, not in grad graph)
            xi:    [B, L, d_l]       — precision-weighted normalised error
            xi_up: [B, L, d_{l+1}]  — projected error for upper level input
        """
        mu = self.prediction_net(z_upper)

        # Symmetric RMSNorm on both sides before any further computation.
        # This makes the error scale-invariant: prediction errors start bounded
        # (RMS ≈ 1) regardless of state magnitude, so the EMA never explodes and
        # precision never collapses.  The Jacobian d(rms_normalize(mu))/d(theta)
        # ∝ 1/rms(mu) amplifies gradients for the near-zero prediction-net init,
        # rescuing the dying-gradient trap identified in Session N1.
        z_lower_norm = rms_normalize(z_lower)
        mu_norm = rms_normalize(mu)
        eps = z_lower_norm - mu_norm   # [B, L, d_l], RMS ≈ 1 at init

        # Update EMA with normalised error (no_grad — precision is a constant).
        # ema_var now tracks normalised variance, which stays bounded near 1.0;
        # precision/level{i}_* diagnostics reflect a health indicator, not raw scale.
        self.running_precision.update(eps)
        pi = self.running_precision.precision  # [dim_lower], not in grad graph

        xi = pi * eps       # broadcasts [dim_lower] over [B, L, dim_lower]
        xi_up = self.error_up_proj(xi)
        return mu, eps, pi, xi, xi_up


# ---------------------------------------------------------------------------
# Loss functions for predictive coding
# ---------------------------------------------------------------------------


def precision_weighted_prediction_loss(
    eps: torch.Tensor, pi: torch.Tensor
) -> torch.Tensor:
    """Precision-weighted squared prediction error.

    L_pred = mean(π.detach() ⊙ ε²)

    π is treated as a constant (already detached via RunningPrecision).
    Encourages accurate predictions; high-precision dimensions penalised
    more for errors.

    Args:
        eps: [B, L, d_l] — prediction error.
        pi:  [d_l] or [B, L, d_l] — precision vector (constant, detached).

    Returns:
        Scalar loss.
    """
    return (pi.detach() * eps.pow(2)).mean(dim=-1).mean()
