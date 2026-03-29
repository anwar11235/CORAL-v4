"""CORAL v4 — Precision-weighted predictive coding.

Implements the core free energy mechanism:
  1. PredictionNetwork: top-down prediction μ_l = f(z_{l+1})
  2. PrecisionNetwork: per-dimension precision π_l = g(z_l)
  3. PredictiveCodingModule: combines prediction, error, and precision

CRITICAL: precision regulariser uses the symmetric log-normal prior:
    L_π = (λ_π / 2) * (log π)²
This has its minimum at π=1. The naive -½ log π causes unbounded precision
growth and is NOT used here.

Runtime assertion: assert pi.mean() < 100  (catches sign bugs early)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from coral.config import ModelConfig


# ---------------------------------------------------------------------------
# Prediction network: top-down prediction of level l from level l+1
# ---------------------------------------------------------------------------


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
        # Initialise to near-zero predictions to avoid large initial errors
        nn.init.normal_(self.net[0].weight, std=0.01)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].weight)
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
# Precision network: per-dimension precision vector
# ---------------------------------------------------------------------------


class PrecisionNetwork(nn.Module):
    """Produces per-dimension precision vector π_l.

    Architecture: 2-layer MLP with GELU.
        d_l → d_l → d_l, then softplus + eps_min

    Output is always positive (softplus) and bounded away from zero (eps_min).
    """

    def __init__(self, dim: int, eps_min: float = 0.01) -> None:
        super().__init__()
        self.eps_min = eps_min
        self.net = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.GELU(),
            nn.Linear(dim, dim, bias=True),
        )
        # Initialise to produce precision ≈ 1 initially (log softplus(0) ≈ 0.69 → +eps)
        nn.init.zeros_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, L, d_l]

        Returns:
            pi: [B, L, d_l] — precision vector, values > eps_min.
        """
        return F.softplus(self.net(z)) + self.eps_min


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
        μ_l  = prediction_net(z_{l+1})           # top-down prediction
        ε_l  = z_l - μ_l                          # prediction error
        π_l  = precision_net(z_l)                 # precision vector
        ξ_l  = π_l ⊙ ε_l                          # precision-weighted error
        ξ_up = error_up_proj(ξ_l)                 # projected for level l+1

    Args:
        dim_lower: Dimension d_l of the lower (faster) level.
        dim_upper: Dimension d_{l+1} of the upper (slower) level.
        eps_min: Minimum precision floor.
    """

    def __init__(self, dim_lower: int, dim_upper: int, eps_min: float = 0.01) -> None:
        super().__init__()
        self.dim_lower = dim_lower
        self.dim_upper = dim_upper

        self.prediction_net = PredictionNetwork(dim_upper, dim_lower)
        self.precision_net = PrecisionNetwork(dim_lower, eps_min)
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

        Args:
            z_lower: [B, L, d_l]   — lower level state
            z_upper: [B, L, d_{l+1}] — upper level state

        Returns:
            mu:    [B, L, d_l]       — top-down prediction
            eps:   [B, L, d_l]       — prediction error (z_lower - mu)
            pi:    [B, L, d_l]       — precision vector (>0)
            xi:    [B, L, d_l]       — precision-weighted error
            xi_up: [B, L, d_{l+1}]   — projected error for upper level input
        """
        mu = self.prediction_net(z_upper)
        eps = z_lower - mu
        pi = self.precision_net(z_lower)

        # Runtime guard against precision explosion (catches wrong regulariser sign)
        assert pi.mean().item() < 100, (
            f"Precision explosion detected: pi.mean()={pi.mean().item():.2f}. "
            "Check the precision regulariser sign in the loss function."
        )

        xi = pi * eps
        xi_up = self.error_up_proj(xi)
        return mu, eps, pi, xi, xi_up


# ---------------------------------------------------------------------------
# Loss functions for predictive coding
# ---------------------------------------------------------------------------


def precision_weighted_prediction_loss(
    eps: torch.Tensor, pi: torch.Tensor
) -> torch.Tensor:
    """Precision-weighted squared prediction error.

    L_pred = ½ * Σ π_l ⊙ ε_l²

    Encourages accurate predictions; high-precision dimensions are penalised
    more for errors.

    Args:
        eps: [B, L, d_l] — prediction error.
        pi:  [B, L, d_l] — precision vector.

    Returns:
        Scalar loss.
    """
    return 0.5 * (pi * eps.pow(2)).sum(dim=-1).mean()


def precision_regulariser(pi: torch.Tensor) -> torch.Tensor:
    """Symmetric log-normal precision regulariser.

    L_π = ½ * (log π)²

    Minimum at π=1. Penalises both under-precision (π→0) and over-precision
    (π→∞) symmetrically in log space. This is the CORRECT regulariser.

    Do NOT use -½ log π (naive), which causes unbounded precision growth.

    Args:
        pi: [B, L, d_l] — precision vector.

    Returns:
        Scalar loss.
    """
    return 0.5 * torch.log(pi).pow(2).sum(dim=-1).mean()
