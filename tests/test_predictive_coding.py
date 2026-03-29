"""Unit tests for predictive coding module.

Verification criteria:
  - Prediction error is zero when prediction equals actual state
  - Precision regulariser has minimum at pi=1
  - Precision explosion does NOT occur (1000 steps, pi_mean < 10)
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from coral.model.predictive_coding import (
    PredictionNetwork,
    PrecisionNetwork,
    PredictiveCodingModule,
    precision_regulariser,
    precision_weighted_prediction_loss,
)


def test_prediction_network_shape():
    net = PredictionNetwork(dim_upper=256, dim_lower=512)
    z_upper = torch.randn(4, 81, 256)
    mu = net(z_upper)
    assert mu.shape == (4, 81, 512)


def test_precision_network_shape():
    net = PrecisionNetwork(dim=512, eps_min=0.01)
    z = torch.randn(4, 81, 512)
    pi = net(z)
    assert pi.shape == (4, 81, 512)


def test_precision_network_positive():
    """Precision values must always be positive."""
    net = PrecisionNetwork(dim=128, eps_min=0.01)
    z = torch.randn(10, 20, 128) * 100  # extreme inputs
    pi = net(z)
    assert (pi > 0).all(), "Precision values must be positive"
    assert pi.min().item() >= 0.01 - 1e-6, "Precision must be >= eps_min"


def test_prediction_error_zero_when_accurate():
    """Prediction error should be zero when prediction exactly matches state."""
    pc = PredictiveCodingModule(dim_lower=512, dim_upper=256)
    z_upper = torch.randn(2, 10, 256)
    # Get what the prediction network would produce
    mu = pc.prediction_net(z_upper)
    # Set z_lower = mu (exact match)
    z_lower = mu.detach().clone()

    _, eps, pi, xi, xi_up = pc(z_lower, z_upper)
    assert eps.abs().max().item() < 1e-5, f"Expected ~0 error, got {eps.abs().max().item()}"


def test_precision_regulariser_minimum():
    """Precision regulariser should have its minimum at pi=1."""
    # Create precision tensors
    pi_ones = torch.ones(10, 10, 64)
    pi_small = torch.full((10, 10, 64), 0.1)
    pi_large = torch.full((10, 10, 64), 10.0)

    L_ones = precision_regulariser(pi_ones).item()
    L_small = precision_regulariser(pi_small).item()
    L_large = precision_regulariser(pi_large).item()

    # At pi=1, loss should be ~0
    assert abs(L_ones) < 1e-5, f"Regulariser at pi=1 should be ~0, got {L_ones}"
    # Should be higher for pi != 1
    assert L_small > L_ones, f"Regulariser should increase for pi < 1: {L_small} vs {L_ones}"
    assert L_large > L_ones, f"Regulariser should increase for pi > 1: {L_large} vs {L_ones}"


def test_no_precision_explosion():
    """Train for 1000 steps and verify pi.mean() stays < 10."""
    pc = PredictiveCodingModule(dim_lower=64, dim_upper=32, eps_min=0.01)
    opt = optim.Adam(pc.parameters(), lr=1e-3)

    for step in range(1000):
        z_lower = torch.randn(4, 10, 64)
        z_upper = torch.randn(4, 10, 32)

        mu, eps, pi, xi, xi_up = pc(z_lower, z_upper)

        # Loss with correct regulariser
        L_pred = precision_weighted_prediction_loss(eps, pi)
        L_pi = precision_regulariser(pi) * 0.01

        loss = L_pred + L_pi
        opt.zero_grad()
        loss.backward()
        opt.step()

        pi_mean = pi.mean().item()
        assert pi_mean < 10, (
            f"Precision explosion at step {step}: pi.mean()={pi_mean:.2f}. "
            "Check regulariser sign."
        )


def test_precision_weighted_loss():
    """Precision-weighted loss should be 0 when error is 0."""
    eps = torch.zeros(4, 10, 64)
    pi = torch.ones(4, 10, 64)
    loss = precision_weighted_prediction_loss(eps, pi)
    assert loss.item() < 1e-8


def test_error_up_projection_bias_free():
    """Error up projection should be bias-free (zero error → zero update)."""
    from coral.model.predictive_coding import ErrorUpProjection
    proj = ErrorUpProjection(dim_lower=512, dim_upper=256)
    zero_error = torch.zeros(4, 10, 512)
    out = proj(zero_error)
    assert out.abs().max().item() < 1e-8, "Zero error must map to zero update (bias-free)"
