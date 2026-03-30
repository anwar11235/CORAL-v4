"""Unit tests for predictive coding module (v4.2).

v4.2 changes:
  - PrecisionNetwork removed; replaced by RunningPrecision (zero learnable params)
  - precision_regulariser removed; precision is a running-EMA constant

Verification criteria:
  - Prediction error is zero when prediction exactly matches state
  - RunningPrecision: zero learnable parameters, not in grad graph
  - RunningPrecision: uniform init, differentiates after non-uniform errors
  - RunningPrecision: per_head_precision returns H values
  - Prediction loss computes without NaN when pi is uniform/non-uniform
  - No precision explosion over 1000 steps without regulariser
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from coral.model.predictive_coding import (
    PredictionNetwork,
    RunningPrecision,
    PredictiveCodingModule,
    precision_weighted_prediction_loss,
)


def test_prediction_network_shape():
    net = PredictionNetwork(dim_upper=256, dim_lower=512)
    z_upper = torch.randn(4, 81, 256)
    mu = net(z_upper)
    assert mu.shape == (4, 81, 512)


# ---------------------------------------------------------------------------
# RunningPrecision tests (replaces PrecisionNetwork tests)
# ---------------------------------------------------------------------------

def test_running_precision_zero_params():
    """RunningPrecision must have exactly zero learnable parameters."""
    rp = RunningPrecision(dim=512, momentum=0.99, eps=0.01)
    n_params = sum(p.numel() for p in rp.parameters())
    assert n_params == 0, f"Expected 0 learnable params, got {n_params}"


def test_running_precision_init_uniform():
    """RunningPrecision initialises with ema_var=1 → precision ≈ 1/(1+eps)."""
    eps = 0.01
    rp = RunningPrecision(dim=64, momentum=0.99, eps=eps)
    pi = rp.precision
    expected = 1.0 / (1.0 + eps)
    assert torch.allclose(pi, torch.full_like(pi, expected), atol=1e-6), (
        f"Initial precision should be uniform {expected:.4f}, got {pi}"
    )


def test_running_precision_not_in_grad_graph():
    """Precision tensor must not be in the computation graph (requires_grad=False)."""
    rp = RunningPrecision(dim=64, momentum=0.99, eps=0.01)
    pi = rp.precision
    assert not pi.requires_grad, "precision must NOT require gradients"


def test_running_precision_differentiates_after_updates():
    """After feeding non-uniform error variance, precision values should differ."""
    dim = 64
    rp = RunningPrecision(dim=dim, momentum=0.99, eps=0.01)

    # Feed errors where first half of dims have large variance, second half small
    for _ in range(200):
        error = torch.zeros(4, 10, dim)
        error[:, :, :dim // 2] = torch.randn(4, 10, dim // 2) * 5.0   # large variance
        error[:, :, dim // 2:] = torch.randn(4, 10, dim // 2) * 0.01  # small variance
        rp.update(error)

    pi = rp.precision
    pi_high = pi[:dim // 2]   # should be low precision (high variance)
    pi_low  = pi[dim // 2:]   # should be high precision (low variance)

    assert pi_low.mean() > pi_high.mean(), (
        "Dims with small error variance should get higher precision"
    )
    assert pi.std().item() > 0.01, (
        "Precision should not be uniform after non-uniform errors"
    )


def test_running_precision_per_head():
    """per_head_precision returns H precision values."""
    rp = RunningPrecision(dim=512, momentum=0.99, eps=0.01)
    head_pi = rp.per_head_precision(n_heads=8)
    assert head_pi.shape == (8,), f"Expected (8,), got {head_pi.shape}"


def test_running_precision_positive():
    """Precision values must always be positive."""
    rp = RunningPrecision(dim=64, momentum=0.99, eps=0.01)
    # Feed extreme errors to stress-test
    for _ in range(100):
        rp.update(torch.randn(4, 10, 64) * 1000)
    pi = rp.precision
    assert (pi > 0).all(), f"Precision must be positive, got min={pi.min().item()}"


# ---------------------------------------------------------------------------
# PredictiveCodingModule tests
# ---------------------------------------------------------------------------

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


def test_pc_module_pi_shape():
    """PredictiveCodingModule returns pi with shape [dim_lower] (v4.2 change)."""
    pc = PredictiveCodingModule(dim_lower=64, dim_upper=32)
    z_lower = torch.randn(4, 10, 64)
    z_upper = torch.randn(4, 10, 32)
    mu, eps, pi, xi, xi_up = pc(z_lower, z_upper)
    assert pi.shape == (64,), f"Expected pi shape (64,), got {pi.shape}"
    assert xi.shape == (4, 10, 64)
    assert xi_up.shape == (4, 10, 32)


def test_no_precision_explosion():
    """Train for 1000 steps without precision regulariser — precision should stay bounded."""
    pc = PredictiveCodingModule(dim_lower=64, dim_upper=32, eps_min=0.01)
    opt = optim.Adam(pc.parameters(), lr=1e-3)

    for step in range(1000):
        z_lower = torch.randn(4, 10, 64)
        z_upper = torch.randn(4, 10, 32)

        mu, eps, pi, xi, xi_up = pc(z_lower, z_upper)

        loss = precision_weighted_prediction_loss(eps, pi)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # pi is a [dim] tensor derived from EMA variance — must stay bounded
        pi_max = pi.max().item()
        assert pi_max < 1000, (
            f"Precision explosion at step {step}: pi.max()={pi_max:.2f}"
        )


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

def test_precision_weighted_loss_zero_when_error_zero():
    """Precision-weighted loss should be 0 when error is 0."""
    eps = torch.zeros(4, 10, 64)
    pi = torch.ones(64)
    loss = precision_weighted_prediction_loss(eps, pi)
    assert loss.item() < 1e-8


def test_precision_weighted_loss_no_nan():
    """Prediction loss should not produce NaN with typical inputs."""
    eps = torch.randn(4, 10, 64)
    pi = torch.rand(64) + 0.01  # all positive
    loss = precision_weighted_prediction_loss(eps, pi)
    assert not torch.isnan(loss), "Prediction loss must not be NaN"
    assert not torch.isinf(loss), "Prediction loss must not be Inf"


# ---------------------------------------------------------------------------
# Error up projection test
# ---------------------------------------------------------------------------

def test_error_up_projection_bias_free():
    """Error up projection should be bias-free (zero error → zero update)."""
    from coral.model.predictive_coding import ErrorUpProjection
    proj = ErrorUpProjection(dim_lower=512, dim_upper=256)
    zero_error = torch.zeros(4, 10, 512)
    out = proj(zero_error)
    assert out.abs().max().item() < 1e-8, "Zero error must map to zero update (bias-free)"
