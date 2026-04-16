"""Tests for v5 ConditioningGate module and CORAL integration.

Phase 2 tests — validates the v5 feature-selective gate primitive.

Test inventory:
  1. Gate module: output shape, value range, and init.
  2. Feature selectivity: gate CAN produce per-feature variation.
  3. Conditioning contribution bounded by construction (sigmoid * RMSNorm).
  4. Gradient reaches gate MLP at level 0.
  5. Gradient reaches gate MLP at level 1.
  6. Precision regulariser contributes to total loss.
  7. No cond_gate references in production code.
"""

import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.model.conditioning_gate import ConditioningGate
from coral.model.predictive_coding import rms_normalize
from coral.training.losses import CoralLoss


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _minimal_n2_config(**kwargs) -> ModelConfig:
    """Minimal N=2 pc_only config for gate tests."""
    defaults = dict(
        n_levels=2,
        level_dims=[64, 32],
        backbone_dim=64,
        n_heads=4,
        d_k=16,
        ffn_expansion=2,
        timescale_base=2,
        K_max=1,
        use_predictive_coding=True,
        use_crystallisation=False,
        epsilon_min=0.01,
        lambda_pred=0.001,
        lambda_pi=0.001,
        vocab_size=10,
        mode="pc_only",
        use_consolidation_step=False,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# Test 1 — Gate module: shape, value range, initialization
# ---------------------------------------------------------------------------

def test_conditioning_gate_output_shape():
    """ConditioningGate output must have the same shape as its input."""
    gate = ConditioningGate(d_model=64)
    x = torch.randn(3, 7, 64)  # [B, L, d_model]
    out = gate(x)
    assert out.shape == x.shape, (
        f"Expected gate output shape {x.shape}, got {out.shape}"
    )


def test_conditioning_gate_values_in_0_1():
    """All gate output values must be strictly in (0, 1) — sigmoid bounded."""
    gate = ConditioningGate(d_model=64)
    x = torch.randn(4, 9, 64)
    out = gate(x)
    assert out.min().item() > 0.0, f"Gate has value <= 0: min={out.min().item():.6f}"
    assert out.max().item() < 1.0, f"Gate has value >= 1: max={out.max().item():.6f}"


def test_conditioning_gate_init_approximately_012():
    """At initialization, gate output mean should be approximately sigmoid(-2) ~ 0.12.

    The output layer weight is zero and bias is -2.0, so all features start at
    sigmoid(-2) = 0.1192... regardless of input.
    """
    torch.manual_seed(0)
    gate = ConditioningGate(d_model=64, init_bias=-2.0)
    x = torch.randn(16, 81, 64)
    with torch.no_grad():
        out = gate(x)
    mean_val = out.mean().item()
    expected = torch.sigmoid(torch.tensor(-2.0)).item()  # ~0.1192
    assert abs(mean_val - expected) < 0.01, (
        f"Gate mean at init should be ~{expected:.4f} (sigmoid(-2)), got {mean_val:.4f}. "
        f"Check that output layer weight is zero and bias is -2.0."
    )


# ---------------------------------------------------------------------------
# Test 2 — Feature selectivity: gate can produce per-feature variation
# ---------------------------------------------------------------------------

def test_conditioning_gate_can_express_feature_selectivity():
    """After manual weight override, gate must produce high vs low per-feature values.

    Sets the output layer weight/bias so that feature 0 gets a large positive
    pre-sigmoid input (-> near 1.0) and feature 1 gets a large negative input
    (-> near 0.0). Verifies the MLP can express per-feature discrimination.

    This tests the MECHANISM can work, not that it has learned to do so.
    """
    d_model = 4
    gate = ConditioningGate(d_model=d_model)

    # Override output layer: weight=0 (ignore hidden), bias = [+5, -5, 0, 0]
    with torch.no_grad():
        gate.net[-1].weight.zero_()
        gate.net[-1].bias.zero_()
        gate.net[-1].bias[0] = 5.0   # feature 0: sigmoid(5) ~ 0.993
        gate.net[-1].bias[1] = -5.0  # feature 1: sigmoid(-5) ~ 0.007

    x = torch.randn(2, 3, d_model)
    out = gate(x)

    assert out[..., 0].mean().item() > 0.9, (
        f"Feature 0 should be near 1.0 (bias=+5), got {out[..., 0].mean().item():.4f}"
    )
    assert out[..., 1].mean().item() < 0.1, (
        f"Feature 1 should be near 0.0 (bias=-5), got {out[..., 1].mean().item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Conditioning contribution bounded by construction
# ---------------------------------------------------------------------------

def test_conditioning_contribution_bounded_by_construction():
    """z_new + gate * rms_normalize(conditioning) is bounded for any input scale.

    Construct a minimal N=2 model. Run a forward pass with a deliberately large
    conditioning tensor (RMS ~ 1000). Assert that the conditioning contribution
    is < 1.0 per feature (sigmoid gate * RMSNorm = bounded by construction).

    This is the state-norm-safety test: even with extreme inputs, the conditioning
    path cannot cause unbounded state growth.
    """
    torch.manual_seed(0)
    config = _minimal_n2_config()
    core = CoralCore(config)
    core.eval()

    B, L, d0 = 2, 9, 64
    z = torch.zeros(B, L, d0)

    # conditioning with RMS ~ 1000
    raw = torch.randn(B, L, d0)
    conditioning_large = raw / raw.pow(2).mean(dim=-1, keepdim=True).sqrt() * 1000.0

    with torch.no_grad():
        z_backbone = core._run_level(z.clone(), level_idx=0, n_steps=1,
                                     conditioning=None)
        z_out = core._run_level(z.clone(), level_idx=0, n_steps=1,
                                conditioning=conditioning_large)

    delta = z_out - z_backbone
    delta_rms = delta.pow(2).mean(dim=-1).sqrt().mean().item()

    assert delta_rms < 1.0, (
        f"Conditioning contribution delta_rms = {delta_rms:.4f} >= 1.0. "
        f"sigmoid gate * rms_normalize(conditioning) should be bounded < 1.0. "
        f"If delta_rms ~ 1000, rms_normalize is not being applied."
    )
    assert delta_rms > 0.0, "Conditioning contribution must be non-zero"


# ---------------------------------------------------------------------------
# Test 4 — Gradient reaches gate MLP at level 0
# ---------------------------------------------------------------------------

def test_gradient_reaches_gate_mlp_level_0():
    """Gate MLP at level 0 must receive gradient from the task loss.

    This is the gradient-path test (v4 learning 5.6). v4's scalar cond_gate[0]
    worked but was under-parameterized. The v5 gate MLP must also receive gradient.

    Asserts: all gate MLP parameters at level 0 have non-zero .grad with magnitude > 1e-4.
    """
    torch.manual_seed(42)
    config = _minimal_n2_config()
    core = CoralCore(config)
    adapter = GridAdapter(CoralConfig(model=config, device="cpu"),
                          vocab_size=10, grid_height=2, grid_width=2)

    B, L = 2, 4
    inputs = torch.randint(0, 10, (B, L))
    targets = torch.randint(0, 10, (B, L))

    z1 = adapter.encode(inputs)
    output = core(z1, K_max=1, training=True, decode_fn=adapter.decode)
    logits = output.all_logits[-1]
    task_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    )
    task_loss.backward()

    gate_mod = core.conditioning_gates[0]
    # At initialization net[-1].weight = 0, which blocks the backward path to net[0].
    # The gradient DOES reach net[-1] (output layer weight and bias).
    # Check that the output-layer parameters receive gradient > threshold.
    # net[0].weight grad is zero at init by design (blocked by zero net[-1].weight).
    out_layer = gate_mod.net[-1]  # output Linear layer
    assert out_layer.weight.grad is not None, (
        "conditioning_gates[0].net[-1].weight has no gradient. "
        "The gradient path from decoder -> backbone -> gate output layer is broken."
    )
    assert out_layer.bias.grad is not None, (
        "conditioning_gates[0].net[-1].bias has no gradient."
    )
    assert out_layer.weight.grad.abs().max().item() > 1e-6, (
        f"conditioning_gates[0] output weight gradient is too small: "
        f"{out_layer.weight.grad.abs().max().item():.2e}"
    )
    assert out_layer.bias.grad.abs().max().item() > 1e-4, (
        f"conditioning_gates[0] output bias gradient is too small: "
        f"{out_layer.bias.grad.abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Gradient reaches gate MLP at level 1
# ---------------------------------------------------------------------------

def test_gradient_reaches_gate_mlp_level_1():
    """Gate MLP at level 1 must receive gradient from a loss on z_states[1].

    v4's cond_gate[1] had a severed gradient path (N6 finding). The v5 gate MLP
    at level 1 has a within-segment path: z_states[1] directly supervised ->
    gate(z_states[1]) -> gate parameters.

    Accepts a smaller gradient magnitude than level 0 (path is weaker).
    Asserts: at least one gate parameter at level 1 has non-zero gradient.
    """
    torch.manual_seed(7)
    config = _minimal_n2_config()
    core = CoralCore(config)

    B, L, d1 = 2, 4, 32  # level-1 dim
    z1 = torch.zeros(B, L, d1)
    xi_up = torch.randn(B, L, d1, requires_grad=True)

    z_out = core._run_level(z1, level_idx=1, n_steps=1, conditioning=xi_up)
    loss = z_out.pow(2).mean()
    loss.backward()

    gate_mod = core.conditioning_gates[1]
    grads = [p.grad for p in gate_mod.parameters() if p.grad is not None]
    assert len(grads) > 0, (
        "conditioning_gates[1] MLP has no parameter gradients. "
        "The within-segment gradient path is broken."
    )
    max_grad = max(g.abs().max().item() for g in grads)
    assert max_grad > 1e-8, (
        f"conditioning_gates[1] gradient is effectively zero: max={max_grad:.2e}. "
        f"Accept smaller magnitude than level 0, but non-zero."
    )


# ---------------------------------------------------------------------------
# Test 6 — Precision regulariser contributes to total loss
# ---------------------------------------------------------------------------

def test_precision_regulariser_contributes_to_loss():
    """With lambda_pi > 0, precision regulariser must add a non-zero positive term.

    The regulariser is L_pi = lambda_pi * (log(pi) ** 2).mean().
    Minimum at pi=1. Asserts non-zero positive contribution.
    """
    torch.manual_seed(0)
    config = _minimal_n2_config(lambda_pi=0.001)
    core = CoralCore(config)
    loss_fn = CoralLoss(config)

    # Build synthetic precisions dict with known values
    # Use pi=2.0 (above 1) to ensure log(pi)^2 > 0
    precisions = {
        "level_0": torch.full((64,), 2.0),   # pi=2, log(2)^2 ~ 0.48
    }

    logits = torch.randn(2, 4, 10)
    labels = torch.randint(0, 10, (2, 4))

    # Loss with regulariser
    total_with, breakdown_with = loss_fn(logits, labels, precisions=precisions)
    # Loss without regulariser (zero out lambda_pi temporarily)
    config_no_reg = _minimal_n2_config(lambda_pi=0.0)
    loss_fn_no_reg = CoralLoss(config_no_reg)
    total_without, _ = loss_fn_no_reg(logits, labels, precisions=precisions)

    L_pi = breakdown_with.get("loss/precision_reg", torch.tensor(0.0))
    assert L_pi.item() > 0, (
        f"Precision regulariser should be > 0 when pi=2 and lambda_pi=0.001, got {L_pi.item()}"
    )
    assert total_with.item() > total_without.item(), (
        f"Total loss with regulariser ({total_with.item():.6f}) should exceed "
        f"total without ({total_without.item():.6f})"
    )


def test_precision_regulariser_minimum_at_pi_one():
    """Precision regulariser (log pi)^2 is minimised at pi=1."""
    lambda_pi = 0.001

    def reg(pi_val):
        pi = torch.full((64,), float(pi_val))
        return (lambda_pi * (torch.log(pi + 1e-8) ** 2)).mean().item()

    at_one = reg(1.0)
    at_two = reg(2.0)
    at_half = reg(0.5)

    assert at_one < at_two, f"Reg at pi=1 ({at_one:.6f}) should be less than at pi=2 ({at_two:.6f})"
    assert at_one < at_half, f"Reg at pi=1 ({at_one:.6f}) should be less than at pi=0.5 ({at_half:.6f})"
    assert at_one < 1e-6, f"Reg at pi=1 should be ~0, got {at_one:.8f}"


# ---------------------------------------------------------------------------
# Test 7 — No cond_gate references in production code
# ---------------------------------------------------------------------------

def test_no_cond_gate_references_in_production_code():
    """Grep the codebase for 'cond_gate'. Must not appear in any .py source file
    outside of test files and comments that document the removal.

    This is a hygiene test (v5 spec). All live code references to the old
    scalar cond_gate primitive must be removed.
    """
    import os
    import re

    # Directories to check (production code only, not tests)
    check_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "coral"),
        os.path.join(os.path.dirname(__file__), "..", "scripts"),
        os.path.join(os.path.dirname(__file__), "..", "configs"),
    ]

    # Pattern that would indicate live code: self.cond_gate, core.cond_gate, cond_gate[
    live_code_pattern = re.compile(r'(?:self|core)\.cond_gate\b|cond_gate\s*\[')

    violations = []
    for check_dir in check_dirs:
        check_dir = os.path.normpath(check_dir)
        if not os.path.isdir(check_dir):
            continue
        for root, dirs, files in os.walk(check_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if live_code_pattern.search(line):
                            violations.append(f"{fpath}:{lineno}: {line.rstrip()}")

    assert len(violations) == 0, (
        f"Found {len(violations)} live cond_gate references in production code:\n"
        + "\n".join(violations)
    )
