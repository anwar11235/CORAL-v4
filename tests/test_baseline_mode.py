"""Tests for CoralCore in baseline mode (mode='baseline', n_levels=1, no PC).

Verification criteria:
  - Forward pass produces correct output shape [B, 81, 512]
  - Forward + backward completes without error
  - Baseline mode has zero prediction/precision network parameters
    (pc_modules is empty when n_levels=1 and use_predictive_coding=False)
"""

import pytest
import torch

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.training.losses import CoralLoss


@pytest.fixture
def baseline_config():
    """Minimal baseline config matching the phase1_baseline_no_pc experiment."""
    return ModelConfig(
        n_levels=1,
        level_dims=[512],
        backbone_layers=2,
        backbone_dim=512,
        n_heads=8,
        d_k=64,
        ffn_expansion=4,
        timescale_base=3,
        K_max=4,
        use_predictive_coding=False,
        epsilon_min=0.01,
        lambda_pred=0.001,
        lambda_pi=0.001,
        vocab_size=11,
        mode="baseline",
    )


@pytest.fixture
def small_baseline_config():
    """Tiny baseline config for fast CPU tests."""
    return ModelConfig(
        n_levels=1,
        level_dims=[64],
        backbone_layers=2,
        backbone_dim=64,
        n_heads=4,
        d_k=16,
        ffn_expansion=2,
        timescale_base=3,
        K_max=2,
        use_predictive_coding=False,
        epsilon_min=0.01,
        lambda_pred=0.001,
        lambda_pi=0.001,
        vocab_size=11,
        mode="baseline",
    )


# ---------------------------------------------------------------------------
# Shape test
# ---------------------------------------------------------------------------

def test_baseline_output_shape(baseline_config):
    """CoralCore in baseline mode must produce output shape [B, 81, 512]."""
    core = CoralCore(baseline_config)
    B, L, d = 2, 81, 512
    z1 = torch.randn(B, L, d)
    out = core(z1, K_max=2, training=False)

    assert out.z_states[0].shape == (B, L, d), (
        f"Expected [B={B}, L={L}, d={d}], got {out.z_states[0].shape}"
    )
    assert out.num_segments >= 1
    assert len(out.z_states) == 1, "Baseline n_levels=1 should have exactly one level state"


# ---------------------------------------------------------------------------
# Forward + backward
# ---------------------------------------------------------------------------

def test_baseline_forward_backward(small_baseline_config):
    """Baseline mode forward + backward must complete without error."""
    config = small_baseline_config
    core = CoralCore(config)
    # Use a 9x9 = 81 grid; for small tests use a tiny 3x3 grid
    adapter = GridAdapter(
        CoralConfig(model=config), vocab_size=11, grid_height=3, grid_width=3
    )
    loss_fn = CoralLoss(config)

    inputs = torch.randint(0, 11, (2, 9))
    labels = torch.randint(0, 11, (2, 9))

    z1 = adapter.encode(inputs)
    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode)

    assert out.all_logits, "No logits produced in baseline mode"

    total_loss = torch.tensor(0.0)
    for i, logits in enumerate(out.all_logits):
        seg_loss, _ = loss_fn(
            logits=logits,
            labels=labels,
            pred_errors=out.pred_errors[i] if out.pred_errors else None,
            precisions=out.precisions[i] if out.precisions else None,
        )
        total_loss = total_loss + seg_loss

    total_loss.backward()  # Must not raise


# ---------------------------------------------------------------------------
# No PC parameters in computation graph
# ---------------------------------------------------------------------------

def test_baseline_no_pc_params(small_baseline_config):
    """Baseline mode must have zero prediction/precision network parameters.

    With n_levels=1 and use_predictive_coding=False, pc_modules should be
    empty — there are no inter-level prediction or precision networks.
    """
    config = small_baseline_config
    core = CoralCore(config)

    # Check effective mode
    assert core.effective_mode == "baseline", (
        f"Expected effective_mode='baseline', got '{core.effective_mode}'"
    )

    # With n_levels=1 and no PC, pc_modules must be empty
    assert len(core.pc_modules) == 0, (
        f"Expected 0 PC modules in baseline mode, got {len(core.pc_modules)}"
    )

    # Verify no prediction/precision parameters exist at all
    pc_param_count = sum(p.numel() for p in core.pc_modules.parameters())
    assert pc_param_count == 0, (
        f"Expected 0 PC parameters, got {pc_param_count}"
    )
