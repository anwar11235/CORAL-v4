"""Tests for Session N4: RMSNorm conditioning signal + NaN/Inf guard.

Verification criteria:
  - rms_normalize(μ) at conditioning site: scaling μ by 1000 produces the
    same z_new as the unscaled version (scale invariance). The conditioning
    contribution has RMS ≈ gate × π × 1.0, not gate × π × 1000.
  - NaN loss raises RuntimeError with "Non-finite loss" before backward.
  - Inf loss raises RuntimeError with "Non-finite loss" before backward.
"""

import pytest
import torch
from unittest.mock import patch

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.training.losses import CoralLoss
from coral.training.trainer import TrainerV4


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pc_n2_config() -> ModelConfig:
    """Minimal N=2 pc_only config."""
    return ModelConfig(
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
        vocab_size=10,
        mode="pc_only",
        use_consolidation_step=False,
    )


def _make_trainer() -> TrainerV4:
    """Minimal single-level baseline trainer for guard tests."""
    config = ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, timescale_base=2, K_max=1,
        use_predictive_coding=False, use_crystallisation=False,
        epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="baseline",
        use_consolidation_step=False,
    )
    coral_config = CoralConfig(model=config, device="cpu")
    coral_config.training.precision = "float32"
    coral_config.training.gradient_clip = 1.0
    adapter = GridAdapter(coral_config, vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(config)
    loss_fn = CoralLoss(config)
    return TrainerV4(adapter, core, loss_fn, coral_config)


def _make_batch():
    return {
        "inputs": torch.randint(0, 10, (2, 9)),
        "labels": torch.randint(1, 10, (2, 9)),
    }


# ---------------------------------------------------------------------------
# Test 1 — RMSNorm conditioning scale invariance
# ---------------------------------------------------------------------------

def test_conditioning_scale_invariant_under_rmsnorm():
    """Scaling μ by 1000 must produce the same z_new (rms_normalize is scale-invariant).

    The level-0 conditioning injection is now:
        z_new = z_backbone + gate × (π × rms_normalize(conditioning))

    rms_normalize(k × x) == rms_normalize(x) for any scalar k ≠ 0, so
    conditioning_large = 1000 × conditioning_unit must produce the same
    z_new as conditioning_unit.  The backbone output z_backbone is identical
    for both calls (backbone_in does not include conditioning).
    """
    torch.manual_seed(42)
    config = _pc_n2_config()
    core = CoralCore(config)
    core.eval()

    # Pin gate=1.0 and pi=1.0 (ema_var=0.99 → pi = 1/(0.99+0.01) = 1.0)
    core.cond_gate[0].data.fill_(1.0)
    core.pc_modules[0].running_precision.ema_var.fill_(0.99)

    B, L, d0 = 1, 4, 64
    z = torch.zeros(B, L, d0)

    conditioning_unit = torch.randn(B, L, d0)
    conditioning_large = conditioning_unit * 1000.0   # same direction, 1000× scale

    with torch.no_grad():
        z_out_unit  = core._run_level(z.clone(), level_idx=0, n_steps=1,
                                      conditioning=conditioning_unit)
        z_out_large = core._run_level(z.clone(), level_idx=0, n_steps=1,
                                      conditioning=conditioning_large)

    # Scale invariance: outputs must be identical up to floating-point precision.
    max_err = (z_out_large - z_out_unit).abs().max().item()
    assert max_err < 1e-4, (
        f"rms_normalize should make conditioning scale-invariant, "
        f"but 1000× scaling produced max diff = {max_err:.6f}"
    )


def test_conditioning_contribution_rms_is_gate_times_pi():
    """The conditioning addition must have RMS ≈ gate × pi × 1.0, not gate × pi × 1000.

    With gate=1.0, pi=1.0:
      z_backbone = _run_level(z, conditioning=None)
      z_out      = _run_level(z, conditioning=large_tensor)
      delta      = z_out - z_backbone  ← the conditioning addition
      rms(delta) ≈ gate × pi × 1.0 = 1.0   (because rms_normalize bounds the signal)

    Before the N4 fix, rms(delta) would be ≈ 1000.0.
    """
    torch.manual_seed(0)
    config = _pc_n2_config()
    core = CoralCore(config)
    core.eval()

    core.cond_gate[0].data.fill_(1.0)
    core.pc_modules[0].running_precision.ema_var.fill_(0.99)  # pi = 1.0

    B, L, d0 = 2, 9, 64
    z = torch.zeros(B, L, d0)

    # conditioning with RMS ≈ 1000 per token
    raw = torch.randn(B, L, d0)
    conditioning_large = raw / raw.pow(2).mean(dim=-1, keepdim=True).sqrt() * 1000.0

    with torch.no_grad():
        z_backbone = core._run_level(z.clone(), level_idx=0, n_steps=1,
                                     conditioning=None)
        z_out      = core._run_level(z.clone(), level_idx=0, n_steps=1,
                                     conditioning=conditioning_large)

    delta = z_out - z_backbone   # purely the conditioning addition
    delta_rms = delta.pow(2).mean(dim=-1).sqrt().mean().item()

    # Expected: gate × pi × rms(rms_normalize(conditioning)) = 1.0 × 1.0 × 1.0 = 1.0
    # Allow 10% tolerance as stated in the prompt.
    expected = 1.0
    assert abs(delta_rms - expected) / expected < 0.10, (
        f"Conditioning contribution RMS should be ≈{expected:.1f} "
        f"(gate=1, pi=1, rms_normalize output RMS=1), got {delta_rms:.4f}. "
        f"If this is near 1000, RMSNorm is not being applied."
    )


# ---------------------------------------------------------------------------
# Test 2 — NaN/Inf guard in train_step
# ---------------------------------------------------------------------------

def test_nan_loss_raises_runtime_error():
    """train_step must raise RuntimeError with 'Non-finite loss' when loss is NaN."""
    trainer = _make_trainer()
    batch = _make_batch()

    nan_loss = torch.tensor(float("nan"))
    with patch.object(trainer.loss_fn, "forward", return_value=(nan_loss, {})):
        with pytest.raises(RuntimeError, match="Non-finite loss"):
            trainer.train_step(batch)


def test_inf_loss_raises_runtime_error():
    """train_step must raise RuntimeError with 'Non-finite loss' when loss is Inf."""
    trainer = _make_trainer()
    batch = _make_batch()

    inf_loss = torch.tensor(float("inf"))
    with patch.object(trainer.loss_fn, "forward", return_value=(inf_loss, {})):
        with pytest.raises(RuntimeError, match="Non-finite loss"):
            trainer.train_step(batch)
