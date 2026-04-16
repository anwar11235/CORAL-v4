"""Tests for Session N4: RMSNorm conditioning signal + NaN/Inf guard.

Updated for v5: scalar cond_gate replaced by ConditioningGate MLP.
RMSNorm at conditioning site is kept (N4). Precision removed from path (Phase 1).

Verification criteria:
  - rms_normalize(μ) at conditioning site: scaling μ by 1000 produces the
    same z_new as the unscaled version (scale invariance).
  - Conditioning contribution to z_new is bounded by ~1.0 per feature
    (sigmoid gate in (0,1) × rms_normalize output with RMS=1).
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

    The v5 level-0 conditioning injection is:
        gate = conditioning_gates[0](z_new)        # per-feature sigmoid in (0,1)
        z_new = z_new + gate * rms_normalize(conditioning)

    rms_normalize(k × x) == rms_normalize(x) for any scalar k ≠ 0, so
    conditioning_large = 1000 × conditioning_unit must produce the same
    z_new as conditioning_unit (gate is computed from z_new, which is the same
    for both since backbone_in does not include conditioning).
    """
    torch.manual_seed(42)
    config = _pc_n2_config()
    core = CoralCore(config)
    core.eval()

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


def test_conditioning_contribution_bounded():
    """Conditioning contribution to z_new is bounded by ~1.0 per feature (by construction).

    v5 primitive: z_new = z_new + gate * rms_normalize(conditioning)
    - gate = sigmoid(MLP(z_new)) in (0, 1) — bounded by construction
    - rms_normalize(conditioning) has RMS ≈ 1.0 regardless of input scale

    So delta = z_out - z_backbone satisfies delta_rms < 1.0 for any input magnitude.
    With the at-init gate (~0.12), delta_rms ≈ 0.12. Even after max-gate training,
    delta_rms < 1.0 by construction (sigmoid max < 1.0).

    This test uses a conditioning tensor with RMS ≈ 1000 to confirm scale invariance
    AND that the contribution is bounded well below 1000.
    """
    torch.manual_seed(0)
    config = _pc_n2_config()
    core = CoralCore(config)
    core.eval()

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

    # gate in (0,1) * rms_normalize with RMS=1 => contribution << 1.0
    # At init: gate ~ sigmoid(-2) ~ 0.12, so delta_rms ~ 0.12 (not 1000).
    assert delta_rms < 1.0, (
        f"Conditioning contribution RMS should be < 1.0 (sigmoid gate * RMSNorm = bounded), "
        f"got {delta_rms:.4f}. If this is near 1000, RMSNorm is not being applied."
    )
    assert delta_rms > 0.0, "Conditioning contribution must be non-zero (gate is not exactly 0)"


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
