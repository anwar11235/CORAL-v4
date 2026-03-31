"""Integration tests for CoralCore mode="full" (Session 5).

Verification criteria:
  - Forward pass in mode="full" with use_crystallisation=True completes without error
  - Output shapes are unchanged compared to mode="pc_only"
  - crystal_stats are present in output when use_crystallisation=True
  - crystal_stats list is empty when use_crystallisation=False (mode="pc_only")
  - Backward pass works in mode="full" (backbone gradients survive crystallisation enforce)
  - commitment_loss and disentanglement_loss appear in loss breakdown
"""

import pytest
import torch

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.training.losses import CoralLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_full_config(**overrides):
    """Config with mode='full' and crystallisation enabled."""
    defaults = dict(
        n_levels=2,
        level_dims=[64, 32],
        backbone_dim=64,
        n_heads=4,
        d_k=16,
        ffn_expansion=2,
        timescale_base=2,
        K_max=2,
        use_predictive_coding=True,
        use_crystallisation=True,
        epsilon_min=0.01,
        lambda_pred=0.001,
        lambda_pi=0.0,
        lambda_commit=0.25,
        lambda_dis=0.01,
        codebook_heads=4,
        codebook_entries_per_head=8,
        tau_converge=0.01,
        tau_decrystallise=0.05,
        n_stable=2,
        precision_momentum=0.99,
        vocab_size=10,
        mode="full",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_pconly_config():
    """Config for mode='pc_only' (no crystallisation)."""
    return ModelConfig(
        n_levels=2,
        level_dims=[64, 32],
        backbone_dim=64,
        n_heads=4,
        d_k=16,
        ffn_expansion=2,
        timescale_base=2,
        K_max=2,
        use_predictive_coding=True,
        epsilon_min=0.01,
        lambda_pred=0.001,
        vocab_size=10,
        # mode defaults to "baseline" but use_predictive_coding=True promotes to pc_only
    )


def _make_grid_adapter(config, H=3, W=3):
    return GridAdapter(CoralConfig(model=config), vocab_size=config.vocab_size,
                       grid_height=H, grid_width=W)


# ---------------------------------------------------------------------------
# Test 1: Forward pass completes without error
# ---------------------------------------------------------------------------

def test_full_mode_forward_completes():
    """mode='full' with use_crystallisation=True runs without error."""
    config = _make_full_config()
    adapter = _make_grid_adapter(config)
    core = CoralCore(config)

    inputs = torch.randint(0, 10, (2, 9))  # 3×3 grid
    z1 = adapter.encode(inputs)
    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode)

    assert out.num_segments >= 1
    assert len(out.all_logits) == out.num_segments


# ---------------------------------------------------------------------------
# Test 2: Output shapes unchanged
# ---------------------------------------------------------------------------

def test_full_mode_output_shapes_match_pc_only():
    """Output shapes from mode='full' must equal those from mode='pc_only'."""
    B, H, W = 2, 3, 3
    L = H * W

    config_full = _make_full_config()
    config_pc = _make_pconly_config()

    adapter_full = _make_grid_adapter(config_full, H, W)
    adapter_pc = _make_grid_adapter(config_pc, H, W)
    core_full = CoralCore(config_full)
    core_pc = CoralCore(config_pc)

    inputs = torch.randint(0, 10, (B, L))

    z_full = adapter_full.encode(inputs)
    z_pc = adapter_pc.encode(inputs)

    out_full = core_full(z_full, K_max=2, training=True, decode_fn=adapter_full.decode)
    out_pc = core_pc(z_pc, K_max=2, training=True, decode_fn=adapter_pc.decode)

    # z_states shapes must match
    assert out_full.z_states[0].shape == out_pc.z_states[0].shape
    assert out_full.z_states[1].shape == out_pc.z_states[1].shape

    # Logit shapes must match
    assert out_full.all_logits[0].shape == out_pc.all_logits[0].shape


# ---------------------------------------------------------------------------
# Test 3: crystal_stats present when use_crystallisation=True
# ---------------------------------------------------------------------------

def test_crystal_stats_present_when_enabled():
    """output.crystal_stats should be populated when use_crystallisation=True."""
    config = _make_full_config()
    adapter = _make_grid_adapter(config)
    core = CoralCore(config)

    inputs = torch.randint(0, 10, (2, 9))
    z1 = adapter.encode(inputs)
    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode)

    assert len(out.crystal_stats) == out.num_segments, (
        f"Expected {out.num_segments} crystal_stats entries, got {len(out.crystal_stats)}"
    )
    # At least the first segment should have stats (even if empty dict on seg 0)
    assert isinstance(out.crystal_stats[0], dict)


# ---------------------------------------------------------------------------
# Test 4: crystal_stats absent when use_crystallisation=False
# ---------------------------------------------------------------------------

def test_crystal_stats_empty_when_disabled():
    """output.crystal_stats should be empty dicts when use_crystallisation=False."""
    config = _make_pconly_config()
    adapter = _make_grid_adapter(config)
    core = CoralCore(config)

    inputs = torch.randint(0, 10, (2, 9))
    z1 = adapter.encode(inputs)
    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode)

    # crystal_stats should exist but contain only empty dicts (no crystallisation)
    for stats in out.crystal_stats:
        assert stats == {}, f"Expected empty stats dict, got {stats}"

    # No commit or dis losses
    assert len(out.commit_losses) == 0
    assert len(out.dis_losses) == 0


# ---------------------------------------------------------------------------
# Test 5: Backward pass works — backbone gets gradients despite crystallisation
# ---------------------------------------------------------------------------

def test_full_mode_backward_completes_with_gradients():
    """Backbone parameters must have non-zero gradients after backward in mode='full'."""
    config = _make_full_config()
    adapter = _make_grid_adapter(config)
    core = CoralCore(config)
    loss_fn = CoralLoss(config)

    inputs = torch.randint(0, 10, (2, 9))
    labels = torch.randint(1, 10, (2, 9))
    z1 = adapter.encode(inputs)

    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode)

    total_loss = torch.tensor(0.0)
    for i, logits in enumerate(out.all_logits):
        seg_loss, _ = loss_fn(
            logits=logits,
            labels=labels,
            pred_errors=out.pred_errors[i] if out.pred_errors else None,
            precisions=out.precisions[i] if out.precisions else None,
            commitment_loss=out.commit_losses[i] if out.commit_losses else None,
            disentanglement_loss=out.dis_losses[i] if out.dis_losses else None,
        )
        total_loss = total_loss + seg_loss

    total_loss.backward()

    for name, param in core.backbone.named_parameters():
        assert param.grad is not None, f"No gradient for backbone.{name}"
        assert param.grad.abs().sum().item() > 0, f"Zero gradient for backbone.{name}"


# ---------------------------------------------------------------------------
# Test 6: Crystallisation losses appear in loss breakdown
# ---------------------------------------------------------------------------

def test_crystallisation_losses_in_breakdown():
    """commitment_loss and disentanglement_loss must appear in loss breakdown."""
    config = _make_full_config(lambda_commit=0.25, lambda_dis=0.01)
    adapter = _make_grid_adapter(config)
    core = CoralCore(config)
    loss_fn = CoralLoss(config)

    inputs = torch.randint(0, 10, (2, 9))
    labels = torch.randint(1, 10, (2, 9))
    z1 = adapter.encode(inputs)

    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode)

    assert out.commit_losses, "Expected commit_losses to be populated in mode='full'"

    # Use the first segment's crystallisation losses
    _, breakdown = loss_fn(
        logits=out.all_logits[0],
        labels=labels,
        commitment_loss=out.commit_losses[0],
        disentanglement_loss=out.dis_losses[0],
    )

    assert "loss/commitment" in breakdown
    assert "loss/crystallisation" in breakdown
    assert not torch.isnan(breakdown["loss/commitment"]), "commitment loss is NaN"
    assert not torch.isnan(breakdown["loss/crystallisation"]), "disentanglement loss is NaN"
    # Losses should be non-negative
    assert breakdown["loss/commitment"].item() >= 0
    assert breakdown["loss/crystallisation"].item() >= 0


# ---------------------------------------------------------------------------
# Test 7: Crystallisation manager is instantiated iff use_crystallisation=True
# ---------------------------------------------------------------------------

def test_crystallisation_manager_gated_by_config():
    """CrystallisationManager exists iff use_crystallisation=True."""
    config_on = _make_full_config(use_crystallisation=True)
    config_off = _make_full_config(use_crystallisation=False)

    core_on = CoralCore(config_on)
    core_off = CoralCore(config_off)

    assert core_on.crystallisation_manager is not None
    assert core_off.crystallisation_manager is None


# ---------------------------------------------------------------------------
# Test 8: No NaN gradients in mode="full"
# ---------------------------------------------------------------------------

def test_full_mode_no_nan_gradients():
    """No NaN or Inf gradients should appear after backward in mode='full'."""
    config = _make_full_config()
    adapter = _make_grid_adapter(config)
    core = CoralCore(config)
    loss_fn = CoralLoss(config)

    inputs = torch.randint(0, 10, (2, 9))
    labels = torch.randint(1, 10, (2, 9))
    z1 = adapter.encode(inputs)

    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode)

    total_loss = torch.tensor(0.0)
    for i, logits in enumerate(out.all_logits):
        seg_loss, _ = loss_fn(
            logits=logits, labels=labels,
            commitment_loss=out.commit_losses[i] if out.commit_losses else None,
            disentanglement_loss=out.dis_losses[i] if out.dis_losses else None,
        )
        total_loss = total_loss + seg_loss

    total_loss.backward()

    for name, param in list(adapter.named_parameters()) + list(core.named_parameters()):
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
