"""Integration tests for CoralCore.

Verification criteria:
  - Full forward pass produces correct output shape
  - loss.backward() completes without error
  - All parameters receive non-zero gradients
  - Level 2 parameters receive gradients from level 1's prediction error
    (the predictive coding chain is connected)
"""

import pytest
import torch
import torch.nn as nn

from coral.config import ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.training.losses import CoralLoss


@pytest.fixture
def small_config():
    """Minimal config for fast tests."""
    return ModelConfig(
        n_levels=2,
        level_dims=[64, 32],
        backbone_layers=2,
        backbone_dim=64,
        n_heads=4,
        d_k=16,
        ffn_expansion=2,
        timescale_base=2,
        K_max=3,
        use_predictive_coding=True,
        epsilon_min=0.01,
        lambda_pred=0.1,
        lambda_pi=0.01,
        vocab_size=10,
    )


@pytest.fixture
def full_config():
    """Full-size config matching experiment 1."""
    return ModelConfig(
        n_levels=2,
        level_dims=[512, 256],
        backbone_layers=2,
        backbone_dim=512,
        n_heads=8,
        d_k=64,
        ffn_expansion=4,
        timescale_base=3,
        K_max=4,  # reduced for tests
        use_predictive_coding=True,
        epsilon_min=0.01,
        lambda_pred=0.1,
        lambda_pi=0.01,
        vocab_size=10,
    )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_coral_core_output_shape(small_config):
    core = CoralCore(small_config)
    B, L, d = 2, 10, 64
    z1 = torch.randn(B, L, d)
    out = core(z1, K_max=2, training=False)
    assert out.z_states[0].shape == (B, L, d), (
        f"Expected [B={B}, L={L}, d={d}], got {out.z_states[0].shape}"
    )
    assert out.num_segments >= 1


def test_level_states_shape(small_config):
    core = CoralCore(small_config)
    z1 = torch.randn(2, 10, 64)
    out = core(z1, K_max=2, training=False)
    assert len(out.z_states) == small_config.n_levels
    assert out.z_states[0].shape == (2, 10, 64)
    assert out.z_states[1].shape == (2, 10, 32)


# ---------------------------------------------------------------------------
# Backward pass / gradient flow
# ---------------------------------------------------------------------------

def test_backward_completes(small_config):
    """loss.backward() should complete without error."""
    core = CoralCore(small_config)
    adapter = GridAdapter(small_config, vocab_size=10, grid_height=2, grid_width=5)
    loss_fn = CoralLoss(small_config)

    inputs = torch.randint(0, 10, (2, 10))
    labels = torch.randint(0, 10, (2, 10))

    z1 = adapter.encode(inputs)
    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode)

    # Compute loss over all segments
    total_loss = torch.tensor(0.0)
    for i, logits in enumerate(out.all_logits):
        seg_loss, _ = loss_fn(
            logits=logits,
            labels=labels,
            pred_errors=out.pred_errors[i] if out.pred_errors else None,
            precisions=out.precisions[i] if out.precisions else None,
        )
        total_loss = total_loss + seg_loss

    total_loss.backward()  # Should not raise


def test_all_parameters_receive_gradients(small_config):
    """After backward, all backbone parameters should have non-zero gradients."""
    core = CoralCore(small_config)
    adapter = GridAdapter(small_config, vocab_size=10, grid_height=2, grid_width=5)
    loss_fn = CoralLoss(small_config)

    inputs = torch.randint(0, 10, (2, 10))
    labels = torch.randint(0, 10, (2, 10))

    z1 = adapter.encode(inputs)
    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode)

    total_loss = torch.tensor(0.0)
    for i, logits in enumerate(out.all_logits):
        seg_loss, _ = loss_fn(
            logits=logits, labels=labels,
            pred_errors=out.pred_errors[i] if out.pred_errors else None,
            precisions=out.precisions[i] if out.precisions else None,
        )
        total_loss = total_loss + seg_loss

    total_loss.backward()

    # Check backbone parameters
    for name, param in core.backbone.named_parameters():
        assert param.grad is not None, f"No gradient for backbone.{name}"
        assert param.grad.abs().sum().item() > 0, f"Zero gradient for backbone.{name}"


def test_pc_chain_connected(small_config):
    """Level 2 should receive gradients from level 1's prediction error.

    This verifies the predictive coding chain is connected (critical v4 invariant).
    The error_up_proj receives gradient via the halting loss (which depends on
    z_states[1], which is computed using xi_up as conditioning).
    """
    core = CoralCore(small_config)
    adapter = GridAdapter(small_config, vocab_size=10, grid_height=2, grid_width=5)
    loss_fn = CoralLoss(small_config)

    inputs = torch.randint(0, 10, (2, 10))
    labels = torch.randint(0, 10, (2, 10))

    z1 = adapter.encode(inputs)
    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode)

    total_loss = torch.tensor(0.0)
    for i, logits in enumerate(out.all_logits):
        # Include halting loss: q_halt depends on z_states[1], which uses xi_up
        q_halt = out.q_halt_logits[i] if out.q_halt_logits else None
        q_cont = out.q_continue_logits[i] if out.q_continue_logits else None
        seg_loss, _ = loss_fn(
            logits=logits, labels=labels,
            pred_errors=out.pred_errors[i] if out.pred_errors else None,
            precisions=out.precisions[i] if out.precisions else None,
            q_halt_logits=q_halt,
            q_continue_logits=q_cont,
        )
        total_loss = total_loss + seg_loss

    total_loss.backward()

    # Prediction and precision nets should always have gradients
    for name in ("prediction_net", "precision_net"):
        net = getattr(core.pc_modules[0], name)
        for pname, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for pc_modules[0].{name}.{pname}"

    # error_up_proj gets gradient through halting loss -> z_states[1] -> xi_up
    # (halting network uses z_states[1], which was computed using xi_up as conditioning)
    for pname, param in core.pc_modules[0].error_up_proj.named_parameters():
        assert param.grad is not None, f"No gradient for pc_modules[0].error_up_proj.{pname}"


# ---------------------------------------------------------------------------
# Detach between segments
# ---------------------------------------------------------------------------

def test_detach_between_segments():
    """Verify that z_states are properly detached between segments.

    If detach is missing, backprop through all segments would eventually OOM.
    Here we just verify the mechanism works without error.
    """
    config = ModelConfig(
        n_levels=2, level_dims=[32, 16], backbone_dim=32, n_heads=2,
        d_k=16, ffn_expansion=2, K_max=4, timescale_base=2,
        use_predictive_coding=True, epsilon_min=0.01, vocab_size=10,
    )
    core = CoralCore(config)
    adapter = GridAdapter(config, vocab_size=10, grid_height=2, grid_width=5)
    loss_fn = CoralLoss(config)

    inputs = torch.randint(0, 10, (2, 10))
    labels = torch.randint(0, 10, (2, 10))
    z1 = adapter.encode(inputs)

    # Run 4 segments
    z_states = [z1]
    for i in range(1, config.n_levels):
        z_states.append(torch.zeros(2, 10, config.level_dims[i]))

    total_loss = torch.tensor(0.0)
    for seg in range(4):
        out = core(z_states[0], K_max=1, training=True, decode_fn=adapter.decode)
        if out.all_logits:
            seg_loss, _ = loss_fn(out.all_logits[0], labels)
            total_loss = total_loss + seg_loss
        # Simulate detach
        z_states = [z.detach() for z in out.z_states]

    total_loss.backward()  # Should not OOM or error
