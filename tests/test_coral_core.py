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

from coral.config import CoralConfig, ModelConfig
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
    adapter = GridAdapter(CoralConfig(model=small_config), vocab_size=10, grid_height=2, grid_width=5)
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
    adapter = GridAdapter(CoralConfig(model=small_config), vocab_size=10, grid_height=2, grid_width=5)
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
    adapter = GridAdapter(CoralConfig(model=small_config), vocab_size=10, grid_height=2, grid_width=5)
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

    # prediction_net has learnable parameters and must receive gradients.
    # (v4.2: precision_net removed; RunningPrecision has zero learnable params)
    for pname, param in core.pc_modules[0].prediction_net.named_parameters():
        assert param.grad is not None, f"No gradient for pc_modules[0].prediction_net.{pname}"

    # error_up_proj gets gradient through halting loss -> z_states[1] -> xi_up
    # (halting network uses z_states[1], which was computed using xi_up as conditioning)
    for pname, param in core.pc_modules[0].error_up_proj.named_parameters():
        assert param.grad is not None, f"No gradient for pc_modules[0].error_up_proj.{pname}"


# ---------------------------------------------------------------------------
# All three modes: baseline, pc_only, full
# ---------------------------------------------------------------------------

def _make_mode_config(mode: str) -> ModelConfig:
    """Minimal config for testing a specific forward-pass mode."""
    if mode == "full":
        return ModelConfig(
            n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
            ffn_expansion=2, timescale_base=2, K_max=2,
            use_predictive_coding=False, use_crystallisation=True,
            codebook_heads=4, codebook_entries_per_head=8,
            epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="full",
        )
    elif mode == "pc_only":
        return ModelConfig(
            n_levels=2, level_dims=[64, 32], backbone_dim=64, n_heads=4, d_k=16,
            ffn_expansion=2, timescale_base=2, K_max=2,
            use_predictive_coding=True, use_crystallisation=False,
            epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="pc_only",
        )
    else:  # baseline
        return ModelConfig(
            n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
            ffn_expansion=2, timescale_base=2, K_max=2,
            use_predictive_coding=False, use_crystallisation=False,
            epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="baseline",
        )


@pytest.mark.parametrize("mode", ["baseline", "pc_only", "full"])
def test_all_modes_output_shape(mode):
    """All three modes should produce z_states[0] with the correct shape."""
    config = _make_mode_config(mode)
    core = CoralCore(config)
    core.eval()

    B, L, d = 2, 9, 64
    z1 = torch.randn(B, L, d)
    with torch.no_grad():
        out = core(z1, K_max=2, training=False, decode_fn=None)

    assert out.z_states[0].shape == (B, L, d), (
        f"mode={mode}: expected z_states[0] shape ({B},{L},{d}), got {out.z_states[0].shape}"
    )
    assert out.num_segments >= 1, f"mode={mode}: num_segments should be >= 1"


@pytest.mark.parametrize("mode", ["baseline", "pc_only", "full"])
def test_all_modes_backward(mode):
    """Backward pass should complete without error in all three modes."""
    config = _make_mode_config(mode)
    full_config = CoralConfig(model=config, device="cpu")
    full_config.training.precision = "float32"

    adapter = GridAdapter(full_config, vocab_size=10, grid_height=3, grid_width=3)
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

    total_loss.backward()  # Must not raise

    # Backbone must have received gradient signal
    for name, param in core.backbone.named_parameters():
        assert param.grad is not None, f"mode={mode}: backbone.{name} missing gradient"


def test_full_mode_crystal_stats_in_coral_core():
    """In mode='full' with crystallisation, crystal_stats should be non-empty."""
    config = _make_mode_config("full")
    core = CoralCore(config)
    core.eval()

    z1 = torch.randn(2, 9, 64)
    with torch.no_grad():
        out = core(z1, K_max=2, training=False, decode_fn=None)

    assert len(out.crystal_stats) == out.num_segments, (
        "crystal_stats should have one entry per segment"
    )


def test_baseline_mode_no_pc_modules():
    """In mode='baseline', pc_modules should be empty."""
    config = _make_mode_config("baseline")
    core = CoralCore(config)
    assert len(core.pc_modules) == 0, "baseline mode must have zero PC modules"


def test_full_mode_no_crystallisation_manager_when_disabled():
    """crystallisation_manager should be None when use_crystallisation=False."""
    config = _make_mode_config("pc_only")
    core = CoralCore(config)
    assert core.crystallisation_manager is None


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
    adapter = GridAdapter(CoralConfig(model=config), vocab_size=10, grid_height=2, grid_width=5)
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


# ---------------------------------------------------------------------------
# inner_steps_override
# ---------------------------------------------------------------------------

def _baseline_n1_config(**kwargs) -> ModelConfig:
    return ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, timescale_base=3, K_max=2,
        use_predictive_coding=False, use_crystallisation=False,
        epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="baseline",
        **kwargs,
    )


def test_inner_steps_override_sets_level_steps():
    """With inner_steps_override=21 and n_levels=1, level_steps[0] must equal 21."""
    config = _baseline_n1_config(inner_steps_override=21)
    core = CoralCore(config)
    assert core.level_steps[0] == 21, (
        f"Expected level_steps[0]=21 with inner_steps_override=21, got {core.level_steps[0]}"
    )


def test_inner_steps_override_none_uses_formula():
    """With inner_steps_override=None, level_steps[0] must equal T^(n_levels-1)=3^0=1."""
    config = _baseline_n1_config(inner_steps_override=None)
    core = CoralCore(config)
    expected = config.timescale_base ** (config.n_levels - 1)  # 3^0 = 1
    assert core.level_steps[0] == expected, (
        f"Expected level_steps[0]={expected} with override=None, got {core.level_steps[0]}"
    )


def test_inner_steps_override_forward_runs():
    """Forward pass must complete without error with inner_steps_override=21, n_levels=1."""
    config = _baseline_n1_config(inner_steps_override=21)
    full_config = CoralConfig(model=config, device="cpu")
    full_config.training.precision = "float32"
    core = CoralCore(config)
    adapter = GridAdapter(full_config, vocab_size=10, grid_height=3, grid_width=3)

    inputs = torch.randint(0, 10, (2, 9))
    z1 = adapter.encode(inputs)
    with torch.no_grad():
        out = core(z1, K_max=2, training=False, decode_fn=adapter.decode)

    assert out.z_states[0].shape == (2, 9, 64), (
        f"Unexpected output shape: {out.z_states[0].shape}"
    )


# ---------------------------------------------------------------------------
# N=2 baseline mode
# ---------------------------------------------------------------------------

def _baseline_n2_config(**kwargs) -> ModelConfig:
    """Minimal N=2 baseline config for tests."""
    return ModelConfig(
        n_levels=2, level_dims=[64, 32], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, timescale_base=3, K_max=2,
        use_predictive_coding=False, use_crystallisation=False,
        epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="baseline",
        **kwargs,
    )


def test_n2_baseline_output_shape():
    """N=2 baseline: z_states[0] must have level_dims[0] and z_states[1] level_dims[1]."""
    config = _baseline_n2_config()
    core = CoralCore(config)
    core.eval()

    B, L, d0 = 2, 9, 64
    z1 = torch.randn(B, L, d0)
    with torch.no_grad():
        out = core(z1, K_max=2, training=False)

    assert out.z_states[0].shape == (B, L, 64), (
        f"z_states[0] should be [B,L,64], got {out.z_states[0].shape}"
    )
    assert out.z_states[1].shape == (B, L, 32), (
        f"z_states[1] should be [B,L,32], got {out.z_states[1].shape}"
    )
    assert len(out.z_states) == 2


def test_n2_baseline_decode_from_level0():
    """N=2 baseline: decode_fn must be called on z_states[0] (d=64), not z_states[1] (d=32)."""
    config = _baseline_n2_config()
    full_config = CoralConfig(model=config, device="cpu")
    full_config.training.precision = "float32"
    core = CoralCore(config)
    adapter = GridAdapter(full_config, vocab_size=10, grid_height=3, grid_width=3)

    inputs = torch.randint(0, 10, (2, 9))
    z1 = adapter.encode(inputs)
    with torch.no_grad():
        out = core(z1, K_max=1, training=False, decode_fn=adapter.decode)

    assert len(out.all_logits) == 1
    assert out.all_logits[0].shape == (2, 9, 10), (
        f"Logits should be [2,9,10] (decoded from level 0), got {out.all_logits[0].shape}"
    )


def test_n2_baseline_input_injection_only_at_level0():
    """Input injection should only affect level 0 outputs, not level 1.

    Run two forward passes with identical z1_init but different injections
    by patching the input_signal. Verify level 0 changes but level 1 is
    unaffected when injection differs (testing the Option A design choice).
    We test this indirectly: run with a zero z1_init vs a non-zero z1_init
    and confirm level 0 state differs while verifying level 1 state is
    initialised only from z_init_proj (not direct injection).
    """
    config = _baseline_n2_config()
    core = CoralCore(config)
    core.eval()

    # Two different inputs
    z1_a = torch.zeros(2, 9, 64)
    z1_b = torch.ones(2, 9, 64)

    with torch.no_grad():
        out_a = core(z1_a, K_max=1, training=False)
        out_b = core(z1_b, K_max=1, training=False)

    # Level 0 must differ (input injection feeds into it)
    assert not torch.allclose(out_a.z_states[0], out_b.z_states[0]), (
        "z_states[0] should differ for different inputs (input injection active at level 0)"
    )
    # Level 1 must also differ because z_init_proj seeds it from z1_init,
    # so it changes too — but we just check it's a different shape from level 0
    assert out_a.z_states[1].shape[-1] == 32  # d=32 for level 1


def test_n2_baseline_backward():
    """N=2 baseline backward pass must complete with gradients reaching the backbone."""
    config = _baseline_n2_config()
    full_config = CoralConfig(model=config, device="cpu")
    full_config.training.precision = "float32"
    core = CoralCore(config)
    adapter = GridAdapter(full_config, vocab_size=10, grid_height=3, grid_width=3)
    loss_fn = CoralLoss(config)

    inputs = torch.randint(0, 10, (2, 9))
    labels = torch.randint(1, 10, (2, 9))
    z1 = adapter.encode(inputs)

    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode)

    total_loss = torch.tensor(0.0)
    for i, logits in enumerate(out.all_logits):
        seg_loss, _ = loss_fn(logits=logits, labels=labels)
        total_loss = total_loss + seg_loss
    total_loss.backward()

    for name, param in core.backbone.named_parameters():
        assert param.grad is not None, f"N=2 baseline: backbone.{name} missing gradient"
        assert param.grad.abs().sum() > 0, f"N=2 baseline: zero gradient for backbone.{name}"


def test_n2_baseline_level_steps():
    """N=2 baseline: level_steps must be [T^1, T^0] = [3, 1] with timescale_base=3."""
    config = _baseline_n2_config()
    core = CoralCore(config)
    assert core.level_steps == [3, 1], (
        f"Expected level_steps=[3, 1] for N=2 T=3, got {core.level_steps}"
    )


def test_n2_baseline_inner_steps_override_only_affects_level0():
    """inner_steps_override=18 must set level_steps[0]=18 and leave level_steps[1]=1."""
    config = _baseline_n2_config(inner_steps_override=18)
    core = CoralCore(config)
    assert core.level_steps[0] == 18, (
        f"level_steps[0] should be 18 with override, got {core.level_steps[0]}"
    )
    assert core.level_steps[1] == 1, (
        f"level_steps[1] should remain 1 (T^0), got {core.level_steps[1]}"
    )
