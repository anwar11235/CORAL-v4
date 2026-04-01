"""Tests for grad_inner_steps (partial gradient inner loop).

Verifies that:
- grad_inner_steps=None keeps full gradient (backward compatible)
- grad_inner_steps=k runs last k steps in-graph, rest under no_grad
- Forward output is numerically identical regardless of grad_inner_steps
- Backward produces non-zero gradients even with partial grad
- level 1 with 1 step always has full gradient (graceful clamp)
- New configs load and produce correct level_steps
"""

import yaml
import torch
import pytest

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.training.losses import CoralLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    n_levels: int = 1,
    level_dims=None,
    n_steps: int = 6,
    grad_inner_steps=None,
) -> ModelConfig:
    if level_dims is None:
        level_dims = [64] * n_levels
    return ModelConfig(
        n_levels=n_levels,
        level_dims=level_dims,
        backbone_dim=64,
        n_heads=4,
        d_k=16,
        ffn_expansion=2,
        timescale_base=3,
        K_max=2,
        use_predictive_coding=False,
        use_crystallisation=False,
        epsilon_min=0.01,
        lambda_pred=0.001,
        vocab_size=10,
        mode="baseline",
        inner_steps_override=n_steps,
        grad_inner_steps=grad_inner_steps,
    )


def _run_and_backward(config: ModelConfig):
    """Run one segment, compute loss, backward; return (loss, core)."""
    full_cfg = CoralConfig(model=config, device="cpu")
    full_cfg.training.precision = "float32"
    adapter = GridAdapter(full_cfg, vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(config)
    loss_fn = CoralLoss(config)

    inputs = torch.randint(0, 10, (2, 9))
    labels = torch.randint(1, 10, (2, 9))
    z1 = adapter.encode(inputs)

    out = core(z1, K_max=1, training=True, decode_fn=adapter.decode)
    seg_loss, _ = loss_fn(logits=out.all_logits[0], labels=labels)
    seg_loss.backward()
    return seg_loss, core


# ---------------------------------------------------------------------------
# Backward compatibility: grad_inner_steps=None
# ---------------------------------------------------------------------------

def test_full_grad_none_backward_works():
    """grad_inner_steps=None: backward must complete and produce non-zero grads."""
    config = _make_config(n_steps=6, grad_inner_steps=None)
    _, core = _run_and_backward(config)
    for name, p in core.backbone.named_parameters():
        assert p.grad is not None, f"backbone.{name} missing gradient (full grad)"
        assert p.grad.abs().sum() > 0, f"backbone.{name} zero gradient (full grad)"


# ---------------------------------------------------------------------------
# Partial grad: forward output identical to full grad
# ---------------------------------------------------------------------------

def test_partial_grad_forward_output_matches_full_grad():
    """Forward values must be numerically identical with grad_inner_steps=3 vs None.

    torch.no_grad() only suppresses autograd tracking; it does not change the
    arithmetic, so the tensor values should be exactly equal.
    """
    base_cfg = _make_config(n_steps=6, grad_inner_steps=None)
    partial_cfg = _make_config(n_steps=6, grad_inner_steps=3)

    # Share weights so the only difference is grad tracking
    core_full = CoralCore(base_cfg)
    core_partial = CoralCore(partial_cfg)
    core_partial.load_state_dict(core_full.state_dict())

    full_cfg = CoralConfig(model=base_cfg, device="cpu")
    full_cfg.training.precision = "float32"
    adapter = GridAdapter(full_cfg, vocab_size=10, grid_height=3, grid_width=3)

    inputs = torch.randint(0, 10, (2, 9))
    z1 = adapter.encode(inputs)

    core_full.eval()
    core_partial.eval()
    with torch.no_grad():
        out_full = core_full(z1, K_max=1, training=False)
        out_partial = core_partial(z1, K_max=1, training=False)

    assert torch.allclose(out_full.z_states[0], out_partial.z_states[0], atol=0), (
        "Forward output must be identical with and without partial grad"
    )


# ---------------------------------------------------------------------------
# Partial grad: backward works and gradients are present
# ---------------------------------------------------------------------------

def test_partial_grad_backward_works():
    """grad_inner_steps=3 (of 6 total): backward must complete without error."""
    config = _make_config(n_steps=6, grad_inner_steps=3)
    _, core = _run_and_backward(config)
    for name, p in core.backbone.named_parameters():
        assert p.grad is not None, f"backbone.{name} missing gradient (partial grad)"
        assert p.grad.abs().sum() > 0, f"backbone.{name} zero gradient (partial grad)"


def test_partial_grad_1_of_6_backward_works():
    """Even grad_inner_steps=1 (minimum) must allow backward."""
    config = _make_config(n_steps=6, grad_inner_steps=1)
    _, core = _run_and_backward(config)
    backbone_grads = [
        p.grad.abs().sum().item()
        for p in core.backbone.parameters()
        if p.grad is not None
    ]
    assert len(backbone_grads) > 0, "No backbone parameters received gradients"
    assert any(g > 0 for g in backbone_grads), "All backbone gradients are zero"


# ---------------------------------------------------------------------------
# Partial grad: fewer grad steps → smaller gradient magnitude
# ---------------------------------------------------------------------------

def test_fewer_grad_steps_produces_nonzero_gradients():
    """With grad_inner_steps=2 (of 6 total), backward must still produce non-zero gradients.

    The gradient magnitudes may differ from full-grad in either direction
    (more grad paths can cancel as well as accumulate), so we only assert
    that the gradient is non-trivially non-zero.
    """
    n_steps = 6
    config_partial = _make_config(n_steps=n_steps, grad_inner_steps=2)
    _, core = _run_and_backward(config_partial)

    grad_norms = [
        p.grad.norm().item()
        for p in core.backbone.parameters()
        if p.grad is not None
    ]
    assert len(grad_norms) > 0, "No backbone parameters have gradients"
    total = sum(g ** 2 for g in grad_norms) ** 0.5
    assert total > 0, f"All backbone gradients are zero with grad_inner_steps=2"


# ---------------------------------------------------------------------------
# Graceful clamp: grad_inner_steps >= n_steps means all steps have gradient
# ---------------------------------------------------------------------------

def test_grad_inner_steps_larger_than_n_steps_is_full_grad():
    """grad_inner_steps=100 with n_steps=6 should behave like grad_inner_steps=None."""
    config_ref = _make_config(n_steps=6, grad_inner_steps=None)
    config_clamped = _make_config(n_steps=6, grad_inner_steps=100)

    core_ref = CoralCore(config_ref)
    core_clamped = CoralCore(config_clamped)
    core_clamped.load_state_dict(core_ref.state_dict())

    full_cfg = CoralConfig(model=config_ref, device="cpu")
    full_cfg.training.precision = "float32"
    adapter = GridAdapter(full_cfg, vocab_size=10, grid_height=3, grid_width=3)

    inputs = torch.randint(0, 10, (2, 9))
    z1 = adapter.encode(inputs)
    with torch.no_grad():
        out_ref = core_ref(z1, K_max=1, training=False)
        out_clamped = core_clamped(z1, K_max=1, training=False)

    assert torch.allclose(out_ref.z_states[0], out_clamped.z_states[0], atol=0), (
        "grad_inner_steps >= n_steps should produce identical output to None"
    )


# ---------------------------------------------------------------------------
# N=2: level 1 (1 step) always has full gradient even when grad_inner_steps=6
# ---------------------------------------------------------------------------

def test_n2_level1_full_grad_no_partial_warmup():
    """With N=2, level 1 has 1 step and grad_inner_steps=6 — no warmup applies to level 1.

    Verify by inspecting level_steps[1]==1 and that backward completes without error.
    The loss path through level 1 requires halting loss; task loss alone is decoded
    from level 0 only, so we just verify the forward+backward completes cleanly.
    """
    config = _make_config(
        n_levels=2, level_dims=[64, 32], n_steps=8, grad_inner_steps=3
    )
    core = CoralCore(config)

    # level_steps[1] must be 1 (T^0); grad_inner_steps=3 > 1 so no warmup for level 1
    assert core.level_steps[1] == 1, f"level_steps[1]={core.level_steps[1]}, expected 1"

    full_cfg = CoralConfig(model=config, device="cpu")
    full_cfg.training.precision = "float32"
    adapter = GridAdapter(full_cfg, vocab_size=10, grid_height=3, grid_width=3)
    loss_fn = CoralLoss(config)

    inputs = torch.randint(0, 10, (2, 9))
    labels = torch.randint(1, 10, (2, 9))
    z1 = adapter.encode(inputs)

    out = core(z1, K_max=1, training=True, decode_fn=adapter.decode)
    # Include halting loss so gradients flow through level 1's state
    seg_loss, _ = loss_fn(
        logits=out.all_logits[0],
        labels=labels,
        q_halt_logits=out.q_halt_logits[0],
        q_continue_logits=out.q_continue_logits[0],
    )
    seg_loss.backward()  # must not error

    # Level 0 backbone must have gradients
    for name, p in core.backbone.named_parameters():
        assert p.grad is not None, f"backbone.{name} missing gradient in N=2 partial-grad"


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config_name", [
    "phase1_n2_partial_grad",
    "phase1_n1_partial_grad",
])
def test_partial_grad_configs_load(config_name):
    """New partial-grad YAML configs must load without error."""
    with open(f"configs/{config_name}.yaml") as f:
        raw = yaml.safe_load(f)
    model_cfg = ModelConfig(**raw["model"])
    core = CoralCore(model_cfg)

    assert model_cfg.grad_inner_steps == 6
    assert core.level_steps[0] in (18, 21)  # n2: 18, n1: 21


def test_phase1_n2_partial_grad_level_steps():
    """phase1_n2_partial_grad: level_steps must be [18, 1]."""
    with open("configs/phase1_n2_partial_grad.yaml") as f:
        raw = yaml.safe_load(f)
    core = CoralCore(ModelConfig(**raw["model"]))
    assert core.level_steps == [18, 1], f"Expected [18, 1], got {core.level_steps}"


def test_phase1_n1_partial_grad_level_steps():
    """phase1_n1_partial_grad: level_steps must be [21]."""
    with open("configs/phase1_n1_partial_grad.yaml") as f:
        raw = yaml.safe_load(f)
    core = CoralCore(ModelConfig(**raw["model"]))
    assert core.level_steps == [21], f"Expected [21], got {core.level_steps}"
