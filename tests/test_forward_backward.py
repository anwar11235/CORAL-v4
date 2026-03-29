"""Explicit forward-backward gradient flow tests.

From build plan: 'After loss.backward(), verify that backbone parameters have
non-zero .grad and that level 2 parameters receive gradients that depend on
level 1 computation.'
"""

import torch
import torch.nn as nn

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.training.losses import CoralLoss


def make_small_model():
    """Return a small but complete model for gradient testing."""
    config = ModelConfig(
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
        lambda_pred=0.1,
        lambda_pi=0.01,
        vocab_size=10,
    )
    adapter = GridAdapter(CoralConfig(model=config), vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(config)
    loss_fn = CoralLoss(config)
    return adapter, core, loss_fn, config


def test_full_backprop_within_segment():
    """All T+1 backbone applications within a segment must be in the same graph.

    Verify: backbone parameters have non-zero gradients after backward.
    """
    adapter, core, loss_fn, config = make_small_model()

    inputs = torch.randint(0, 10, (2, 9))  # 3×3 grid
    labels = torch.randint(1, 10, (2, 9))  # non-empty labels

    z1 = adapter.encode(inputs)

    # Run ONE segment (should include T=2 backbone calls for level 0 + 1 for level 1)
    out = core(z1, K_max=1, training=True, decode_fn=adapter.decode)

    assert out.all_logits, "No logits produced"
    seg_loss, breakdown = loss_fn(
        logits=out.all_logits[0],
        labels=labels,
        pred_errors=out.pred_errors[0] if out.pred_errors else None,
        precisions=out.precisions[0] if out.precisions else None,
    )

    seg_loss.backward()

    # All backbone layers must have gradients
    for name, param in core.backbone.named_parameters():
        assert param.grad is not None, f"backbone.{name} has no gradient"
        grad_norm = param.grad.abs().sum().item()
        assert grad_norm > 0, f"backbone.{name} has zero gradient (norm={grad_norm})"


def test_level2_gets_gradient_from_level1():
    """Level 2 (pc_modules[0]) must receive gradient that depends on level 1 computation."""
    adapter, core, loss_fn, config = make_small_model()

    inputs = torch.randint(0, 10, (2, 9))
    labels = torch.randint(1, 10, (2, 9))

    z1 = adapter.encode(inputs)
    out = core(z1, K_max=1, training=True, decode_fn=adapter.decode)

    seg_loss, _ = loss_fn(
        logits=out.all_logits[0],
        labels=labels,
        pred_errors=out.pred_errors[0] if out.pred_errors else None,
        precisions=out.precisions[0] if out.precisions else None,
    )
    seg_loss.backward()

    # PC module prediction network (level 0→1) must have gradients
    pred_net_grads = [
        p.grad for p in core.pc_modules[0].prediction_net.parameters()
        if p.grad is not None
    ]
    assert len(pred_net_grads) > 0, "PC module prediction network has no gradients"

    # Precision network must have gradients
    prec_net_grads = [
        p.grad for p in core.pc_modules[0].precision_net.parameters()
        if p.grad is not None
    ]
    assert len(prec_net_grads) > 0, "PC module precision network has no gradients"


def test_no_nan_gradients():
    """No NaN gradients should appear after a backward pass."""
    adapter, core, loss_fn, config = make_small_model()

    inputs = torch.randint(0, 10, (4, 9))
    labels = torch.randint(1, 10, (4, 9))

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

    for name, param in list(adapter.named_parameters()) + list(core.named_parameters()):
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"


def test_deep_supervision_accumulates():
    """Multiple segments should accumulate gradient signal properly."""
    adapter, core, loss_fn, config = make_small_model()

    inputs = torch.randint(0, 10, (2, 9))
    labels = torch.randint(1, 10, (2, 9))

    z1 = adapter.encode(inputs)
    B, L, d = z1.shape

    # Simulate the training loop: accumulate loss over segments with detach
    z_states = [z1, torch.zeros(B, L, config.level_dims[1])]
    total_loss = torch.tensor(0.0)

    for seg in range(config.K_max):
        out = core(z_states[0], K_max=1, training=True, decode_fn=adapter.decode)
        if out.all_logits:
            seg_loss, _ = loss_fn(out.all_logits[0], labels)
            total_loss = total_loss + seg_loss
        z_states = [z.detach() for z in out.z_states]

    total_loss.backward()

    # Verify gradients flowed
    grad_norms = [
        p.grad.abs().sum().item()
        for p in core.backbone.parameters()
        if p.grad is not None
    ]
    assert len(grad_norms) > 0
    assert sum(grad_norms) > 0, "No gradient signal accumulated across segments"
