"""Tests for v5 conditioning: gate-only formulation (updated from Session N3).

Session N3 added precision-gated conditioning (gate x pi x rms_normalize(mu)).
Phase 1 synthetic validation showed gate x precision is harmful (gradient competition).
v5 removes precision from the conditioning path entirely.

The test_precision_scales_conditioning_linearly test is removed — that invariant
is no longer valid (precision is not in the conditioning path).

Kept: gradient path test (prediction network receives gradients from task loss).
"""

import torch
import torch.nn.functional as F

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.model.predictive_coding import rms_normalize


def _pc_n2_config(**kwargs) -> ModelConfig:
    """Minimal N=2 pc_only config for conditioning tests."""
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
        **kwargs,
    )


def test_task_loss_gradients_reach_prediction_network():
    """Prediction network must receive gradients from the backbone task loss.

    Under the v5 gate * rms_normalize(mu) formulation, predictions enter the
    backbone state and the task loss gradient reaches the prediction network
    via: decoder -> backbone state -> conditioning gate * rms_normalize(mu) -> prediction_net.

    Setting lambda_pred=0 disables the prediction loss; only task loss is active.
    If prediction_net still gets gradients, the path through the backbone is confirmed.
    """
    torch.manual_seed(7)
    config = _pc_n2_config()
    config.lambda_pred = 0.0  # zero out prediction loss — task loss only
    full_config = CoralConfig(model=config, device="cpu")
    full_config.training.precision = "float32"

    core = CoralCore(config)
    adapter = GridAdapter(full_config, vocab_size=10, grid_height=2, grid_width=2)

    B, L = 2, 4
    inputs = torch.randint(0, 10, (B, L))
    targets = torch.randint(0, 10, (B, L))

    z1 = adapter.encode(inputs)
    output = core(z1, K_max=1, training=True, decode_fn=adapter.decode)

    # Task loss only — pure cross-entropy on the final segment's logits.
    logits = output.all_logits[-1]  # [B, L, vocab]
    task_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    task_loss.backward()

    # Prediction network parameters must have non-zero gradients arriving
    # through the gate * rms_normalize(mu) -> backbone -> task-loss path.
    for name, param in core.pc_modules[0].prediction_net.named_parameters():
        assert param.grad is not None, (
            f"pc_modules[0].prediction_net.{name} has no gradient after task loss backward"
        )
        assert param.grad.abs().max() > 0, (
            f"pc_modules[0].prediction_net.{name} gradient is all-zero after task loss backward"
        )
