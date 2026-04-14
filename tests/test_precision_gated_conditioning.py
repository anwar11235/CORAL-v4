"""Tests for Session N3: precision-gated top-down conditioning.

Verification criteria:
  - gate × (π × μ): doubling π produces exactly 2× the conditioning addition
    (backbone output is identical; only the conditioning term differs)
  - Prediction network receives gradients from the backbone task loss via the
    new gate × (π × μ) path (not just from loss/prediction)
"""

import torch
import torch.nn.functional as F

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore


def _pc_n2_config(**kwargs) -> ModelConfig:
    """Minimal N=2 pc_only config for precision-gated conditioning tests."""
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


def test_precision_scales_conditioning_linearly():
    """Doubling π must double the conditioning addition to z_new.

    Under gate × (π × μ), the addition is gate * pi * conditioning.
    Backbone_in does not contain conditioning, so the backbone output
    z_backbone is identical for both precision values.  Therefore:

        z_out1 = z_backbone + gate * pi1 * conditioning
        z_out2 = z_backbone + gate * pi2 * conditioning
        z_out2 - z_out1 = gate * (pi2 - pi1) * conditioning

    With gate=1.0, pi1=1.0, pi2=2.0:
        z_out2 - z_out1 == conditioning  (elementwise)
    """
    torch.manual_seed(42)
    config = _pc_n2_config()
    core = CoralCore(config)
    core.eval()

    # Pin gate to 1.0 so the effect is purely from precision scaling.
    core.cond_gate[0].data.fill_(1.0)

    B, L, d0 = 1, 4, 64
    z = torch.zeros(B, L, d0)
    conditioning = torch.randn(B, L, d0)

    # Pass 1: ema_var = 0.99  →  pi = 1 / (0.99 + 0.01) = 1.0
    core.pc_modules[0].running_precision.ema_var.fill_(0.99)
    with torch.no_grad():
        z_out1 = core._run_level(z.clone(), level_idx=0, n_steps=1,
                                 conditioning=conditioning)

    # Pass 2: ema_var = 0.49  →  pi = 1 / (0.49 + 0.01) = 2.0
    core.pc_modules[0].running_precision.ema_var.fill_(0.49)
    with torch.no_grad():
        z_out2 = core._run_level(z.clone(), level_idx=0, n_steps=1,
                                 conditioning=conditioning)

    # Expected: z_out2 - z_out1 = gate * (pi2 - pi1) * conditioning
    #                            = 1.0  *  (2.0 - 1.0)  * conditioning
    #                            = conditioning
    diff = z_out2 - z_out1
    max_err = (diff - conditioning).abs().max().item()
    assert max_err < 1e-4, (
        f"Expected z_out2 - z_out1 == conditioning (pi doubles, gate=1), "
        f"max elementwise error = {max_err:.6f}"
    )


def test_task_loss_gradients_reach_prediction_network():
    """Prediction network must receive gradients from the backbone task loss.

    Under the new gate × (π × μ) formulation, predictions enter the backbone
    state directly (not only via loss/prediction).  Setting lambda_pred=0
    disables the prediction loss, leaving only the task loss.  If the
    prediction network still gets gradients, the coupling through the
    backbone is confirmed.
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
    # through the gate × (π × μ) → backbone → task-loss path.
    for name, param in core.pc_modules[0].prediction_net.named_parameters():
        assert param.grad is not None, (
            f"pc_modules[0].prediction_net.{name} has no gradient after task loss backward"
        )
        assert param.grad.abs().max() > 0, (
            f"pc_modules[0].prediction_net.{name} gradient is all-zero after task loss backward"
        )
