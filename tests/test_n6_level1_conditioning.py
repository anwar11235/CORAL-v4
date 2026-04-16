"""Tests for v5 conditioning at level 1 (updated from Session N6).

N6 fixed the self-erasing gate x (xi_up - z_in) formula.
v5 replaces the scalar gate entirely with ConditioningGate MLP.

Both levels now use the same formula:
  gate = conditioning_gates[level_idx](z_new)    # per-feature sigmoid
  z_new = z_new + gate * rms_normalize(conditioning)

Verification criteria:
  1. Level-1 conditioning contribution is scale-invariant w.r.t. xi_up scale.
  2. Level-0 conditioning fix (N4) is still working correctly (regression).
  3. conditioning_gates[1] MLP receives a non-trivial gradient from a loss on
     z_states[1] — the within-segment gradient path is intact.
"""

import torch

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pc_n2_config() -> ModelConfig:
    """Minimal N=2 pc_only config for level-1 conditioning tests."""
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


# ---------------------------------------------------------------------------
# Test 1 — Level-1 conditioning is scale-invariant under rms_normalize
# ---------------------------------------------------------------------------

def test_level1_conditioning_scale_invariant():
    """Scaling xi_up by 1000 must produce the same level-1 z_new.

    v5 formula: z_new = backbone(z) + gate(z_new) * rms_normalize(xi_up).
    rms_normalize(k * xi_up) == rms_normalize(xi_up) for any scalar k != 0.
    gate(z_new) is computed from z_new (same for both calls since backbone
    input doesn't include conditioning), so the outputs must be identical.
    """
    torch.manual_seed(42)
    config = _pc_n2_config()
    core = CoralCore(config)
    core.eval()

    B, L, d1 = 1, 4, 32   # level-1 dim
    z1 = torch.zeros(B, L, d1)

    xi_up_unit = torch.randn(B, L, d1)
    xi_up_large = xi_up_unit * 1000.0   # same direction, 1000x scale

    with torch.no_grad():
        z_out_unit  = core._run_level(z1.clone(), level_idx=1, n_steps=1,
                                      conditioning=xi_up_unit)
        z_out_large = core._run_level(z1.clone(), level_idx=1, n_steps=1,
                                      conditioning=xi_up_large)

    max_err = (z_out_large - z_out_unit).abs().max().item()
    assert max_err < 1e-4, (
        f"rms_normalize should make level-1 conditioning scale-invariant, "
        f"but 1000x scaling produced max diff = {max_err:.6f}. "
        f"Check that rms_normalize is applied in the v5 conditioning formula."
    )


# ---------------------------------------------------------------------------
# Test 2 — Level-0 fix (N4) still works correctly after v5 changes
# ---------------------------------------------------------------------------

def test_level0_conditioning_still_bounded_after_v5():
    """N4's rms_normalize at level 0 must be unaffected by the v5 changes.

    Inject a conditioning signal with RMS ~ 1000 at level 0. The conditioning
    delta (z_out - z_backbone) must be << 1000 (bounded by sigmoid gate * RMSNorm).
    At init, gate ~ sigmoid(-2) ~ 0.12, so delta_rms ~ 0.12.
    """
    torch.manual_seed(0)
    config = _pc_n2_config()
    core = CoralCore(config)
    core.eval()

    B, L, d0 = 2, 9, 64
    z = torch.zeros(B, L, d0)

    raw = torch.randn(B, L, d0)
    conditioning_large = raw / raw.pow(2).mean(dim=-1, keepdim=True).sqrt() * 1000.0

    with torch.no_grad():
        z_backbone = core._run_level(z.clone(), level_idx=0, n_steps=1,
                                     conditioning=None)
        z_out      = core._run_level(z.clone(), level_idx=0, n_steps=1,
                                     conditioning=conditioning_large)

    delta = z_out - z_backbone
    delta_rms = delta.pow(2).mean(dim=-1).sqrt().mean().item()

    assert delta_rms < 1.0, (
        f"Level-0 conditioning delta_rms should be < 1.0 (sigmoid gate * RMSNorm). "
        f"Got {delta_rms:.4f}. v5 changes may have broken the level-0 path."
    )


# ---------------------------------------------------------------------------
# Test 3 — conditioning_gates[1] MLP is in the computation graph
# ---------------------------------------------------------------------------

def test_level1_gate_in_computation_graph():
    """conditioning_gates[1] MLP parameters must receive gradients from a loss on z_states[1].

    v5 formula at level 1:
      gate = conditioning_gates[1](z_new)    # gate net takes z_new as input
      z_out = z_new + gate * rms_normalize(xi_up)

    The path from z_out -> gate -> gate_net parameters is:
      loss(z_out) -> d(z_out)/d(gate) -> d(gate)/d(gate_net.parameters)

    This is the within-segment gradient path. v4's scalar cond_gate[1] had
    an effectively-severed gradient path from the task loss (N6 finding).
    The v5 gate MLP gets gradient via this direct path.
    """
    torch.manual_seed(7)
    config = _pc_n2_config()
    core = CoralCore(config)

    B, L, d1 = 2, 4, 32  # level-1 dim
    z1 = torch.zeros(B, L, d1)

    # xi_up = conditioning for level 1
    xi_up = torch.randn(B, L, d1, requires_grad=True)

    # Run level 1: z_out = backbone(z1) + gate(z_new) * rms_normalize(xi_up)
    z_out = core._run_level(z1, level_idx=1, n_steps=1, conditioning=xi_up)

    # Synthetic loss on z_states[1]
    synthetic_loss = z_out.pow(2).mean()
    synthetic_loss.backward()

    gate_mod = core.conditioning_gates[1]
    grads = [p.grad for p in gate_mod.parameters() if p.grad is not None]
    assert len(grads) > 0, (
        "conditioning_gates[1] MLP has no parameter gradients from a loss on z_states[1]. "
        "The within-segment gradient path may be broken."
    )
    max_grad = max(g.abs().max().item() for g in grads)
    assert max_grad > 1e-6, (
        f"conditioning_gates[1] gradient is effectively zero: max={max_grad:.2e}. "
        f"Expected non-trivial gradient via gate(z_new) * rms_normalize(xi_up) -> z_out -> loss."
    )
