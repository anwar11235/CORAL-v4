"""Tests for input re-injection in CoralCore (Bug 1 fix, Session 9).

Verifies that z1_init is added to the backbone input at every inner step in
every segment, so the backbone always has direct access to the task constraints
(e.g. given digits in Sudoku) rather than relying solely on recurrent state.
"""

import torch
import pytest
import torch.nn.functional as F

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(mode: str = "baseline") -> ModelConfig:
    if mode == "full":
        return ModelConfig(
            n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
            ffn_expansion=2, timescale_base=2, K_max=3,
            use_predictive_coding=False, use_crystallisation=True,
            codebook_heads=4, codebook_entries_per_head=8,
            epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="full",
        )
    elif mode == "pc_only":
        return ModelConfig(
            n_levels=2, level_dims=[64, 32], backbone_dim=64, n_heads=4, d_k=16,
            ffn_expansion=2, timescale_base=2, K_max=3,
            use_predictive_coding=True, use_crystallisation=False,
            epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="pc_only",
        )
    else:  # baseline
        return ModelConfig(
            n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
            ffn_expansion=2, timescale_base=2, K_max=3,
            use_predictive_coding=False, use_crystallisation=False,
            epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="baseline",
        )


def _make_adapter_and_core(mode: str = "baseline"):
    config = _make_config(mode)
    full_config = CoralConfig(model=config, device="cpu")
    full_config.training.precision = "float32"
    adapter = GridAdapter(full_config, vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(config)
    return adapter, core, config


# ---------------------------------------------------------------------------
# Different inputs produce different outputs in EVERY segment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["baseline", "pc_only", "full"])
def test_different_inputs_differ_in_all_segments(mode):
    """Different z1_init tensors must yield different logits in every segment.

    Without re-injection: later segments detach from the input and the backbone
    can drift towards indistinguishable outputs for different inputs.
    With re-injection: the input signal is present at every step so
    different inputs stay distinguishable throughout all segments.
    """
    adapter, core, _ = _make_adapter_and_core(mode)
    core.eval()

    torch.manual_seed(0)
    inputs_a = torch.randint(0, 10, (2, 9))
    torch.manual_seed(99)
    inputs_b = torch.randint(0, 10, (2, 9))

    # Ensure the two inputs are actually different
    assert not torch.equal(inputs_a, inputs_b), "Test requires distinct inputs"

    with torch.no_grad():
        z1_a = adapter.encode(inputs_a)
        z1_b = adapter.encode(inputs_b)
        out_a = core(z1_a, K_max=3, training=False, decode_fn=adapter.decode)
        out_b = core(z1_b, K_max=3, training=False, decode_fn=adapter.decode)

    assert len(out_a.all_logits) >= 2, "Need ≥2 segments to test across-segment effect"

    for seg_idx, (la, lb) in enumerate(zip(out_a.all_logits, out_b.all_logits)):
        diff = (la - lb).abs().max().item()
        assert diff > 1e-4, (
            f"mode={mode} segment {seg_idx}: logits are identical for different inputs "
            f"(max diff={diff:.2e}). Input re-injection may not be active."
        )


# ---------------------------------------------------------------------------
# Gradients flow from z1_init through every segment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["baseline", "pc_only", "full"])
def test_gradient_flows_to_z1_init(mode):
    """After backward, z1_init must have a non-zero gradient.

    Without re-injection: the encoder only receives gradient from the first
    segment (z1_init enters z_states[0] which is detached after seg 0).
    With re-injection: z1_init is referenced at every backbone call in every
    segment so gradient always reaches the encoder.
    """
    adapter, core, _ = _make_adapter_and_core(mode)

    inputs = torch.randint(0, 10, (2, 9))
    z1 = adapter.encode(inputs)
    z1.retain_grad()  # keep grad on non-leaf

    out = core(z1, K_max=3, training=True, decode_fn=adapter.decode)
    loss = sum(lg.sum() for lg in out.all_logits)
    loss.backward()

    assert z1.grad is not None, (
        f"mode={mode}: z1_init received no gradient. "
        "Input re-injection must create a direct gradient path from all segments."
    )
    assert z1.grad.abs().sum().item() > 0, (
        f"mode={mode}: z1_init gradient is all-zero."
    )


def test_gradient_norm_grows_with_more_segments():
    """More segments → larger gradient on z1_init via cumulative injection paths."""
    adapter, _, config_obj = _make_adapter_and_core("baseline")
    inputs = torch.randint(0, 10, (2, 9))

    def grad_norm_for_k(k: int) -> float:
        core = CoralCore(config_obj)
        z1 = adapter.encode(inputs)
        z1.retain_grad()
        out = core(z1, K_max=k, training=True, decode_fn=adapter.decode)
        loss = sum(lg.sum() for lg in out.all_logits)
        loss.backward()
        return z1.grad.norm().item() if z1.grad is not None else 0.0

    norm_k1 = grad_norm_for_k(1)
    norm_k3 = grad_norm_for_k(3)

    assert norm_k3 > norm_k1, (
        f"Gradient norm K=3 ({norm_k3:.4f}) should exceed K=1 ({norm_k1:.4f}). "
        "Each additional segment adds a gradient path through z1_init."
    )


# ---------------------------------------------------------------------------
# Shape invariance
# ---------------------------------------------------------------------------

def test_injection_does_not_change_output_shapes():
    """Input re-injection must not alter any output tensor shapes."""
    adapter, core, _ = _make_adapter_and_core("baseline")
    core.eval()

    B, L = 2, 9
    inputs = torch.randint(0, 10, (B, L))
    z1 = adapter.encode(inputs)

    with torch.no_grad():
        out = core(z1, K_max=3, training=False, decode_fn=adapter.decode)

    assert out.z_states[0].shape == (B, L, 64)
    assert all(lg.shape == (B, L, 10) for lg in out.all_logits)
    assert out.num_segments >= 1


# ---------------------------------------------------------------------------
# z1_init is NOT part of z_states (not detached between segments)
# ---------------------------------------------------------------------------

def test_z1_init_not_included_in_detach():
    """z_states detach at segment boundaries but z1_init must stay in the graph.

    We verify that z1_init.grad_fn is still reachable from ALL segment logits,
    even after the internal detach between segments.
    """
    adapter, core, _ = _make_adapter_and_core("baseline")

    inputs = torch.randint(0, 10, (2, 9))
    z1 = adapter.encode(inputs)
    z1.retain_grad()

    # K_max=3: three segments, two detach boundaries
    out = core(z1, K_max=3, training=True, decode_fn=adapter.decode)

    # Gradient from LAST segment (after two detach boundaries) must still reach z1
    last_logits = out.all_logits[-1]
    last_logits.sum().backward()

    assert z1.grad is not None, (
        "Gradient from the last segment must reach z1_init via the injection path. "
        "z1_init must NOT be detached at segment boundaries."
    )
    assert z1.grad.abs().sum().item() > 0
