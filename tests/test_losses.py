"""Tests for coral/training/losses.py.

Verification criteria:
  - commitment_loss is non-negative
  - disentanglement_loss is non-negative
  - Total loss in full mode includes commitment and disentanglement
  - Total loss in baseline mode does NOT include PC or crystallisation losses
  - Precision regulariser is NOT in the loss (verified removed in v4.2)
  - Stablemax cross-entropy is finite and non-negative
  - Amortisation loss is non-negative
"""

import torch
import pytest

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.model.crystallisation import MultiHeadedCodebook
from coral.training.losses import CoralLoss, stablemax_cross_entropy, amortisation_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _baseline_config():
    return ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, timescale_base=2, K_max=2,
        use_predictive_coding=False, use_crystallisation=False,
        use_amort=False, lambda_amort=0.0,
        epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="baseline",
    )


def _full_mode_config():
    return ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, timescale_base=2, K_max=2,
        use_predictive_coding=False, use_crystallisation=True,
        codebook_heads=4, codebook_entries_per_head=8,
        lambda_commit=0.25, lambda_dis=0.01,
        epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="full",
    )


def _make_fake_logits_labels(B=2, L=9, vocab=10):
    logits = torch.randn(B, L, vocab)
    labels = torch.randint(1, vocab, (B, L))
    return logits, labels


# ---------------------------------------------------------------------------
# stablemax_cross_entropy
# ---------------------------------------------------------------------------

def test_stablemax_ce_finite_and_nonnegative():
    """Stablemax cross-entropy values must be finite and >= 0."""
    logits, labels = _make_fake_logits_labels()
    per_token = stablemax_cross_entropy(logits, labels)
    assert per_token.shape == (2, 9)
    assert per_token.isfinite().all(), "stablemax CE must be finite"
    assert (per_token >= 0).all(), "stablemax CE must be non-negative"


def test_stablemax_ce_ignored_positions_zero():
    """Positions with label=-100 should contribute 0 to the loss."""
    logits, labels = _make_fake_logits_labels()
    labels[:, 3] = -100  # mark position 3 as ignored
    per_token = stablemax_cross_entropy(logits, labels)
    assert per_token[:, 3].abs().max().item() < 1e-9, (
        "Ignored position should have zero loss contribution"
    )


# ---------------------------------------------------------------------------
# commitment_loss and disentanglement_loss (non-negativity)
# ---------------------------------------------------------------------------

def test_commitment_loss_nonnegative():
    """Commitment loss ||z - sg(e)||² is always >= 0."""
    cb = MultiHeadedCodebook(dim=64, n_heads=4, entries_per_head=8)
    z = torch.randn(2, 9, 64)
    z_q, _, _ = cb.quantise(z, hard=True)
    loss = cb.commitment_loss(z, z_q)
    assert loss.item() >= 0.0, f"commitment_loss must be >= 0, got {loss.item()}"


def test_disentanglement_loss_nonnegative():
    """Disentanglement loss Σ||C_h1.T @ C_h2||²_F is always >= 0."""
    cb = MultiHeadedCodebook(dim=64, n_heads=4, entries_per_head=8)
    loss = cb.disentanglement_loss()
    assert loss.item() >= 0.0, f"disentanglement_loss must be >= 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# CoralLoss in baseline mode
# ---------------------------------------------------------------------------

def test_baseline_loss_no_pc_no_crystal():
    """In baseline mode, PC and crystallisation losses should be zero."""
    config = _baseline_config()
    loss_fn = CoralLoss(config)

    logits, labels = _make_fake_logits_labels()
    total, breakdown = loss_fn(logits=logits, labels=labels)

    # These should be zero in baseline mode
    assert breakdown["loss/prediction"].item() == pytest.approx(0.0), (
        "baseline mode: prediction loss should be 0"
    )
    assert breakdown["loss/commitment"].item() == pytest.approx(0.0), (
        "baseline mode: commitment loss should be 0"
    )
    assert breakdown["loss/crystallisation"].item() == pytest.approx(0.0), (
        "baseline mode: disentanglement loss should be 0"
    )
    assert breakdown["loss/amortisation"].item() == pytest.approx(0.0), (
        "baseline mode: amortisation loss should be 0"
    )

    # Task loss must be positive
    assert breakdown["loss/task"].item() > 0.0, "Task loss must be positive"


def test_precision_regulariser_not_in_loss():
    """Precision regulariser was removed in v4.2: breakdown must NOT contain loss/precision_reg."""
    config = _baseline_config()
    loss_fn = CoralLoss(config)
    logits, labels = _make_fake_logits_labels()
    _, breakdown = loss_fn(logits=logits, labels=labels)

    assert "loss/precision_reg" not in breakdown, (
        "loss/precision_reg was removed in v4.2 — should not appear in breakdown"
    )


# ---------------------------------------------------------------------------
# CoralLoss in full mode (crystallisation losses present)
# ---------------------------------------------------------------------------

def test_full_mode_loss_includes_commit_and_dis():
    """In full mode with commit + dis losses passed, total should increase."""
    config = _full_mode_config()
    loss_fn = CoralLoss(config)

    logits, labels = _make_fake_logits_labels()

    # Without crystallisation losses
    total_no_crys, bd_no_crys = loss_fn(logits=logits, labels=labels)

    # With non-trivial crystallisation losses
    commit_loss = torch.tensor(1.0)
    dis_loss = torch.tensor(0.5)
    total_with_crys, bd_with_crys = loss_fn(
        logits=logits, labels=labels,
        commitment_loss=commit_loss,
        disentanglement_loss=dis_loss,
    )

    assert bd_with_crys["loss/commitment"].item() > 0, (
        "full mode: commitment loss should be positive when non-zero input provided"
    )
    assert bd_with_crys["loss/crystallisation"].item() > 0, (
        "full mode: crystallisation loss should be positive when non-zero input provided"
    )
    assert total_with_crys.item() > total_no_crys.item(), (
        "Total loss should increase when crystallisation losses are included"
    )


def test_full_mode_loss_total_is_sum_of_parts():
    """Total loss in full mode should equal the sum of its components."""
    config = _full_mode_config()
    loss_fn = CoralLoss(config)

    logits, labels = _make_fake_logits_labels()
    commit_loss = torch.tensor(0.5)
    dis_loss = torch.tensor(0.3)

    total, bd = loss_fn(
        logits=logits, labels=labels,
        commitment_loss=commit_loss,
        disentanglement_loss=dis_loss,
    )

    expected = (
        bd["loss/task"].float()
        + bd["loss/prediction"]
        + bd["loss/halting"]
        + bd["loss/amortisation"]
        + bd["loss/commitment"]
        + bd["loss/crystallisation"]
    )
    assert torch.allclose(total.float(), expected.float(), atol=1e-5), (
        f"Total {total.item():.6f} != sum of parts {expected.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Amortisation loss
# ---------------------------------------------------------------------------

def test_amortisation_loss_nonnegative():
    """Amortisation loss must be >= 0 (it is a sum of squared norms)."""
    fake_errors = [
        {"l0": torch.randn(2, 9, 32)},
        {"l0": torch.randn(2, 9, 32)},
    ]
    loss = amortisation_loss(fake_errors)
    assert loss.item() >= 0.0, f"amortisation_loss must be >= 0, got {loss.item()}"


def test_amortisation_loss_empty():
    """amortisation_loss on empty list returns 0."""
    loss = amortisation_loss([])
    assert loss.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end: run a forward pass and verify loss properties
# ---------------------------------------------------------------------------

def test_end_to_end_baseline_loss_no_nan():
    """Full forward + loss in baseline mode must produce finite, non-NaN loss."""
    config = _baseline_config()
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
        seg_loss, _ = loss_fn(logits=logits, labels=labels)
        total_loss = total_loss + seg_loss

    assert total_loss.isfinite(), "Baseline end-to-end loss must be finite"
    assert total_loss.item() >= 0.0, "Baseline end-to-end loss must be >= 0"


# ---------------------------------------------------------------------------
# Q-continue loss gating (Bug 3 fix, Session 9)
# ---------------------------------------------------------------------------

def test_continue_loss_disabled_by_default():
    """With use_continue_loss=False (default), halt loss = 0.5 * BCE(q_halt) only."""
    config = _baseline_config()  # use_continue_loss defaults to False
    assert not getattr(config, "use_continue_loss", False), "Expected False by default"
    loss_fn = CoralLoss(config)

    B, L, vocab = 2, 9, 10
    logits = torch.randn(B, L, vocab)
    labels = torch.randint(1, vocab, (B, L))
    q_halt = torch.randn(B)
    q_cont = torch.randn(B)  # provided but must be ignored when flag is False

    _, breakdown = loss_fn(
        logits=logits, labels=labels,
        q_halt_logits=q_halt, q_continue_logits=q_cont,
    )

    # Manually compute expected halt loss (halt term only, no continue)
    import torch.nn.functional as F_local
    mask = labels != -100
    preds = logits.argmax(dim=-1)
    seq_correct = ((preds == labels) & mask).sum(-1) == mask.sum(-1)
    halt_target = seq_correct.float()
    expected = 0.5 * F_local.binary_cross_entropy_with_logits(
        q_halt, halt_target, reduction="mean"
    )

    assert breakdown["loss/halting"].item() == pytest.approx(expected.item(), abs=1e-5), (
        "use_continue_loss=False: halt loss must equal 0.5 * BCE(q_halt, target) with "
        "no continue-loss contribution"
    )


def test_continue_loss_adds_to_halt_when_enabled():
    """With use_continue_loss=True, halt loss differs from the halt-only value."""
    from coral.config import ModelConfig as _MC
    config_on = _MC(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, timescale_base=2, K_max=2,
        use_predictive_coding=False, use_crystallisation=False,
        use_amort=False, lambda_amort=0.0,
        epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="baseline",
        use_continue_loss=True,
    )
    config_off = _baseline_config()  # use_continue_loss=False
    fn_on = CoralLoss(config_on)
    fn_off = CoralLoss(config_off)

    B, L, vocab = 2, 9, 10
    logits = torch.randn(B, L, vocab)
    labels = torch.randint(1, vocab, (B, L))
    q_halt = torch.randn(B)
    # Use extreme q_cont so the continue-BCE is clearly non-zero
    q_cont = torch.full((B,), 5.0)

    _, bd_off = fn_off(logits=logits, labels=labels, q_halt_logits=q_halt, q_continue_logits=q_cont)
    _, bd_on = fn_on(logits=logits, labels=labels, q_halt_logits=q_halt, q_continue_logits=q_cont)

    halt_off = bd_off["loss/halting"].item()
    halt_on = bd_on["loss/halting"].item()
    assert halt_on != pytest.approx(halt_off, abs=1e-5), (
        "use_continue_loss=True must produce a different halt loss than use_continue_loss=False "
        "when q_continue_logits are non-trivial"
    )


def test_end_to_end_full_mode_loss_no_nan():
    """Full forward + loss in full mode must produce finite, non-NaN loss."""
    config = _full_mode_config()
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

    assert total_loss.isfinite(), "Full mode end-to-end loss must be finite"
    assert total_loss.item() >= 0.0, "Full mode end-to-end loss must be >= 0"
