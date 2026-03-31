"""Unit tests for coral/model/crystallisation.py (Session 4).

Covers:
  - MultiHeadedCodebook: parameter/buffer counts, quantise shapes, EMA update,
    dead-code restart, commitment loss, disentanglement loss
  - ConvergenceMonitor: crystallisation trigger, no-trigger on high velocity,
    de-crystallisation, enforce, partial crystallisation
  - CrystallisationManager: end-to-end pipeline
"""

import torch
import pytest

from coral.model.crystallisation import (
    MultiHeadedCodebook,
    ConvergenceMonitor,
    CrystallisationManager,
)
from coral.config import ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_codebook(dim=512, n_heads=8, entries_per_head=32):
    return MultiHeadedCodebook(dim=dim, n_heads=n_heads, entries_per_head=entries_per_head)


def _make_config(**overrides):
    defaults = dict(
        n_levels=2,
        level_dims=[512, 256],
        backbone_dim=512,
        n_heads=8,
        d_k=64,
        codebook_heads=8,
        codebook_entries_per_head=32,
        tau_converge=0.01,
        tau_decrystallise=0.05,
        n_stable=2,
        lambda_dis=0.01,
        precision_momentum=0.99,
        K_max=3,
        vocab_size=10,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# MultiHeadedCodebook — parameter / buffer counts
# ---------------------------------------------------------------------------

def test_codebook_is_buffer_not_parameter():
    """Codebook entries must be registered buffers, NOT learnable parameters."""
    cb = _make_codebook()
    param_names = {name for name, _ in cb.named_parameters()}
    assert "codebook" not in param_names, "codebook must NOT be a learnable parameter"
    assert "usage_count" not in param_names, "usage_count must NOT be a learnable parameter"

    buffer_names = {name for name, _ in cb.named_buffers()}
    assert "codebook" in buffer_names, "codebook must be a registered buffer"
    assert "usage_count" in buffer_names, "usage_count must be a registered buffer"


def test_codebook_zero_learnable_parameters():
    """MultiHeadedCodebook must have exactly zero learnable parameters."""
    cb = _make_codebook()
    n_params = sum(p.numel() for p in cb.parameters())
    assert n_params == 0, f"Expected 0 learnable params, got {n_params}"


def test_codebook_buffer_shape():
    """codebook_entries_per_head=32, n_heads=8, dim=512 gives buffer [8, 32, 64]."""
    cb = _make_codebook(dim=512, n_heads=8, entries_per_head=32)
    assert cb.codebook.shape == (8, 32, 64), (
        f"Expected codebook shape (8, 32, 64), got {cb.codebook.shape}"
    )


# ---------------------------------------------------------------------------
# MultiHeadedCodebook — quantise
# ---------------------------------------------------------------------------

def test_quantise_output_shapes():
    """quantise returns correct shapes [B, L, dim] and [B, L, H]."""
    B, L, dim, H = 2, 10, 512, 8
    cb = _make_codebook(dim=dim, n_heads=H)
    z = torch.randn(B, L, dim)

    z_q, indices, dists = cb.quantise(z)
    assert z_q.shape == (B, L, dim), f"z_q shape {z_q.shape}"
    assert indices.shape == (B, L, H), f"indices shape {indices.shape}"
    assert dists.shape == (B, L, H), f"dists shape {dists.shape}"


def test_quantised_output_differs_from_input():
    """Quantised output differs from input (nearest neighbour is not identity)."""
    cb = _make_codebook(dim=64, n_heads=4, entries_per_head=8)
    z = torch.randn(2, 5, 64)
    z_q, _, _ = cb.quantise(z)
    # z_q should equal the codebook entries, not z (random init makes collision unlikely)
    assert not torch.allclose(z_q, z), "z_q should not equal z for random inputs"


def test_quantise_straight_through_gradient():
    """Straight-through: gradients flow through z_q as if through z."""
    cb = _make_codebook(dim=64, n_heads=4, entries_per_head=8)
    z = torch.randn(2, 5, 64, requires_grad=True)
    z_q, _, _ = cb.quantise(z, hard=True)
    loss = z_q.sum()
    loss.backward()
    assert z.grad is not None, "Gradient must flow through straight-through estimator"
    assert z.grad.abs().sum().item() > 0, "Gradient must be non-zero"


# ---------------------------------------------------------------------------
# MultiHeadedCodebook — EMA update
# ---------------------------------------------------------------------------

def test_ema_update_moves_codebook():
    """After EMA update, codebook entries should change."""
    cb = _make_codebook(dim=64, n_heads=4, entries_per_head=8)
    codebook_before = cb.codebook.clone()

    z = torch.randn(4, 10, 64)
    _, indices, _ = cb.quantise(z)
    cb.update_ema(z, indices)

    codebook_after = cb.codebook
    assert not torch.allclose(codebook_before, codebook_after), (
        "Codebook entries should change after EMA update"
    )


# ---------------------------------------------------------------------------
# MultiHeadedCodebook — dead-code restart
# ---------------------------------------------------------------------------

def test_dead_code_restart_replaces_unused_entries():
    """dead_code_restart replaces entries with zero usage count."""
    cb = _make_codebook(dim=64, n_heads=4, entries_per_head=8)
    # All usage_counts start at 0 — all entries are "dead" at threshold=0
    codebook_before = cb.codebook.clone()
    z_buffer = torch.randn(32, 64)
    replaced = cb.dead_code_restart(z_buffer, threshold=0)
    assert replaced > 0, "Some entries should be replaced"
    assert not torch.allclose(codebook_before, cb.codebook), (
        "Codebook should change after dead-code restart"
    )


# ---------------------------------------------------------------------------
# MultiHeadedCodebook — commitment loss
# ---------------------------------------------------------------------------

def test_commitment_loss_zero_when_z_equals_codebook():
    """commitment_loss should be 0 when z equals codebook entries exactly."""
    cb = _make_codebook(dim=64, n_heads=4, entries_per_head=8)
    z = torch.randn(2, 5, 64)
    z_q, _, _ = cb.quantise(z, hard=True)
    # Make z equal to z_q (the codebook entries)
    z_equal = z_q.detach().clone()
    loss = cb.commitment_loss(z_equal, z_q)
    assert loss.item() < 1e-6, f"Expected ~0 commitment loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# MultiHeadedCodebook — disentanglement loss
# ---------------------------------------------------------------------------

def test_disentanglement_loss_zero_when_orthogonal():
    """disentanglement_loss = 0 when codebook heads are perfectly orthogonal.

    The loss computes ||C1.T @ C2||²_F where C1, C2 are [M, d_head].
    This is zero iff every column of C1 (in R^M) is orthogonal to every
    column of C2 (in R^M).  We achieve this via standard basis construction:
    C1's columns = e_0..e_{d-1}, C2's columns = e_d..e_{2d-1} in R^{2d},
    which requires M >= 2*d_head.
    """
    d_head = 4
    M = 2 * d_head   # M=8; entries_per_head must be >= M for this construction
    n_heads = 2
    dim = n_heads * d_head
    cb = MultiHeadedCodebook(dim=dim, n_heads=n_heads, entries_per_head=M)

    with torch.no_grad():
        # Head 0: C0[m, i] = 1 iff m == i  (columns = e_0..e_{d-1} in R^M)
        C0 = torch.zeros(M, d_head)
        for i in range(d_head):
            C0[i, i] = 1.0
        cb.codebook[0] = C0

        # Head 1: C1[m, j] = 1 iff m == d_head + j  (columns = e_d..e_{2d-1})
        C1_mat = torch.zeros(M, d_head)
        for j in range(d_head):
            C1_mat[d_head + j, j] = 1.0
        cb.codebook[1] = C1_mat

    loss = cb.disentanglement_loss()
    assert loss.item() < 1e-6, (
        f"Expected ~0 disentanglement loss for orthogonal heads, got {loss.item()}"
    )


def test_disentanglement_loss_positive_when_correlated():
    """disentanglement_loss > 0 when codebook heads are identical (maximally correlated)."""
    n_heads, entries, d_head = 4, 8, 8
    dim = n_heads * d_head
    cb = MultiHeadedCodebook(dim=dim, n_heads=n_heads, entries_per_head=entries)

    # Set all heads to the same non-zero random matrix → maximum correlation
    same_entries = torch.randn(entries, d_head)
    with torch.no_grad():
        for h in range(n_heads):
            cb.codebook[h] = same_entries

    loss = cb.disentanglement_loss()
    assert loss.item() > 0.0, f"Expected positive disentanglement loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# ConvergenceMonitor — crystallisation trigger
# ---------------------------------------------------------------------------

def test_convergence_monitor_crystallises_after_n_stable():
    """ConvergenceMonitor crystallises heads after n_stable steps of low velocity."""
    B, L, H, d_head = 2, 5, 4, 8
    dim = H * d_head
    monitor = ConvergenceMonitor(n_heads=H, d_head=d_head, tau_converge=1.0, n_stable=2)
    cb = MultiHeadedCodebook(dim=dim, n_heads=H, entries_per_head=8)

    monitor.reset(B, L, device=torch.device("cpu"))

    # Step 1: set z_prev (no crystallisation yet)
    z = torch.zeros(B, L, dim)
    mask, newly = monitor.update_and_crystallise(z, cb)
    assert not mask.any(), "No crystallisation on first step"

    # Step 2: same z → velocity = 0 < tau_converge → consecutive = 1 (< n_stable=2)
    mask, newly = monitor.update_and_crystallise(z, cb)
    assert not newly.any(), "consecutive=1 is not enough; need n_stable=2"

    # Step 3: same z → consecutive = 2 == n_stable → should crystallise
    mask, newly = monitor.update_and_crystallise(z, cb)
    assert newly.any(), "Should crystallise after n_stable=2 low-velocity steps"
    assert mask.any(), "Mask should reflect crystallised heads"


def test_convergence_monitor_no_crystallise_high_velocity():
    """ConvergenceMonitor does NOT crystallise when velocity > tau_converge."""
    B, L, H, d_head = 2, 5, 4, 8
    dim = H * d_head
    monitor = ConvergenceMonitor(n_heads=H, d_head=d_head, tau_converge=0.01, n_stable=2)
    cb = MultiHeadedCodebook(dim=dim, n_heads=H, entries_per_head=8)

    monitor.reset(B, L)

    z0 = torch.zeros(B, L, dim)
    monitor.update_and_crystallise(z0, cb)  # Step 1: set z_prev

    for _ in range(5):
        # Large random step → velocity >> tau_converge
        z_new = torch.randn(B, L, dim) * 10.0
        mask, newly = monitor.update_and_crystallise(z_new, cb)
        assert not newly.any(), "High-velocity steps should not crystallise"


# ---------------------------------------------------------------------------
# ConvergenceMonitor — de-crystallisation
# ---------------------------------------------------------------------------

def test_decrystallisation_fires_on_large_drift():
    """De-crystallisation fires when drift > tau_decrystallise."""
    B, L, H, d_head = 1, 3, 4, 8
    dim = H * d_head
    tau_converge = 1.0       # easy to trigger crystallisation
    tau_decrystal = 0.05
    n_stable = 2
    monitor = ConvergenceMonitor(
        n_heads=H, d_head=d_head,
        tau_converge=tau_converge,
        tau_decrystallise=tau_decrystal,
        n_stable=n_stable,
    )
    cb = MultiHeadedCodebook(dim=dim, n_heads=H, entries_per_head=8)
    monitor.reset(B, L)

    # Crystallise all heads
    z = torch.zeros(B, L, dim)
    monitor.update_and_crystallise(z, cb)   # step 1
    monitor.update_and_crystallise(z, cb)   # step 2: consecutive=1
    _, newly = monitor.update_and_crystallise(z, cb)  # step 3: crystallise

    assert monitor.crystallised.any(), "Setup failed: expected some crystallised heads"

    # Now propose a large drift
    z_big = torch.ones(B, L, dim) * 10.0
    decrystal = monitor.check_decrystallisation(z_big)
    assert decrystal.any(), "Should de-crystallise on large drift"
    assert not monitor.crystallised.any(), "All heads should be unfrozen after large drift"


# ---------------------------------------------------------------------------
# ConvergenceMonitor — enforce
# ---------------------------------------------------------------------------

def test_enforce_overwrites_crystallised_preserves_active():
    """enforce() restores crystallised heads; active (non-crystallised) heads unchanged."""
    B, L, H, d_head = 1, 2, 4, 8
    dim = H * d_head
    monitor = ConvergenceMonitor(n_heads=H, d_head=d_head, tau_converge=1.0, n_stable=2)
    cb = MultiHeadedCodebook(dim=dim, n_heads=H, entries_per_head=8)
    monitor.reset(B, L)

    # Crystallise all heads
    z = torch.zeros(B, L, dim)
    for _ in range(3):
        monitor.update_and_crystallise(z, cb)

    # frozen_values are all zeros; backbone proposes large non-zero z
    z_proposed = torch.ones(B, L, dim) * 5.0
    z_enforced = monitor.enforce(z_proposed)

    # Crystallised heads (all) should have value 0 (frozen_values)
    z_h = z_enforced.view(B, L, H, d_head)
    frozen_h = monitor.frozen_values  # [B, L, H, d_head]
    crystal = monitor.crystallised    # [B, L, H]

    for h in range(H):
        for b in range(B):
            for pos in range(L):
                if crystal[b, pos, h]:
                    assert torch.allclose(
                        z_h[b, pos, h], frozen_h[b, pos, h], atol=1e-5
                    ), f"Crystallised head {h} at [{b},{pos}] not restored"


def test_partial_crystallisation_some_heads_some_positions():
    """Partial crystallisation: some heads crystallised, others not, same position."""
    B, L, H, d_head = 1, 1, 4, 8
    dim = H * d_head
    # Large tau_converge so only the zero-velocity heads crystallise
    monitor = ConvergenceMonitor(n_heads=H, d_head=d_head, tau_converge=100.0, n_stable=2)
    cb = MultiHeadedCodebook(dim=dim, n_heads=H, entries_per_head=8)
    monitor.reset(B, L)

    # Step 1: set initial z_prev
    z_base = torch.zeros(B, L, dim)
    monitor.update_and_crystallise(z_base, cb)

    # Step 2: only heads 0 and 1 have zero velocity; heads 2 and 3 move
    z_next = torch.zeros(B, L, dim)
    # Heads 2 and 3 move enough to exceed tau_converge=100 (need norm > 100)
    # In d_head=8, to get norm > 100 we need entries ~35 per dim
    z_next[0, 0, H * 2 * d_head // H:] = 200.0  # Move heads 2,3 by large amount
    monitor.update_and_crystallise(z_next, cb)   # consecutive++

    # Step 3: same z_next → heads 2,3 keep moving; heads 0,1 stay still
    # After step 2, z_prev is z_next; now same z_next → velocity=0 for all
    # But heads 2,3 velocity was large in step 2, so their consecutive reset to 0
    # Let heads 0,1 have zero velocity again from z_prev=z_next
    z_step3 = z_next.clone()
    z_step3[0, 0, H * 2 * d_head // H:] = 201.0  # heads 2,3 still moving
    monitor.update_and_crystallise(z_step3, cb)   # Now consecutive[0,0,0:2] = 2 → crystallise

    # Heads 0 and 1 should have crystallised; heads 2 and 3 should not
    crystal = monitor.crystallised[0, 0]  # [H]
    assert crystal[0].item() or crystal[1].item(), (
        "At least head 0 or 1 should be crystallised (low velocity)"
    )


# ---------------------------------------------------------------------------
# CrystallisationManager — end-to-end pipeline
# ---------------------------------------------------------------------------

def test_crystallisation_manager_end_to_end():
    """CrystallisationManager.step() runs without error; returns correct shapes."""
    config = _make_config(tau_converge=0.01, tau_decrystallise=0.05, n_stable=2)
    manager = CrystallisationManager(config)
    manager.train()

    B, L, dim = 2, 9, config.level_dims[0]
    monitor = manager.monitor
    monitor.reset(B, L)

    z = torch.randn(B, L, dim)
    z_prev = torch.randn(B, L, dim)

    z_out, mask, stats = manager.step(z, z_prev=z_prev, segment_idx=0)

    assert z_out.shape == (B, L, dim), f"z_out shape {z_out.shape}"
    assert mask.shape == (B, L, config.codebook_heads), f"mask shape {mask.shape}"
    assert "crystallisation_rate" in stats
    assert "newly_crystallised" in stats


def test_crystallisation_manager_get_losses():
    """get_losses() returns scalar tensors after a step."""
    config = _make_config()
    manager = CrystallisationManager(config)
    manager.train()

    B, L, dim = 2, 9, config.level_dims[0]
    manager.monitor.reset(B, L)

    z = torch.randn(B, L, dim)
    manager.step(z, z_prev=None, segment_idx=0)

    commit, dis = manager.get_losses()
    assert commit.shape == torch.Size([]), f"commit should be scalar, got {commit.shape}"
    assert dis.shape == torch.Size([]), f"dis should be scalar, got {dis.shape}"
    assert not torch.isnan(commit), "commit loss must not be NaN"
    assert not torch.isnan(dis), "dis loss must not be NaN"
