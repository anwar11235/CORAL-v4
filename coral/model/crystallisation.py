"""CORAL v4.2 — Multi-headed semantic codebooks and convergence-driven crystallisation.

Component 4: MultiHeadedCodebook
    - H=8 independent codebook heads, each d_head=dim/H dimensions wide, M entries deep.
    - Codebooks are EMA-updated BUFFERS (not parameters) — they never see gradient steps.
    - Effective capacity: M^H composite states (~10^12 for M=32, H=8).

Component 5: ConvergenceMonitor
    - Tracks per-position, per-head state velocity across segments.
    - Crystallises heads whose velocity has stayed below τ_converge for ≥ N_stable segments.
    - De-crystallises heads whose frozen value drifts too far from what the backbone proposes.
    - Zero parameters, zero gradients — pure statistical monitoring.

Component 5 (manager): CrystallisationManager
    - Owns one MultiHeadedCodebook and one ConvergenceMonitor.
    - Orchestrates the crystallise → enforce → decrystalise cycle per segment.
    - Computes commitment and disentanglement losses for the training objective.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from coral.config import ModelConfig


# ---------------------------------------------------------------------------
# Simple k-means helper (used for codebook initialisation)
# ---------------------------------------------------------------------------


def _kmeans(
    data: torch.Tensor,
    k: int,
    n_iter: int = 100,
) -> torch.Tensor:
    """Offline k-means on CPU/GPU tensors.

    Args:
        data:   [N, d] float tensor of data points.
        k:      Number of centroids.
        n_iter: Iteration count.

    Returns:
        centroids: [k, d] float tensor.
    """
    N, d = data.shape
    # Random initialisation: pick k distinct data points
    perm = torch.randperm(N, device=data.device)[:k]
    centroids = data[perm].clone()

    for _ in range(n_iter):
        dists = torch.cdist(data, centroids)          # [N, k]
        assignments = dists.argmin(dim=1)             # [N]

        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = assignments == j
            if mask.any():
                new_centroids[j] = data[mask].mean(dim=0)
            else:
                new_centroids[j] = centroids[j]       # keep to avoid dead centroid
        centroids = new_centroids

    return centroids


# ---------------------------------------------------------------------------
# Component 4: Multi-headed semantic codebook
# ---------------------------------------------------------------------------


class MultiHeadedCodebook(nn.Module):
    """H independent codebook heads for factored discrete representation.

    The full d-dimensional state z is split into H equal chunks of d_head = d/H
    dimensions.  Each head maintains its own codebook of M entries.

    Codebooks are stored as BUFFERS (not nn.Parameters) and are updated via
    exponential moving average — never via gradient descent.

    Args:
        dim:              Full state dimension (must be divisible by n_heads).
        n_heads:          Number of independent codebook heads (H).
        entries_per_head: Codebook entries per head (M).
        ema_decay:        EMA decay rate for codebook updates (default 0.99).
    """

    def __init__(
        self,
        dim: int = 512,
        n_heads: int = 8,
        entries_per_head: int = 32,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.dim = dim
        self.n_heads = n_heads
        self.entries_per_head = entries_per_head
        self.d_head = dim // n_heads
        self.ema_decay = ema_decay

        # Codebook entries — BUFFERS, not parameters.
        # Shape [H, M, d_head]; initialised with small random values.
        self.register_buffer(
            "codebook",
            torch.randn(n_heads, entries_per_head, self.d_head) * 0.02,
        )
        # Usage count per entry; reset periodically in dead-code restart.
        self.register_buffer(
            "usage_count",
            torch.zeros(n_heads, entries_per_head),
        )

    # ------------------------------------------------------------------
    # Quantisation
    # ------------------------------------------------------------------

    def quantise(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise z to nearest codebook entries.

        Args:
            z:           [B, L, dim] — continuous state to quantise.
            temperature: Gumbel-Softmax temperature (only used when hard=False).
            hard:        True → hard straight-through nearest-neighbour.
                         False → Gumbel-Softmax soft assignment.

        Returns:
            z_q:               [B, L, dim]    — quantised state.
            indices:           [B, L, H]      — nearest entry index per head.
            per_head_distances:[B, L, H]      — L2 distance to nearest entry per head.
        """
        B, L, _ = z.shape
        z_h = z.view(B, L, self.n_heads, self.d_head)  # [B, L, H, d_head]

        indices_list = []
        dists_list = []
        z_q_list = []

        for h in range(self.n_heads):
            z_flat = z_h[:, :, h, :].reshape(B * L, self.d_head)  # [BL, d_head]
            c_h = self.codebook[h]                                  # [M, d_head]

            # Squared L2 distances: ||z - c||² = ||z||² + ||c||² - 2z·c^T
            dists = (
                z_flat.pow(2).sum(dim=1, keepdim=True)  # [BL, 1]
                + c_h.pow(2).sum(dim=1)                  # [M]
                - 2.0 * z_flat @ c_h.T                   # [BL, M]
            ).clamp(min=0.0)                              # [BL, M]

            if hard:
                idx_h = dists.argmin(dim=1)               # [BL]
                e_h = c_h[idx_h]                          # [BL, d_head]
                e_h = e_h.view(B, L, self.d_head)         # [B, L, d_head]
                z_h_view = z_h[:, :, h, :]               # [B, L, d_head]
                # Straight-through: forward passes e_h; backward passes through z
                z_q_h = z_h_view + (e_h - z_h_view).detach()

                min_dists = dists.gather(1, idx_h.unsqueeze(1)).squeeze(1)  # [BL]
            else:
                # Gumbel-Softmax soft assignment
                logits = -dists / max(temperature, 1e-8)   # [BL, M]
                weights = F.gumbel_softmax(logits, tau=temperature, hard=False)  # [BL, M]
                e_h = (weights.unsqueeze(2) * c_h.unsqueeze(0)).sum(dim=1)  # [BL, d_head]
                e_h = e_h.view(B, L, self.d_head)
                z_q_h = e_h

                idx_h = dists.argmin(dim=1)               # [BL] (hard argmax for tracking)
                min_dists = dists.gather(1, idx_h.unsqueeze(1)).squeeze(1)

            indices_list.append(idx_h.view(B, L))
            dists_list.append(min_dists.view(B, L))
            z_q_list.append(z_q_h)

        indices = torch.stack(indices_list, dim=2)         # [B, L, H]
        per_head_distances = torch.stack(dists_list, dim=2)  # [B, L, H]
        z_q = torch.stack(z_q_list, dim=2).view(B, L, self.dim)  # [B, L, dim]

        return z_q, indices, per_head_distances

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_ema(self, z: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook entries towards assigned states via EMA.

        Args:
            z:       [B, L, dim] — continuous states (detached internally).
            indices: [B, L, H]   — nearest-entry indices from quantise().
        """
        B, L, _ = z.shape
        z_h = z.detach().view(B, L, self.n_heads, self.d_head)
        M = self.entries_per_head

        for h in range(self.n_heads):
            z_flat = z_h[:, :, h, :].reshape(B * L, self.d_head)  # [BL, d_head]
            idx = indices[:, :, h].reshape(B * L)                  # [BL]

            # Per-entry assignment count
            counts = torch.bincount(idx, minlength=M).float()      # [M]

            # Sum of assigned states per entry
            sums = torch.zeros(M, self.d_head, device=z.device)
            sums.scatter_add_(0, idx.unsqueeze(1).expand(-1, self.d_head), z_flat)

            # EMA update for entries that received assignments
            has_assign = counts > 0
            if has_assign.any():
                assigned_mean = sums[has_assign] / counts[has_assign].unsqueeze(1)
                self.codebook[h][has_assign] = (
                    self.ema_decay * self.codebook[h][has_assign]
                    + (1.0 - self.ema_decay) * assigned_mean
                )
                self.usage_count[h] += counts

    # ------------------------------------------------------------------
    # Dead-code restart
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dead_code_restart(
        self,
        z_buffer: torch.Tensor,
        threshold: int = 0,
    ) -> int:
        """Replace stale codebook entries using recent states.

        Args:
            z_buffer: [N, dim] — pool of recent continuous states.
            threshold: Entries with usage_count <= threshold are replaced.

        Returns:
            Number of entries replaced (for monitoring).
        """
        B_buf = z_buffer.shape[0]
        z_h = z_buffer.detach().view(B_buf, self.n_heads, self.d_head)  # [N, H, d_head]
        M = self.entries_per_head
        replaced = 0

        for h in range(self.n_heads):
            dead = (self.usage_count[h] <= threshold).nonzero(as_tuple=True)[0]
            if dead.numel() == 0:
                continue
            # Sample randomly from the buffer for this head
            n_dead = dead.numel()
            sample_idx = torch.randint(0, B_buf, (n_dead,), device=z_buffer.device)
            self.codebook[h][dead] = z_h[sample_idx, h, :]
            self.usage_count[h][dead] = 0
            replaced += n_dead

        # Reset usage counts for the next monitoring window
        self.usage_count.zero_()
        return replaced

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def commitment_loss(
        self,
        z: torch.Tensor,
        z_q: torch.Tensor,
    ) -> torch.Tensor:
        """Per-head VQ commitment loss: ||z_h - sg(e_h)||².

        Encourages z to stay close to its assigned codebook entry.
        Gradient flows through z (straight-through); z_q is detached (sg).

        Args:
            z:   [B, L, dim] — continuous states.
            z_q: [B, L, dim] — quantised states (codebook entries, ST-gradient).

        Returns:
            Scalar commitment loss averaged over batch, positions, and heads.
        """
        B, L, _ = z.shape
        z_h  = z.view(B, L, self.n_heads, self.d_head)
        zq_h = z_q.view(B, L, self.n_heads, self.d_head)
        # sg(e_h) = zq_h.detach()
        per_head = (z_h - zq_h.detach()).pow(2).mean(dim=-1)  # [B, L, H]
        return per_head.mean()

    def disentanglement_loss(self) -> torch.Tensor:
        """Cross-head codebook correlation penalty.

        L_dis = Σ_{h1 ≠ h2} ||C_h1^T · C_h2||²_F / (M · M)

        Minimising this encourages heads to encode orthogonal (non-redundant)
        semantic directions.

        Returns:
            Scalar (non-negative).
        """
        M = self.entries_per_head
        total = torch.tensor(0.0, device=self.codebook.device)

        for h1 in range(self.n_heads):
            for h2 in range(h1 + 1, self.n_heads):
                C1 = self.codebook[h1]          # [M, d_head]
                C2 = self.codebook[h2]          # [M, d_head]
                # C1.T @ C2: [d_head, M] @ [M, d_head] = [d_head, d_head]
                cross = C1.T @ C2               # [d_head, d_head]
                total = total + cross.pow(2).sum() / (M * M)

        # Multiply by 2 for symmetry (sum over ALL h1≠h2, not just h1<h2)
        return total * 2.0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def initialise_from_kmeans(self, states: torch.Tensor, n_iter: int = 100) -> None:
        """Initialise codebook entries from k-means centroids.

        Args:
            states: [N, dim] — pool of collected reasoning states.
            n_iter: k-means iterations per head.
        """
        N, _ = states.shape
        z_h = states.view(N, self.n_heads, self.d_head)  # [N, H, d_head]

        for h in range(self.n_heads):
            data_h = z_h[:, h, :]                         # [N, d_head]
            centroids = _kmeans(data_h, self.entries_per_head, n_iter=n_iter)
            self.codebook[h] = centroids
        self.usage_count.zero_()

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def get_perplexity(self) -> torch.Tensor:
        """Per-head effective codebook usage (perplexity = exp(H(p))).

        A perplexity of M means all entries used equally; close to 1 means
        most entries are ignored (codebook collapse).

        Returns:
            [H] perplexity values.
        """
        M = self.entries_per_head
        total = self.usage_count.sum(dim=1, keepdim=True).clamp(min=1.0)
        p = self.usage_count / total                          # [H, M]
        entropy = -(p * (p + 1e-10).log()).sum(dim=1)        # [H]
        return entropy.exp()                                  # [H]

    def get_nearest_entries(self, h: int, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook entries by index for head h.

        Args:
            h:       Head index.
            indices: [*] integer indices.

        Returns:
            [*, d_head] codebook entries.
        """
        return self.codebook[h][indices]


# ---------------------------------------------------------------------------
# Component 5: Convergence monitor (no parameters — pure statistics)
# ---------------------------------------------------------------------------


class ConvergenceMonitor:
    """Tracks per-head, per-position state velocity and manages crystallisation.

    No learnable parameters.  All operations are outside the gradient graph.
    Call reset() at the start of each forward pass before the segment loop.

    Crystallisation trigger:
        velocity_h[b, i] < tau_converge   for n_stable consecutive segments
        → snap to nearest codebook entry and freeze.

    De-crystallisation trigger:
        drift = ||z_proposed_h - z_frozen_h||  > tau_decrystallise
        → unfreeze that (b, i, h) and use backbone's value.

    Args:
        n_heads:           Number of codebook heads H.
        d_head:            Dimension per head d_head = dim/H.
        tau_converge:      Velocity threshold for crystallisation trigger.
        tau_decrystallise: Drift threshold for de-crystallisation trigger.
        n_stable:          Consecutive low-velocity segments required.
    """

    def __init__(
        self,
        n_heads: int,
        d_head: int,
        tau_converge: float = 0.01,
        tau_decrystallise: float = 0.05,
        n_stable: int = 2,
    ) -> None:
        self.n_heads = n_heads
        self.d_head = d_head
        self.tau_converge = tau_converge
        self.tau_decrystallise = tau_decrystallise
        self.n_stable = n_stable

        # Per-batch tracking state (initialised by reset())
        self.z_prev: Optional[torch.Tensor] = None             # [B, L, dim]
        self.consecutive_converged: Optional[torch.Tensor] = None  # [B, L, H] long
        self.crystallised: Optional[torch.Tensor] = None       # [B, L, H] bool
        self.frozen_values: Optional[torch.Tensor] = None      # [B, L, H, d_head]

    def reset(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialise all tracking state for a new forward pass.

        Must be called before the segment loop, once per batch.

        Args:
            batch_size: B.
            seq_len:    L.
            device:     Target device.
        """
        H, d = self.n_heads, self.d_head
        self.z_prev = None
        self.consecutive_converged = torch.zeros(
            batch_size, seq_len, H, dtype=torch.long, device=device
        )
        self.crystallised = torch.zeros(
            batch_size, seq_len, H, dtype=torch.bool, device=device
        )
        self.frozen_values = torch.zeros(
            batch_size, seq_len, H, d, device=device
        )

    def update_and_crystallise(
        self,
        z_current: torch.Tensor,
        codebook: "MultiHeadedCodebook",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute velocity, update convergence counters, crystallise as needed.

        Args:
            z_current: [B, L, dim] — current segment's state (detached externally).
            codebook:  MultiHeadedCodebook instance for snapping newly crystallised heads.

        Returns:
            crystallisation_mask: [B, L, H] bool — all currently crystallised.
            newly_crystallised:   [B, L, H] bool — just crystallised this step.
        """
        B, L, dim = z_current.shape
        device = z_current.device

        # Ensure monitor state is on the right device
        if self.consecutive_converged is not None:
            self.consecutive_converged = self.consecutive_converged.to(device)
            self.crystallised = self.crystallised.to(device)
            self.frozen_values = self.frozen_values.to(device)

        newly_crystallised = torch.zeros(B, L, self.n_heads, dtype=torch.bool, device=device)

        if self.z_prev is None:
            # First segment: record state, no velocity yet
            self.z_prev = z_current.detach()
            return self.crystallised.clone(), newly_crystallised

        z_h = z_current.detach().view(B, L, self.n_heads, self.d_head)
        prev_h = self.z_prev.view(B, L, self.n_heads, self.d_head)

        # Per-head L2 velocity
        velocity = (z_h - prev_h).norm(dim=-1)  # [B, L, H]

        # Update consecutive-converged counter
        converged = velocity < self.tau_converge  # [B, L, H] bool
        self.consecutive_converged = torch.where(
            converged,
            self.consecutive_converged + 1,
            torch.zeros_like(self.consecutive_converged),
        )

        # Newly crystallised: reached n_stable AND not already frozen
        newly_crystallised = (
            (self.consecutive_converged >= self.n_stable) & ~self.crystallised
        )

        if newly_crystallised.any():
            # Snap newly crystallised heads to nearest codebook entry
            with torch.no_grad():
                _, indices, _ = codebook.quantise(z_current, hard=True)  # [B, L, H]
                for h in range(self.n_heads):
                    new_h = newly_crystallised[:, :, h]  # [B, L]
                    if new_h.any():
                        idx_h = indices[:, :, h]          # [B, L]
                        entries_h = codebook.codebook[h][idx_h]  # [B, L, d_head]
                        self.frozen_values[:, :, h, :] = torch.where(
                            new_h.unsqueeze(-1),
                            entries_h,
                            self.frozen_values[:, :, h, :],
                        )

        self.crystallised = self.crystallised | newly_crystallised
        self.z_prev = z_current.detach()

        return self.crystallised.clone(), newly_crystallised

    def check_decrystallisation(
        self,
        z_proposed: torch.Tensor,
        codebook_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Detect and execute de-crystallisation where drift exceeds threshold.

        Args:
            z_proposed:     [B, L, dim] — what the backbone would produce.
            codebook_values: Unused (reserved for future per-step codebook comparison).

        Returns:
            decrystallised: [B, L, H] bool — heads unfrozen this step.
        """
        device = z_proposed.device
        B, L, dim = z_proposed.shape

        if self.crystallised is None or not self.crystallised.any():
            return torch.zeros(B, L, self.n_heads, dtype=torch.bool, device=device)

        self.crystallised = self.crystallised.to(device)
        self.frozen_values = self.frozen_values.to(device)

        z_h = z_proposed.detach().view(B, L, self.n_heads, self.d_head)
        drift = (z_h - self.frozen_values).norm(dim=-1)  # [B, L, H]

        should_decrystal = (drift > self.tau_decrystallise) & self.crystallised

        if should_decrystal.any():
            self.crystallised = self.crystallised & ~should_decrystal

        return should_decrystal

    def enforce(
        self,
        z: torch.Tensor,
        codebook_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Overwrite crystallised heads with their frozen codebook values.

        Active (non-crystallised) heads are passed through unchanged.

        Args:
            z:               [B, L, dim] — backbone output to enforce upon.
            codebook_values: If provided, use these instead of self.frozen_values.

        Returns:
            z_enforced: [B, L, dim] with crystallised heads restored.
        """
        if self.crystallised is None or not self.crystallised.any():
            return z

        device = z.device
        B, L, dim = z.shape

        crystal = self.crystallised.to(device)                # [B, L, H]
        frozen = (
            codebook_values.to(device)
            if codebook_values is not None
            else self.frozen_values.to(device)
        )                                                     # [B, L, H, d_head]

        z_h = z.view(B, L, self.n_heads, self.d_head).clone()
        z_h = torch.where(crystal.unsqueeze(-1), frozen, z_h)
        return z_h.view(B, L, dim)


# ---------------------------------------------------------------------------
# Component 5 (manager): Crystallisation manager
# ---------------------------------------------------------------------------


class CrystallisationManager(nn.Module):
    """Orchestrates multi-headed codebooks and convergence-driven crystallisation.

    Owns one MultiHeadedCodebook and one ConvergenceMonitor.  Used by CoralCore
    (Session 5) to crystallise converged heads after each backbone pass.

    Args:
        config: ModelConfig with codebook_heads, codebook_entries_per_head,
                tau_converge, tau_decrystallise, n_stable, precision_momentum,
                and level_dims[0] for the state dimension.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        dim = config.level_dims[0]
        n_heads = config.codebook_heads
        entries = config.codebook_entries_per_head
        ema_decay = getattr(config, "precision_momentum", 0.99)

        self.codebook = MultiHeadedCodebook(
            dim=dim,
            n_heads=n_heads,
            entries_per_head=entries,
            ema_decay=ema_decay,
        )

        d_head = dim // n_heads
        self.monitor = ConvergenceMonitor(
            n_heads=n_heads,
            d_head=d_head,
            tau_converge=config.tau_converge,
            tau_decrystallise=config.tau_decrystallise,
            n_stable=config.n_stable,
        )

        # Internal step counter for dead-code restart scheduling
        self._step_count: int = 0
        # Small rolling buffer for dead-code restart (stored as Python list)
        self._z_buffer: list = []
        self._buffer_max: int = 512  # at most 512 flattened states in buffer

        # Store last computed quantities for get_losses() / get_stats()
        self._last_z: Optional[torch.Tensor] = None
        self._last_z_q: Optional[torch.Tensor] = None
        self._last_mask: Optional[torch.Tensor] = None

    def step(
        self,
        z: torch.Tensor,
        z_prev: Optional[torch.Tensor],
        segment_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Execute one crystallisation step for a segment.

        Called after each backbone segment.

        Args:
            z:            [B, L, dim] — current post-backbone state.
            z_prev:       [B, L, dim] or None — previous segment's state.
                          If provided, overrides the monitor's internal z_prev.
            segment_idx:  Current segment index (0-based).

        Returns:
            z_crystallised: [B, L, dim] — z with crystallised heads enforced.
            mask:           [B, L, H]   — bool mask of crystallised heads.
            stats:          dict with monitoring metrics.
        """
        # Allow caller to explicitly set z_prev in the monitor
        if z_prev is not None:
            self.monitor.z_prev = z_prev.detach()

        # Update convergence monitor and crystallise newly converged heads
        mask, newly = self.monitor.update_and_crystallise(z, self.codebook)

        # Codebook EMA update (training only: track during forward pass)
        if self.training:
            with torch.no_grad():
                z_q, indices, _ = self.codebook.quantise(z, hard=True)
                self.codebook.update_ema(z, indices)

                # Maintain z buffer for dead-code restart
                B, L, dim = z.shape
                z_flat = z.detach().reshape(B * L, dim)
                self._z_buffer.append(z_flat)
                if len(self._z_buffer) > self._buffer_max:
                    self._z_buffer.pop(0)
        else:
            z_q, indices, _ = self.codebook.quantise(z, hard=True)

        self._last_z = z
        self._last_z_q = z_q

        # Dead-code restart every 1000 steps
        self._step_count += 1
        if self.training and self._step_count % 1000 == 0 and self._z_buffer:
            buf = torch.cat(self._z_buffer[-32:], dim=0)  # last 32 batches
            self.codebook.dead_code_restart(buf, threshold=0)

        # Enforce crystallised heads
        z_enforced = self.monitor.enforce(z)
        self._last_mask = mask

        stats: Dict[str, torch.Tensor] = {
            "crystallisation_rate": mask.float().mean(),
            "newly_crystallised": newly.float().sum(),
            "per_head_rates": mask.float().mean(dim=(0, 1)),  # [H]
        }

        return z_enforced, mask, stats

    def enforce_after_backbone(self, z: torch.Tensor) -> torch.Tensor:
        """Enforce crystallisation constraints after each backbone application.

        Unlike step(), this does NOT update the monitor's velocity state.
        Used between inner steps of a segment.

        Args:
            z: [B, L, dim]

        Returns:
            z with crystallised heads restored.
        """
        return self.monitor.enforce(z)

    def get_losses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute commitment and disentanglement losses.

        Returns:
            commitment_loss:     scalar — ||z - sg(z_q)||² per head.
            disentanglement_loss: scalar — head orthogonality penalty.
        """
        if self._last_z is None or self._last_z_q is None:
            device = self.codebook.codebook.device
            zero = torch.tensor(0.0, device=device)
            return zero, zero

        commit = self.codebook.commitment_loss(self._last_z, self._last_z_q)
        dis = self.codebook.disentanglement_loss()
        return commit, dis

    def get_stats(self) -> Dict[str, torch.Tensor]:
        """Monitoring statistics for W&B logging.

        Returns:
            Dict with crystallisation_rate, per_head_rates, perplexity.
        """
        stats: Dict[str, torch.Tensor] = {}
        stats["perplexity"] = self.codebook.get_perplexity()  # [H]

        if self._last_mask is not None:
            stats["crystallisation_rate"] = self._last_mask.float().mean()
            stats["per_head_rates"] = self._last_mask.float().mean(dim=(0, 1))
        else:
            device = self.codebook.codebook.device
            stats["crystallisation_rate"] = torch.tensor(0.0, device=device)
            stats["per_head_rates"] = torch.zeros(self.config.codebook_heads, device=device)

        return stats
