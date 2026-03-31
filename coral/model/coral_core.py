"""CORAL v4 — Main reasoning core.

Assembles backbone, level modules, predictive coding, and halting into the
complete amodal reasoning loop.

Inner loop structure for N=2, T=3:
    For each segment k in 1..K_max:
        # Top-down: level 2 predicts level 1
        mu_1 = prediction_net(z_2)

        # Level 1: T^1 = 3 inner steps (fast)
        for t in 0..2:
            h = backbone(W_up[1](z_1) + level_emb[1] + ts_emb[t] + z1_input_injection + mu_1_projected)
            z_1 = W_down[1](h)

        # Precision-weighted error
        eps_1, pi_1, xi_1, xi_1_up = pc_module(z_1, z_2)

        # Level 2: T^0 = 1 inner step (slow)
        h = backbone(W_up[2](z_2) + level_emb[2] + ts_emb[0] + xi_1_up)
        z_2 = W_down[2](h)

        # Halting check
        h_k, q_halt_logit, q_continue_logit = halting_net([z_1, z_2])

        # Detach for next segment (deep supervision boundary)
        z_1, z_2 = z_1.detach(), z_2.detach()

CRITICAL: all backbone applications within a segment are in the SAME
computation graph. The detach() only happens between segments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from coral.config import CoralConfig, ModelConfig
from coral.model.backbone import CoralBackbone, LevelEmbedding, TimescaleEmbedding
from coral.model.crystallisation import CrystallisationManager
from coral.model.halting import HaltingNetwork, should_halt
from coral.model.level_module import LevelStack
from coral.model.predictive_coding import PredictiveCodingModule


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class CoralOutput:
    """Outputs from one forward pass of CoralCore.

    Attributes:
        z_states:          List of final level states [B, L, d_l] per level.
        all_logits:        Logits from each segment (for deep supervision).
        halting_probs:     [B] halting probability at each segment.
        q_halt_logits:     [B] raw halt Q-value logit at each segment.
        q_continue_logits: [B] raw continue Q-value logit at each segment.
        pred_errors:       Prediction error stats per segment.
        precisions:        Precision stats per segment.
        num_segments:      Number of segments actually computed.
    """
    z_states: List[torch.Tensor] = field(default_factory=list)
    all_logits: List[torch.Tensor] = field(default_factory=list)
    halting_probs: List[torch.Tensor] = field(default_factory=list)
    q_halt_logits: List[torch.Tensor] = field(default_factory=list)
    q_continue_logits: List[torch.Tensor] = field(default_factory=list)
    pred_errors: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    precisions: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    num_segments: int = 0
    # Crystallisation outputs (empty when use_crystallisation=False)
    crystal_stats: List[Dict] = field(default_factory=list)
    commit_losses: List[torch.Tensor] = field(default_factory=list)
    dis_losses: List[torch.Tensor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CoralCore
# ---------------------------------------------------------------------------


class CoralCore(nn.Module):
    """The amodal reasoning core.

    Takes embedding vectors as input, returns refined embedding vectors.
    Operates as a multi-timescale recurrent hierarchy with precision-weighted
    predictive coding.

    Args:
        config: Full CoralConfig or ModelConfig.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_levels = config.n_levels
        self.level_dims = config.level_dims
        self.T = config.timescale_base
        self.K_max = config.K_max

        # Determine effective operating mode with backward compatibility.
        # Old-style configs set use_predictive_coding=True but do not set mode,
        # so they would get mode="baseline" by default.  We promote those to
        # "pc_only" so existing configs and tests keep working unchanged.
        if config.mode == "baseline" and config.use_predictive_coding:
            self.effective_mode = "pc_only"
        else:
            self.effective_mode = config.mode

        # Compute inner steps per level: level i (0-indexed) gets T^(n_levels-1-i) steps
        # Level 0 (fastest) = T^(n_levels-1) steps, Level n_levels-1 (slowest) = T^0=1 step
        self.level_steps = [
            self.T ** (self.n_levels - 1 - i) for i in range(self.n_levels)
        ]

        # Shared backbone (weight-shared across all levels and steps)
        self.backbone = CoralBackbone(config)

        # Level modules (up/down projections)
        self.levels = LevelStack(config)

        # Predictive coding modules (one per adjacent level pair)
        if config.use_predictive_coding:
            self.pc_modules = nn.ModuleList([
                PredictiveCodingModule(
                    dim_lower=config.level_dims[i],
                    dim_upper=config.level_dims[i + 1],
                    eps_min=config.epsilon_min,
                    momentum=getattr(config, "precision_momentum", 0.99),
                )
                for i in range(config.n_levels - 1)
            ])
        else:
            self.pc_modules = nn.ModuleList()

        # Level and timescale embeddings
        self.level_emb = LevelEmbedding(config.n_levels, config.backbone_dim)
        max_steps = self.T ** (self.n_levels - 1) + 4
        self.timescale_emb = TimescaleEmbedding(config.backbone_dim, max_steps=max_steps)

        # Halting network
        self.halting = HaltingNetwork(config)

        # Crystallisation manager (mode="full" + use_crystallisation=True only)
        if config.use_crystallisation:
            self.crystallisation_manager = CrystallisationManager(config)
        else:
            self.crystallisation_manager = None

        # Learnable scalar gate per level that controls conditioning strength.
        # Initialised to 0.01 so conditioning starts as a 1% nudge.
        # The model can increase the gate if the prediction proves useful.
        self.cond_gate = nn.ParameterList([
            nn.Parameter(torch.full((1,), 0.01)) for _ in range(config.n_levels)
        ])

        # Linear projection to initialise higher-level states from the level
        # below instead of zeros. Gives the prediction network a non-trivial
        # starting point from step 0 so predictive coding can engage earlier.
        self.z_init_proj = nn.ModuleList([
            nn.Linear(config.level_dims[i - 1], config.level_dims[i], bias=False)
            for i in range(1, config.n_levels)
        ])

    def _run_level(
        self,
        z: torch.Tensor,
        level_idx: int,
        n_steps: int,
        conditioning: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        input_injection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run backbone recursion for one level.

        The backbone processes the level state without prediction interference.
        After each backbone step, the conditioning signal is applied as a
        residual correction in d_l space:

            z_new  = project_down(backbone(project_up(z) + level_emb + ts_emb + input_injection))
            z      = z_new + gate * (conditioning - z)   # if conditioning given

        The residual (conditioning - z) represents what the conditioning signal
        expects minus what the state currently is.  When conditioning ≈ z the
        residual is near zero, so accurate predictions cause no interference.
        When conditioning diverges from z it actively steers the update.
        The learnable gate (init=1) controls correction strength per level.

        Args:
            z:              [B, L, d_l] — current level state
            level_idx:      0-based level index
            n_steps:        number of backbone applications
            conditioning:   [B, L, d_l] residual correction signal
                            (top-down prediction or error from below)
            attention_bias: Optional [L, L] float tensor added to attention
                            logits in every backbone step (v4.2 local bias).
            input_injection: Optional [B, L, backbone_dim] tensor added to
                            the backbone input at every inner step. Used to
                            re-inject the original encoder output so the
                            backbone always has access to the task constraints
                            (e.g. given digits in Sudoku) regardless of how
                            many segments have elapsed.

        Returns:
            Updated z: [B, L, d_l]
        """
        level_mod = self.levels[level_idx]
        device = z.device

        for t in range(n_steps):
            # Build backbone input from level state + positional embeddings only
            backbone_in = level_mod.project_up(z)
            backbone_in = backbone_in + self.level_emb(level_idx, device).unsqueeze(0).unsqueeze(0)
            backbone_in = backbone_in + self.timescale_emb(t).unsqueeze(0).unsqueeze(0)

            # Re-inject the original task input (given digits in Sudoku) so the
            # backbone always has access to the constraint signal at every step.
            if input_injection is not None:
                backbone_in = backbone_in + input_injection

            # Run backbone — no conditioning in the backbone input
            z_new = level_mod.project_down(
                self.backbone(backbone_in, attention_bias=attention_bias)
            )

            # Apply conditioning as a post-backbone residual correction in d_l space.
            # conditioning must be in d_l space (callers are responsible for this).
            if conditioning is not None:
                z = z_new + self.cond_gate[level_idx] * (conditioning - z)
            else:
                z = z_new

        return z

    def forward(
        self,
        z1_init: torch.Tensor,
        K_max: Optional[int] = None,
        training: bool = True,
        decode_fn: Optional[callable] = None,
        attention_masks: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> CoralOutput:
        """Run the full reasoning loop.

        Args:
            z1_init:         [B, L, d_1] — initial level-1 state from adapter
            K_max:           Override maximum segments (defaults to config.K_max)
            training:        Training vs eval mode (affects halting behaviour)
            decode_fn:       Optional callable([B, L, d_1]) → [B, L, vocab_size]
                             used to produce logits for deep supervision.
            attention_masks: Optional tuple of 3 binary [L, L] tensors
                             (same_row, same_col, same_box) from the adapter.
                             When provided and config.use_local_attention_bias
                             is True, the backbone's learned scalars are applied
                             to build a combined [L, L] attention bias used in
                             every backbone call this forward pass.
                             Pass None (default) to skip — backward compatible.

        Returns:
            CoralOutput with all intermediate states and metrics.
        """
        if K_max is None:
            K_max = self.K_max

        B, L, _ = z1_init.shape
        device = z1_init.device
        dtype = z1_init.dtype

        # Compute combined attention bias from structural masks (once per forward).
        # The 3 binary masks are static; only the learned scalars change during training.
        attn_bias: Optional[torch.Tensor] = None
        if attention_masks is not None and self.config.use_local_attention_bias:
            row_mask, col_mask, box_mask = attention_masks
            attn_bias = (
                self.backbone.row_bias * row_mask.to(device)
                + self.backbone.col_bias * col_mask.to(device)
                + self.backbone.box_bias * box_mask.to(device)
            )

        # Initialise level states. Higher levels are seeded from the level
        # below via z_init_proj so the prediction network has a non-trivial
        # starting point from segment 0 instead of dead zeros.
        z_states = [z1_init]
        for i in range(1, self.n_levels):
            z_init = self.z_init_proj[i - 1](z_states[i - 1])
            z_states.append(z_init)

        # Whether crystallisation is active this forward pass
        use_crys = (
            self.effective_mode == "full"
            and self.crystallisation_manager is not None
        )

        # Reset the convergence monitor for this batch (state is per-batch, not per-model)
        if use_crys:
            self.crystallisation_manager.monitor.reset(B, L, device)

        # Input re-injection signal: the original encoder output is added to the backbone
        # input at EVERY inner step in EVERY segment (matching TRM's approach).  This ensures
        # the backbone always has direct access to the task constraints (given digits in Sudoku)
        # rather than relying solely on the recurrent state to remember them.
        #
        # z1_init is [B, L, backbone_dim=512].  Since backbone_in is also at backbone_dim after
        # project_up, the shapes always match regardless of level depth.
        #
        # CRITICAL: z1_init is NOT detached between segments — gradients flow back to the
        # encoder (adapter) through every injection point, giving the embedding layers a
        # learning signal from all K_max segments, not just the first.
        input_signal: Optional[torch.Tensor] = z1_init

        output = CoralOutput()

        for seg in range(K_max):
            seg_eps: Dict[str, torch.Tensor] = {}
            seg_pi: Dict[str, torch.Tensor] = {}
            seg_crystal_stats: Dict = {}

            if self.effective_mode == "baseline":
                # ------------------------------------------------------------
                # Baseline mode: no predictive coding, level 0 only.
                # Run T inner steps at level 0; ignore all higher levels.
                # This is the clean no-PC path used for Phase 1 experiments.
                # ------------------------------------------------------------
                z_states[0] = self._run_level(
                    z_states[0], 0, self.T, conditioning=None, attention_bias=attn_bias,
                    input_injection=input_signal,
                )

            else:
                # effective_mode == "pc_only" or "full"
                # ----------------------------------------------------------------
                # Top-down pass: compute predictions for each level
                # ----------------------------------------------------------------
                predictions = [None] * self.n_levels  # predictions[i] = mu for level i
                if self.config.use_predictive_coding:
                    for i in range(self.n_levels - 2, -1, -1):  # from N-2 down to 0
                        predictions[i] = self.pc_modules[i].predict(z_states[i + 1])

                # ----------------------------------------------------------------
                # Crystallisation step (mode="full" only): detect convergence at
                # the START of each segment and snap newly converged heads to their
                # nearest codebook entry.  Runs BEFORE the backbone so crystallised
                # positions are already pinned when the inner loop begins.
                # ----------------------------------------------------------------
                if use_crys:
                    z_states[0], _crys_mask, seg_crystal_stats = (
                        self.crystallisation_manager.step(
                            z_states[0], z_prev=None, segment_idx=seg
                        )
                    )

                # ----------------------------------------------------------------
                # Bottom-up pass: recurrence + error propagation
                # CRITICAL: all backbone calls are in the SAME computation graph
                # ----------------------------------------------------------------
                error_signals = [None] * self.n_levels  # xi_up projected into each level

                # Process levels from fastest (0) to slowest (n_levels-1)
                for i in range(self.n_levels):
                    # Conditioning = top-down prediction + error from level below
                    cond = None
                    if predictions[i] is not None:
                        cond = predictions[i]
                    if i > 0 and error_signals[i] is not None:
                        if cond is not None:
                            cond = cond + error_signals[i]
                        else:
                            cond = error_signals[i]

                    # Run backbone recursion for this level
                    n_steps = self.level_steps[i]
                    z_states[i] = self._run_level(
                        z_states[i], i, n_steps, conditioning=cond, attention_bias=attn_bias,
                        input_injection=input_signal,
                    )

                    # Compute precision-weighted error and project upward
                    if self.config.use_predictive_coding and i < self.n_levels - 1:
                        mu, eps, pi, xi, xi_up = self.pc_modules[i](z_states[i], z_states[i + 1])

                        # Store for loss computation
                        seg_eps[f"level_{i}"] = eps
                        seg_pi[f"level_{i}"] = pi

                        # xi_up goes into level i+1's input next segment
                        error_signals[i + 1] = xi_up

                # ----------------------------------------------------------------
                # De-crystallisation check (mode="full"): if the backbone proposes
                # a value that drifts far from the frozen entry, unfreeze that head.
                # Must run AFTER the backbone, BEFORE the next crystallisation step.
                # ----------------------------------------------------------------
                if use_crys:
                    self.crystallisation_manager.monitor.check_decrystallisation(z_states[0])

            # ----------------------------------------------------------------
            # Crystallisation losses (accumulated once per segment).
            # Collected after the backbone + decrystal check so they reflect
            # the final state for this segment.
            # ----------------------------------------------------------------
            if use_crys:
                commit, dis = self.crystallisation_manager.get_losses()
                output.commit_losses.append(commit)
                output.dis_losses.append(dis)
            output.crystal_stats.append(seg_crystal_stats)

            # ----------------------------------------------------------------
            # Decode to logits for deep supervision (if decode_fn provided).
            #
            # TRAINING: decode from the raw backbone output (z_states[0]).
            #   Crystallised heads are already pinned at the start of the segment
            #   via crystal_manager.step(), so the backbone only updates active heads.
            #   Decoding the raw backbone output ensures gradients flow unobstructed.
            #
            # EVAL: enforce crystallised heads onto the backbone output before
            #   decoding, so crystallised positions always output their codebook value.
            # ----------------------------------------------------------------
            if decode_fn is not None:
                if training or not use_crys:
                    logits = decode_fn(z_states[0])
                else:
                    z_eval = self.crystallisation_manager.enforce_after_backbone(z_states[0])
                    logits = decode_fn(z_eval)
                output.all_logits.append(logits)

            # ----------------------------------------------------------------
            # Halting check
            # ----------------------------------------------------------------
            h_k, q_halt_logit, q_continue_logit = self.halting(z_states)
            output.halting_probs.append(h_k)
            output.q_halt_logits.append(q_halt_logit)
            output.q_continue_logits.append(q_continue_logit)
            output.pred_errors.append(seg_eps)
            output.precisions.append(seg_pi)
            output.num_segments = seg + 1

            # ----------------------------------------------------------------
            # Check for early halting
            # ----------------------------------------------------------------
            if not training and should_halt(
                h_k,
                threshold=self.config.halting_threshold,
                exploration_prob=0.0,
                training=False,
            ):
                break
            elif training and should_halt(
                h_k,
                threshold=self.config.halting_threshold,
                exploration_prob=self.config.halting_exploration_prob,
                training=True,
            ):
                break

            # ----------------------------------------------------------------
            # Detach between segments (deep supervision boundary)
            # CRITICAL: must detach or backprop goes through all K_max segments
            # ----------------------------------------------------------------
            z_states = [z.detach() for z in z_states]

        output.z_states = z_states
        return output
