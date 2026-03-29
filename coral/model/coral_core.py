"""CORAL v4 — Main reasoning core.

Assembles backbone, level modules, predictive coding, and halting into the
complete amodal reasoning loop.

Inner loop structure for N=2, T=3:
    For each segment k in 1..K_max:
        # Top-down: level 2 predicts level 1
        mu_1 = prediction_net(z_2)

        # Level 1: T^1 = 3 inner steps (fast)
        for t in 0..2:
            h = backbone(W_up[1](z_1) + level_emb[1] + ts_emb[t] + mu_1_projected)
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
        final_precision:   Level-0 precision vector from the last segment
                           [B, L, d_0]; used for logging and precision-gated
                           decode. None when use_predictive_coding is False.
    """
    z_states: List[torch.Tensor] = field(default_factory=list)
    all_logits: List[torch.Tensor] = field(default_factory=list)
    halting_probs: List[torch.Tensor] = field(default_factory=list)
    q_halt_logits: List[torch.Tensor] = field(default_factory=list)
    q_continue_logits: List[torch.Tensor] = field(default_factory=list)
    pred_errors: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    precisions: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    num_segments: int = 0
    final_precision: Optional[torch.Tensor] = None


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

        # Learnable scalar gate per level that controls conditioning strength.
        # Initialised to 1.0 (no-op) so training starts with the full signal.
        # The model can learn to amplify or attenuate conditioning per level.
        self.cond_gate = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(config.n_levels)
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
    ) -> torch.Tensor:
        """Run backbone recursion for one level.

        Args:
            z:           [B, L, d_l] — current level state
            level_idx:   0-based level index
            n_steps:     number of backbone applications
            conditioning: [B, L, d_l] or [B, L, d_backbone] additive signal
                         (top-down prediction or error from below)

        Returns:
            Updated z: [B, L, d_l]
        """
        level_mod = self.levels[level_idx]
        device = z.device

        for t in range(n_steps):
            # Project up to backbone dim
            backbone_in = level_mod.project_up(z)

            # Add level embedding
            le = self.level_emb(level_idx, device)  # [d_backbone]
            backbone_in = backbone_in + le.unsqueeze(0).unsqueeze(0)

            # Add timescale embedding
            te = self.timescale_emb(t)  # [d_backbone]
            backbone_in = backbone_in + te.unsqueeze(0).unsqueeze(0)

            # Add conditioning signal (top-down prediction or error)
            if conditioning is not None:
                # conditioning may be in d_l space; project up if needed
                if conditioning.shape[-1] != self.config.backbone_dim:
                    cond_up = level_mod.project_up(conditioning)
                else:
                    cond_up = conditioning
                backbone_in = backbone_in + self.cond_gate[level_idx] * cond_up

            # Run backbone
            backbone_out = self.backbone(backbone_in)

            # Project down to level dim
            z = level_mod.project_down(backbone_out)

        return z

    def forward(
        self,
        z1_init: torch.Tensor,
        K_max: Optional[int] = None,
        training: bool = True,
        decode_fn: Optional[callable] = None,
    ) -> CoralOutput:
        """Run the full reasoning loop.

        Args:
            z1_init:   [B, L, d_1] — initial level-1 state from adapter
            K_max:     Override maximum segments (defaults to config.K_max)
            training:  Training vs eval mode (affects halting behaviour)
            decode_fn: Optional callable([B, L, d_1]) → [B, L, vocab_size]
                       used to produce logits for deep supervision.

        Returns:
            CoralOutput with all intermediate states and metrics.
        """
        if K_max is None:
            K_max = self.K_max

        B, L, _ = z1_init.shape
        device = z1_init.device
        dtype = z1_init.dtype

        # Initialise level states. Higher levels are seeded from the level
        # below via z_init_proj so the prediction network has a non-trivial
        # starting point from segment 0 instead of dead zeros.
        z_states = [z1_init]
        for i in range(1, self.n_levels):
            z_init = self.z_init_proj[i - 1](z_states[i - 1])
            z_states.append(z_init)

        output = CoralOutput()

        for seg in range(K_max):
            # ----------------------------------------------------------------
            # Top-down pass: compute predictions for each level
            # ----------------------------------------------------------------
            predictions = [None] * self.n_levels  # predictions[i] = mu for level i
            if self.config.use_predictive_coding:
                for i in range(self.n_levels - 2, -1, -1):  # from N-2 down to 0
                    predictions[i] = self.pc_modules[i].predict(z_states[i + 1])

            # ----------------------------------------------------------------
            # Bottom-up pass: recurrence + error propagation
            # CRITICAL: all backbone calls are in the SAME computation graph
            # ----------------------------------------------------------------
            seg_eps: Dict[str, torch.Tensor] = {}
            seg_pi: Dict[str, torch.Tensor] = {}
            error_signals = [None] * self.n_levels  # xi_up projected into each level

            # Process levels from fastest (0) to slowest (n_levels-1)
            for i in range(self.n_levels):
                # Conditioning = top-down prediction + error from level below
                cond = None
                if predictions[i] is not None:
                    cond = predictions[i]
                if i > 0 and error_signals[i] is not None:
                    if cond is not None:
                        # error_signals[i] is already in d_{i} space
                        # Need to project to d_{i} for adding to cond
                        cond = cond + error_signals[i]
                    else:
                        cond = error_signals[i]

                # Run backbone recursion for this level
                n_steps = self.level_steps[i]
                z_states[i] = self._run_level(z_states[i], i, n_steps, conditioning=cond)

                # Compute precision-weighted error and project upward
                if self.config.use_predictive_coding and i < self.n_levels - 1:
                    mu, eps, pi, xi, xi_up = self.pc_modules[i](z_states[i], z_states[i + 1])

                    # Store for loss computation
                    seg_eps[f"level_{i}"] = eps
                    seg_pi[f"level_{i}"] = pi

                    # xi_up goes into level i+1's input next segment
                    error_signals[i + 1] = xi_up

            # ----------------------------------------------------------------
            # Decode to logits for deep supervision (if decode_fn provided).
            # When use_predictive_coding is True, gate z_states[0] by the
            # level-0 precision vector so task-loss gradients flow through
            # precision — dimensions where precision is wrong get corrected
            # by the task loss, not just by the regulariser.
            # ----------------------------------------------------------------
            if decode_fn is not None:
                z_to_decode = z_states[0]
                if self.config.use_predictive_coding and "level_0" in seg_pi:
                    z_to_decode = seg_pi["level_0"] * z_states[0]
                logits = decode_fn(z_to_decode)
                output.all_logits.append(logits)
                output.final_precision = seg_pi.get("level_0")

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
