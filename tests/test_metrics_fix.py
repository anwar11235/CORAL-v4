"""Tests for the metrics reporting fix in TrainerV4.train_step (Bug 2 fix, Session 9).

Before the fix: metrics['loss/total'] was set to total_loss (sum over segments)
but then immediately overwritten by all_breakdowns[-1]['loss/total'] (last
segment's loss only), making it impossible to observe the real training signal.

After the fix:
  metrics['loss/total']          = sum of all-segment losses (what backward() saw)
  metrics['loss/last_seg_total'] = last segment's standalone loss (was incorrectly
                                   reported as 'loss/total' before)
  metrics['loss/per_segment_avg'] = loss/total / num_segments (for easy monitoring)
"""

import pytest
import torch

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.training.losses import CoralLoss
from coral.training.trainer import TrainerV4


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_trainer(K_max: int = 3) -> TrainerV4:
    config = ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, timescale_base=2, K_max=K_max,
        use_predictive_coding=False, use_crystallisation=False,
        use_amort=False, lambda_amort=0.0,
        epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="baseline",
    )
    coral_config = CoralConfig(model=config, device="cpu")
    coral_config.training.precision = "float32"
    coral_config.training.gradient_clip = 1.0

    adapter = GridAdapter(coral_config, vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(config)
    loss_fn = CoralLoss(config)
    return TrainerV4(adapter, core, loss_fn, coral_config)


def _make_batch():
    return {
        "inputs": torch.randint(0, 10, (2, 9)),
        "labels": torch.randint(1, 10, (2, 9)),
    }


# ---------------------------------------------------------------------------
# Key existence checks
# ---------------------------------------------------------------------------

def test_metrics_contains_required_keys():
    """train_step must return all three loss keys."""
    trainer = _make_trainer(K_max=3)
    metrics = trainer.train_step(_make_batch())

    assert "loss/total" in metrics, "metrics must contain 'loss/total'"
    assert "loss/last_seg_total" in metrics, "metrics must contain 'loss/last_seg_total'"
    assert "loss/per_segment_avg" in metrics, "metrics must contain 'loss/per_segment_avg'"
    assert "train/num_segments" in metrics, "metrics must contain 'train/num_segments'"


# ---------------------------------------------------------------------------
# Correctness of loss/total (must be the SUMMED loss)
# ---------------------------------------------------------------------------

def test_loss_total_equals_per_segment_avg_times_num_segments():
    """loss/total must equal per_segment_avg * num_segments (exact relationship)."""
    trainer = _make_trainer(K_max=3)
    metrics = trainer.train_step(_make_batch())

    n = metrics["train/num_segments"]
    expected_total = metrics["loss/per_segment_avg"] * n
    assert abs(metrics["loss/total"] - expected_total) < 1e-5, (
        f"loss/total ({metrics['loss/total']:.6f}) must equal "
        f"per_segment_avg * num_segments ({expected_total:.6f})"
    )


def test_loss_total_greater_than_last_seg_for_multiple_segments():
    """With K_max=3, summed total must exceed the last-segment-only total."""
    trainer = _make_trainer(K_max=3)
    metrics = trainer.train_step(_make_batch())

    n = metrics["train/num_segments"]
    if n >= 2:
        # Sum of N≥2 positive segment losses must exceed any single segment's loss
        assert metrics["loss/total"] > metrics["loss/last_seg_total"], (
            f"loss/total ({metrics['loss/total']:.4f}) should exceed "
            f"loss/last_seg_total ({metrics['loss/last_seg_total']:.4f}) "
            f"when {n} segments were computed. "
            "If these are equal the sum is not being computed correctly."
        )


# ---------------------------------------------------------------------------
# Regression: ensure the old collision no longer occurs
# ---------------------------------------------------------------------------

def test_loss_total_is_not_overwritten_by_breakdown():
    """The sum over segments must not be replaced by a single-segment value.

    This is the specific regression test for the original bug: breakdown[-1]
    contains 'loss/total' which used to silently overwrite the real total.
    """
    trainer = _make_trainer(K_max=3)
    metrics = trainer.train_step(_make_batch())

    n = metrics["train/num_segments"]
    per_seg = metrics["loss/per_segment_avg"]

    # If the bug were still present, loss/total would ≈ per_seg (single segment)
    # rather than ≈ per_seg * n (sum). Check the sum relationship holds.
    assert abs(metrics["loss/total"] - per_seg * n) < 1e-4, (
        "loss/total appears to be a single-segment value rather than the sum. "
        "The breakdown collision may have been reintroduced."
    )
