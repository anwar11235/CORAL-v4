"""Tests for Session N7: unified Pareto and training-step precision logging.

Verification criteria:
  - Old key 'precision_raw_stat/level{i}_mean' must NOT appear in any
    metrics dict produced by train_step.  New key
    'precision_ema_var/level{i}_mean' MUST appear.
  - prediction_error/level{i}_mean from train_step equals
    ev.sqrt().mean() (EMA-variance RMS proxy) — not eps.abs().mean().
  - _log_eval_diagnostics reads the same running_precision buffers
    as train_step, so values are consistent after any forward pass.
"""

import sys
import os
import torch

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.training.losses import CoralLoss
from coral.training.trainer import TrainerV4

# Make scripts/ importable so we can call _log_eval_diagnostics directly.
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from train import _log_eval_diagnostics  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_pc_trainer() -> TrainerV4:
    """N=2 pc_only trainer on CPU."""
    config = ModelConfig(
        n_levels=2, level_dims=[64, 32], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, timescale_base=2, K_max=2,
        use_predictive_coding=True, use_crystallisation=False,
        use_amort=False, lambda_amort=0.0,
        epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="pc_only",
        use_consolidation_step=False,
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
# Test 1 — _log_eval_diagnostics reads from running_precision buffers
# ---------------------------------------------------------------------------

def test_log_eval_diagnostics_reads_running_precision_buffers():
    """_log_eval_diagnostics must read the same running_precision buffers
    as train_step, so precision values are consistent after any forward pass.

    After train_step updates the EMA buffers, calling _log_eval_diagnostics
    (which just reads those buffers) must return the same precision mean as
    train_step reported.  No second forward pass occurs.
    """
    torch.manual_seed(0)
    trainer = _make_pc_trainer()
    batch = _make_batch()

    train_metrics = trainer.train_step(batch)
    train_pi_mean = train_metrics["precision/level0_mean"]

    # _log_eval_diagnostics reads the same running_precision buffers
    # that train_step just updated.  Values must match exactly.
    diag = {}
    _log_eval_diagnostics(trainer, diag)

    assert "precision/level0_mean" in diag, (
        "_log_eval_diagnostics must emit precision/level0_mean"
    )
    assert abs(diag["precision/level0_mean"] - train_pi_mean) < 1e-6, (
        f"_log_eval_diagnostics precision/level0_mean={diag['precision/level0_mean']:.6f} "
        f"should match train_step precision/level0_mean={train_pi_mean:.6f} "
        f"(both read the same running_precision.precision buffer)"
    )


# ---------------------------------------------------------------------------
# Test 2 — Old key absent, new key present
# ---------------------------------------------------------------------------

def test_old_precision_raw_stat_key_absent_new_key_present():
    """precision_raw_stat/level{i}_mean must NOT appear; precision_ema_var must.

    Session N7 renames the EMA variance metric to reflect that after Option 2
    it is the normalized-error EMA variance, not a 'raw stat'.
    """
    torch.manual_seed(1)
    trainer = _make_pc_trainer()
    batch = _make_batch()

    train_metrics = trainer.train_step(batch)

    # Old key must be absent from training-step metrics
    for k in train_metrics:
        assert "precision_raw_stat" not in k, (
            f"Old key '{k}' found in train_step metrics — should have been "
            f"renamed to precision_ema_var/... in Session N7"
        )

    # New key must be present
    assert "precision_ema_var/level0_mean" in train_metrics, (
        "train_step must emit precision_ema_var/level0_mean (renamed from precision_raw_stat)"
    )
    assert train_metrics["precision_ema_var/level0_mean"] >= 0, (
        "EMA variance must be non-negative"
    )

    # Also verify _log_eval_diagnostics uses the new key
    diag = {}
    _log_eval_diagnostics(trainer, diag)

    for k in diag:
        assert "precision_raw_stat" not in k, (
            f"Old key '{k}' found in _log_eval_diagnostics output"
        )
    assert "precision_ema_var/level0_mean" in diag, (
        "_log_eval_diagnostics must emit precision_ema_var/level0_mean"
    )


# ---------------------------------------------------------------------------
# Test 3 — prediction_error semantics match ev.sqrt().mean()
# ---------------------------------------------------------------------------

def test_prediction_error_uses_ema_variance_sqrt():
    """prediction_error/level{i}_mean must equal ev.sqrt().mean() in train_step.

    Both train_step and _log_eval_diagnostics should use the EMA-variance
    RMS proxy (ev.sqrt().mean()), not point-in-time eps.abs().mean().
    This test verifies train_step's value by cross-checking against the buffer.
    """
    torch.manual_seed(2)
    trainer = _make_pc_trainer()
    batch = _make_batch()

    train_metrics = trainer.train_step(batch)

    # Read the buffer directly to compute expected value
    rp = trainer.core.pc_modules[0].running_precision
    expected = rp.ema_var.sqrt().mean().item()

    assert "prediction_error/level0_mean" in train_metrics, (
        "train_step must emit prediction_error/level0_mean"
    )
    assert abs(train_metrics["prediction_error/level0_mean"] - expected) < 1e-5, (
        f"train_step prediction_error/level0_mean={train_metrics['prediction_error/level0_mean']:.6f} "
        f"should equal ev.sqrt().mean()={expected:.6f} "
        f"(EMA-variance RMS proxy, not point-in-time eps.abs().mean())"
    )

    # _log_eval_diagnostics must use the same computation
    diag = {}
    _log_eval_diagnostics(trainer, diag)
    assert abs(diag["prediction_error/level0_mean"] - expected) < 1e-5, (
        f"_log_eval_diagnostics prediction_error/level0_mean={diag['prediction_error/level0_mean']:.6f} "
        f"should equal ev.sqrt().mean()={expected:.6f}"
    )
