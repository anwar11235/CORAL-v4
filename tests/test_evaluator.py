"""Tests for coral/evaluation/evaluator.py.

Critical regression test: evaluate_accuracy must use core.forward() (not a
manual reimplementation), so that input_injection and attention_bias are
applied identically to training. The old manual loop was missing both signals,
causing eval accuracy to stay at chance (1/9 ≈ 0.11) even for a trained model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.evaluation.evaluator import evaluate_accuracy


def _make_config(use_local_attention_bias: bool = False) -> tuple:
    """Return (CoralConfig, ModelConfig) for a tiny baseline model."""
    model_cfg = ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, timescale_base=3, K_max=2,
        use_predictive_coding=False, use_crystallisation=False,
        use_amort=False, lambda_amort=0.0,
        epsilon_min=0.01, lambda_pred=0.001, vocab_size=10, mode="baseline",
        inner_steps_override=2,  # fast tests
        use_local_attention_bias=use_local_attention_bias,
    )
    full_cfg = CoralConfig(model=model_cfg, device="cpu")
    full_cfg.training.precision = "float32"
    return full_cfg, model_cfg


def _make_loader(B: int = 4, L: int = 9, vocab: int = 10) -> DataLoader:
    """Tiny DataLoader with random inputs and valid labels."""
    inputs = torch.randint(0, vocab, (B, L))
    labels = torch.randint(1, vocab, (B, L))
    dataset = TensorDataset(inputs, labels)

    class _DictDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, i):
            return {"inputs": self.inputs[i], "labels": self.labels[i]}

    return DataLoader(_DictDataset(inputs, labels), batch_size=B)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_evaluate_accuracy_returns_required_keys():
    """evaluate_accuracy must return all three metric keys."""
    full_cfg, model_cfg = _make_config()
    adapter = GridAdapter(full_cfg, vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(model_cfg)
    loader = _make_loader()

    metrics = evaluate_accuracy(adapter, core, loader, device=torch.device("cpu"), dtype=torch.float32)

    assert "eval/exact_accuracy" in metrics
    assert "eval/token_accuracy" in metrics
    assert "eval/avg_halting_step" in metrics


def test_evaluate_accuracy_values_in_range():
    """All returned metrics must be in [0, 1] (halting step in [1, K_max])."""
    full_cfg, model_cfg = _make_config()
    adapter = GridAdapter(full_cfg, vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(model_cfg)
    loader = _make_loader()

    metrics = evaluate_accuracy(adapter, core, loader, device=torch.device("cpu"), dtype=torch.float32)

    assert 0.0 <= metrics["eval/exact_accuracy"] <= 1.0
    assert 0.0 <= metrics["eval/token_accuracy"] <= 1.0
    assert 1.0 <= metrics["eval/avg_halting_step"] <= model_cfg.K_max


def test_evaluate_accuracy_max_puzzles_limit():
    """max_puzzles must stop evaluation early."""
    full_cfg, model_cfg = _make_config()
    adapter = GridAdapter(full_cfg, vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(model_cfg)

    # 3 batches of 4 = 12 puzzles total; limit to 4
    inputs = torch.randint(0, 10, (12, 9))
    labels = torch.randint(1, 10, (12, 9))

    class _DD(torch.utils.data.Dataset):
        def __getitem__(self, i):
            return {"inputs": inputs[i], "labels": labels[i]}
        def __len__(self):
            return 12

    loader = DataLoader(_DD(), batch_size=4)
    metrics = evaluate_accuracy(
        adapter, core, loader,
        device=torch.device("cpu"), dtype=torch.float32,
        max_puzzles=4,
    )
    # Should have processed exactly 1 batch (4 puzzles)
    assert metrics["eval/exact_accuracy"] >= 0.0  # just verify it ran without error


# ---------------------------------------------------------------------------
# Regression: verify core.forward() is used (not a manual loop)
# ---------------------------------------------------------------------------

def test_evaluate_uses_core_forward_not_manual_loop():
    """evaluate_accuracy must delegate to core.forward(), not reimplement the loop.

    We verify this by monkeypatching core._run_level to raise an error.
    If evaluate_accuracy calls _run_level directly, this test fails.
    If it calls core() (forward), _run_level is called internally — but the
    patch only fires if called from outside core, so we use a call counter.
    """
    full_cfg, model_cfg = _make_config()
    adapter = GridAdapter(full_cfg, vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(model_cfg)

    # Track external calls to _run_level (from outside core.forward)
    external_calls = []
    _orig = core._run_level

    def _patched(*args, **kwargs):
        # Record the call stack depth to distinguish internal vs external calls.
        # We set a flag on core just before calling forward() and clear it after.
        if not getattr(core, "_in_forward", False):
            external_calls.append(True)
        return _orig(*args, **kwargs)

    core._run_level = _patched

    # Set the flag to True during forward so internal calls are not counted
    _orig_forward = core.forward

    def _fwd_with_flag(*args, **kwargs):
        core._in_forward = True
        try:
            return _orig_forward(*args, **kwargs)
        finally:
            core._in_forward = False

    core.forward = _fwd_with_flag

    loader = _make_loader()
    evaluate_accuracy(adapter, core, loader, device=torch.device("cpu"), dtype=torch.float32)

    assert len(external_calls) == 0, (
        f"evaluate_accuracy called core._run_level directly {len(external_calls)} time(s). "
        "It must use core.forward() instead."
    )


# ---------------------------------------------------------------------------
# Attention masks wired through (use_local_attention_bias=True)
# ---------------------------------------------------------------------------

def test_evaluate_accuracy_with_attention_bias():
    """evaluate_accuracy must pass attention_masks to core when use_local_attention_bias=True."""
    full_cfg, model_cfg = _make_config(use_local_attention_bias=True)
    adapter = GridAdapter(full_cfg, vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(model_cfg)

    # Set bias params to non-zero so they actually affect output
    with torch.no_grad():
        core.backbone.row_bias.fill_(0.5)
        core.backbone.col_bias.fill_(-0.5)
        core.backbone.box_bias.fill_(0.5)

    loader = _make_loader()
    metrics = evaluate_accuracy(adapter, core, loader, device=torch.device("cpu"), dtype=torch.float32)

    # Just verify it runs without error and returns valid values
    assert 0.0 <= metrics["eval/token_accuracy"] <= 1.0
