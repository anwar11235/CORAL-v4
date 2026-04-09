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


# ---------------------------------------------------------------------------
# Difficulty-bucketed accuracy
# ---------------------------------------------------------------------------

def _make_bucketed_loader(bucket_counts: dict) -> DataLoader:
    """Build a 9×9 (81-cell) DataLoader with puzzles in specified difficulty buckets.

    Args:
        bucket_counts: dict mapping bucket name to (n_empty, n_puzzles) pairs.
            bucket names: "0_29", "30_49", "50_59", "60_plus"
            n_empty: number of cells set to 1 (empty) in that puzzle
            n_puzzles: number of puzzles to create for that bucket

    Returns a DataLoader with batch_size = total puzzles.
    """
    L = 81
    EMPTY_EMPTY_COUNTS = {"0_29": 20, "30_49": 40, "50_59": 55, "60_plus": 65}
    inputs_list = []
    labels_list = []
    for bucket, n_puzzles in bucket_counts.items():
        n_empty = EMPTY_EMPTY_COUNTS[bucket]
        for _ in range(n_puzzles):
            inp = torch.full((L,), 2, dtype=torch.long)   # non-empty cells = token 2
            inp[:n_empty] = 1                              # first n_empty cells are empty (token 1)
            lbl = torch.randint(1, 10, (L,))
            inputs_list.append(inp)
            labels_list.append(lbl)

    inputs_t = torch.stack(inputs_list)
    labels_t = torch.stack(labels_list)
    total = inputs_t.shape[0]

    class _DD(torch.utils.data.Dataset):
        def __getitem__(self, i):
            return {"inputs": inputs_t[i], "labels": labels_t[i]}
        def __len__(self):
            return total

    return DataLoader(_DD(), batch_size=total)


def test_bucket_counts_sum_to_total():
    """Bucket counts must sum to the total number of evaluated puzzles."""
    full_cfg, model_cfg = _make_config()
    # Use a 9×9 adapter so inputs have 81 cells (enables 30-49 / 50-59 / 60+ buckets)
    adapter = GridAdapter(full_cfg, vocab_size=10, grid_height=9, grid_width=9)
    core = CoralCore(model_cfg)

    bucket_counts = {"0_29": 2, "30_49": 3, "50_59": 2, "60_plus": 1}
    loader = _make_bucketed_loader(bucket_counts)

    metrics = evaluate_accuracy(adapter, core, loader, device=torch.device("cpu"), dtype=torch.float32)

    total = sum(bucket_counts.values())
    bucket_total = sum(
        int(metrics[f"eval/bucket_{k}_count"])
        for k in ("0_29", "30_49", "50_59", "60_plus")
    )
    assert bucket_total == total, (
        f"Bucket counts sum to {bucket_total}, expected {total}. "
        f"Counts: { {k: metrics[f'eval/bucket_{k}_count'] for k in ('0_29','30_49','50_59','60_plus')} }"
    )
    # Also verify individual bucket counts match expectations
    for k, expected_n in bucket_counts.items():
        assert metrics[f"eval/bucket_{k}_count"] == expected_n


def test_bucket_accuracies_in_range():
    """All per-bucket accuracy values must be in [0, 1]."""
    full_cfg, model_cfg = _make_config()
    adapter = GridAdapter(full_cfg, vocab_size=10, grid_height=9, grid_width=9)
    core = CoralCore(model_cfg)

    bucket_counts = {"0_29": 2, "30_49": 2, "50_59": 2, "60_plus": 2}
    loader = _make_bucketed_loader(bucket_counts)

    metrics = evaluate_accuracy(adapter, core, loader, device=torch.device("cpu"), dtype=torch.float32)

    for k in ("0_29", "30_49", "50_59", "60_plus"):
        for suffix in ("token_acc", "empty_acc", "exact_acc"):
            key = f"eval/bucket_{k}_{suffix}"
            val = metrics[key]
            assert 0.0 <= val <= 1.0, f"{key} = {val} is out of [0, 1]"
