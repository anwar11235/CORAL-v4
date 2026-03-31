"""Tests for Session 6: representation diagnostics, collect_states, codebook_analysis.

Verification criteria:
  - compute_repr_diagnostics returns valid (non-NaN, non-zero) metrics
  - collect_states logic runs on tiny synthetic data without error
  - codebook_analysis functions run on synthetic .npz without error
"""

import importlib
import importlib.util
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.training.losses import CoralLoss
from coral.training.trainer import TrainerV4

# Path to scripts directory
_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sklearn = pytest.importorskip("sklearn", reason="sklearn not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config():
    return ModelConfig(
        n_levels=2,
        level_dims=[64, 32],
        backbone_dim=64,
        n_heads=4,
        d_k=16,
        ffn_expansion=2,
        timescale_base=2,
        K_max=4,
        use_predictive_coding=True,
        epsilon_min=0.01,
        lambda_pred=0.001,
        vocab_size=10,
    )


def _make_trainer(config=None, device="cpu"):
    if config is None:
        config = _small_config()
    full_config = CoralConfig(model=config, device=device)
    full_config.training.precision = "float32"

    adapter = GridAdapter(full_config, vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(config)
    loss_fn = CoralLoss(config)

    return TrainerV4(
        adapter=adapter,
        core=core,
        loss_fn=loss_fn,
        config=full_config,
        wandb_run=None,
    )


def _fake_eval_loader(n_batches=3, batch_size=4, seq_len=9, vocab=10):
    """List of fake batches with some empty cells (token=0)."""
    batches = []
    for _ in range(n_batches):
        inputs = torch.randint(0, vocab, (batch_size, seq_len))
        inputs[:, ::3] = 0   # some empty cells
        labels = torch.randint(1, vocab, (batch_size, seq_len))
        batches.append({"inputs": inputs, "labels": labels})
    return batches


def _load_ca():
    """Load codebook_analysis module from scripts/."""
    spec = importlib.util.spec_from_file_location(
        "codebook_analysis",
        os.path.join(_SCRIPTS_DIR, "codebook_analysis.py"),
    )
    ca = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ca)
    return ca


def _make_synthetic_npz(tmp_dir, N=30, L=9, n_segs=2, d=64):
    """Create a synthetic .npz file resembling collect_states output."""
    rng = np.random.default_rng(0)
    states = rng.standard_normal((N, L, n_segs, d)).astype(np.float32)
    labels = rng.integers(1, 10, (N, L)).astype(np.int16)
    given_mask = rng.integers(0, 2, (N, L)).astype(bool)
    segment_indices = np.array([2, 4], dtype=np.int32)

    path = os.path.join(tmp_dir, "synthetic_states.npz")
    np.savez_compressed(
        path,
        states=states,
        labels=labels,
        given_mask=given_mask,
        segment_indices=segment_indices,
    )
    return path


# ---------------------------------------------------------------------------
# Test 1: compute_repr_diagnostics returns valid metrics
# ---------------------------------------------------------------------------

def test_repr_diagnostics_valid_metrics():
    """compute_repr_diagnostics should return a dict with non-NaN, finite values."""
    trainer = _make_trainer()
    loader = _fake_eval_loader(n_batches=4, batch_size=8)
    metrics = trainer.compute_repr_diagnostics(loader, max_puzzles=32)

    assert isinstance(metrics, dict)
    assert len(metrics) > 0, "Should return at least one metric"
    for key, val in metrics.items():
        assert isinstance(val, float), f"{key} should be a float"
        assert not np.isnan(val), f"{key} is NaN"
        assert not np.isinf(val), f"{key} is Inf"


def test_repr_diagnostics_expected_keys():
    """compute_repr_diagnostics should return all expected metric keys."""
    trainer = _make_trainer()
    loader = _fake_eval_loader(n_batches=4, batch_size=8)
    metrics = trainer.compute_repr_diagnostics(loader, max_puzzles=32)

    for key in ("repr/inter_position_similarity", "repr/state_norm_mean", "repr/state_norm_std"):
        assert key in metrics, f"Missing expected key: {key}"


def test_repr_diagnostics_empty_loader():
    """compute_repr_diagnostics on empty loader should return empty dict."""
    trainer = _make_trainer()
    assert trainer.compute_repr_diagnostics([], max_puzzles=32) == {}


# ---------------------------------------------------------------------------
# Test 2: collect_states logic — unit test without checkpoint
# ---------------------------------------------------------------------------

def test_collect_states_save_load():
    """Verify the .npz output has the correct shapes."""
    config = _small_config()
    full_config = CoralConfig(model=config, device="cpu")
    full_config.training.precision = "float32"

    adapter = GridAdapter(full_config, vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(config)
    adapter.eval()
    core.eval()

    device = torch.device("cpu")
    segment_indices = [2, 4]
    n_segs = len(segment_indices)
    d = config.level_dims[0]
    batch_size = 4
    seq_len = 9

    fake_batches = [
        {"inputs": torch.randint(0, 10, (batch_size, seq_len)),
         "labels": torch.randint(1, 10, (batch_size, seq_len))}
        for _ in range(2)
    ]
    N_total = batch_size * len(fake_batches)

    states_arr = np.zeros((N_total, seq_len, n_segs, d), dtype=np.float32)
    labels_arr = np.zeros((N_total, seq_len), dtype=np.int16)
    given_arr = np.zeros((N_total, seq_len), dtype=bool)

    orig_threshold = core.config.halting_threshold
    core.config.halting_threshold = 2.0

    row = 0
    with torch.no_grad():
        for batch in fake_batches:
            inputs = batch["inputs"]
            labels_b = batch["labels"]
            B = inputs.shape[0]
            labels_arr[row:row + B] = labels_b.numpy().astype(np.int16)
            given_arr[row:row + B] = (inputs != 0).numpy()
            z1 = adapter.encode(inputs)
            for seg_col, target_seg in enumerate(segment_indices):
                out = core(z1, K_max=target_seg, training=False, decode_fn=None)
                states_arr[row:row + B, :, seg_col, :] = out.z_states[0].float().detach().numpy()
            row += B

    core.config.halting_threshold = orig_threshold

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name

    try:
        np.savez_compressed(
            tmp_path,
            states=states_arr,
            labels=labels_arr,
            given_mask=given_arr,
            segment_indices=np.array(segment_indices, dtype=np.int32),
        )

        # Load with mmap_mode=None to avoid Windows file lock
        data = np.load(tmp_path, mmap_mode=None)
        try:
            assert data["states"].shape == (N_total, seq_len, n_segs, d)
            assert data["labels"].shape == (N_total, seq_len)
            assert data["given_mask"].shape == (N_total, seq_len)
            assert list(data["segment_indices"]) == segment_indices
        finally:
            data.close()
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test 3: codebook_analysis functions
# ---------------------------------------------------------------------------

def test_codebook_per_head_analysis():
    """per_head_analysis runs without error on synthetic data."""
    ca = _load_ca()
    rng = np.random.default_rng(42)
    E, d = 200, 64
    empty_states = rng.standard_normal((E, d)).astype(np.float32)
    empty_labels = rng.integers(1, 10, size=E).astype(np.int32)

    results = ca.per_head_analysis(empty_states, empty_labels, n_heads=4, k_values=(8, 16))

    assert 8 in results and 16 in results
    for k in (8, 16):
        assert len(results[k]) == 4
        for hd in results[k]:
            assert 0.0 <= hd["purity"] <= 1.0
            assert hd["perplexity"] >= 1.0


def test_codebook_whole_vector_analysis():
    """whole_vector_analysis returns bypass_accuracy in [0,1]."""
    ca = _load_ca()
    rng = np.random.default_rng(0)
    E, d = 300, 64
    empty_states = rng.standard_normal((E, d)).astype(np.float32)
    empty_labels = rng.integers(1, 10, size=E).astype(np.int32)

    results = ca.whole_vector_analysis(empty_states, empty_labels, k_values=(16, 32))

    for k in (16, 32):
        r = results[k]
        assert "bypass_accuracy" in r
        assert 0.0 <= r["bypass_accuracy"] <= 1.0
        assert r["inertia"] >= 0.0
        assert r["perplexity"] >= 1.0


def test_codebook_analysis_end_to_end():
    """codebook_analysis functions work end-to-end with a synthetic .npz."""
    ca = _load_ca()

    with tempfile.TemporaryDirectory() as tmp_dir:
        npz_path = _make_synthetic_npz(tmp_dir, N=30, L=9, n_segs=2, d=64)

        # Load without mmap to avoid Windows file lock
        raw = np.load(npz_path, mmap_mode=None)
        states_all = raw["states"].copy()
        labels_all = raw["labels"].copy()
        given_mask_arr = raw["given_mask"].copy()
        raw.close()

        N, L, n_segs, d = states_all.shape
        states = states_all[:, :, -1, :]

        empty_mask = ~given_mask_arr.astype(bool)
        states_flat = states.reshape(N * L, d)
        labels_flat = labels_all.reshape(N * L)
        empty_flat = empty_mask.reshape(N * L)

        empty_states = states_flat[empty_flat].astype(np.float32)
        empty_labels = labels_flat[empty_flat]
        valid = (empty_labels >= 1) & (empty_labels <= 9)
        empty_states = empty_states[valid]
        empty_labels = empty_labels[valid].astype(np.int32)

        if empty_states.shape[0] < 10:
            pytest.skip("Not enough empty cells in synthetic data")

        per_head = ca.per_head_analysis(empty_states, empty_labels, n_heads=4, k_values=(8,))
        whole = ca.whole_vector_analysis(empty_states, empty_labels, k_values=(8,))

        assert 8 in per_head
        assert 8 in whole
        assert 0.0 <= whole[8]["bypass_accuracy"] <= 1.0
