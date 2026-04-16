"""Tests for coral/data/arc_dataset.py — ARC-AGI-1 data loader.

Uses small fixture tasks committed under tests/fixtures/arc/ so no
network access is required.

Fixture layout:
  tests/fixtures/arc/training/   — 3 task JSON files (task_001 to task_003)
  tests/fixtures/arc/evaluation/ — 1 task JSON file (eval_task_001)
"""

import os
from pathlib import Path

import pytest
import torch

from coral.data.arc_dataset import ARCTaskDataset, ARC_PAD_TOKEN, ARC_MAX_GRID_SIZE

# Absolute path to the fixture directory
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "arc"


# ---------------------------------------------------------------------------
# Test 1 — Dataset length matches number of JSON files in each split
# ---------------------------------------------------------------------------

def test_dataset_len_train():
    """Train split must return as many samples as there are JSON files."""
    ds = ARCTaskDataset(data_dir=str(FIXTURE_DIR), split="train")
    n_json = len(list((FIXTURE_DIR / "training").glob("*.json")))
    assert len(ds) == n_json, (
        f"Expected len(ds) == {n_json} (number of JSON files), got {len(ds)}"
    )
    assert len(ds) == 3, f"Expected 3 fixture training tasks, got {len(ds)}"


def test_dataset_len_eval():
    """Eval split must return as many samples as there are JSON files."""
    ds = ARCTaskDataset(data_dir=str(FIXTURE_DIR), split="eval")
    n_json = len(list((FIXTURE_DIR / "evaluation").glob("*.json")))
    assert len(ds) == n_json, (
        f"Expected len(ds) == {n_json} (number of JSON files), got {len(ds)}"
    )
    assert len(ds) == 1, f"Expected 1 fixture eval task, got {len(ds)}"


# ---------------------------------------------------------------------------
# Test 2 — Sample shapes, dtype, and value range
# ---------------------------------------------------------------------------

def test_sample_shapes_and_dtype():
    """Loaded sample must have inputs/labels of shape [30, 30], dtype torch.long."""
    ds = ARCTaskDataset(data_dir=str(FIXTURE_DIR), split="train")
    sample = ds[0]

    assert "inputs" in sample, "Sample must contain 'inputs'"
    assert "labels" in sample, "Sample must contain 'labels'"
    assert "input_mask" in sample, "Sample must contain 'input_mask'"
    assert "output_mask" in sample, "Sample must contain 'output_mask'"
    assert "task_id" in sample, "Sample must contain 'task_id'"
    assert "demo_pairs" in sample, "Sample must contain 'demo_pairs'"

    assert sample["inputs"].shape == (ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE), (
        f"inputs shape should be ({ARC_MAX_GRID_SIZE}, {ARC_MAX_GRID_SIZE}), "
        f"got {sample['inputs'].shape}"
    )
    assert sample["labels"].shape == (ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE), (
        f"labels shape should be ({ARC_MAX_GRID_SIZE}, {ARC_MAX_GRID_SIZE}), "
        f"got {sample['labels'].shape}"
    )
    assert sample["inputs"].dtype == torch.long, (
        f"inputs dtype should be torch.long, got {sample['inputs'].dtype}"
    )
    assert sample["labels"].dtype == torch.long, (
        f"labels dtype should be torch.long, got {sample['labels'].dtype}"
    )

    # Values in [0, 10] — colors 0-9 plus pad token 10
    assert sample["inputs"].min().item() >= 0, "inputs values must be >= 0"
    assert sample["inputs"].max().item() <= 10, "inputs values must be <= 10"
    assert sample["labels"].min().item() >= 0, "labels values must be >= 0"
    assert sample["labels"].max().item() <= 10, "labels values must be <= 10"


def test_all_samples_shapes_valid():
    """Every sample in every split must have [30, 30] inputs/labels tensors."""
    for split in ("train", "eval"):
        ds = ARCTaskDataset(data_dir=str(FIXTURE_DIR), split=split)
        for i in range(len(ds)):
            s = ds[i]
            assert s["inputs"].shape == (ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE), (
                f"split={split}, idx={i}: inputs shape {s['inputs'].shape}"
            )
            assert s["labels"].shape == (ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE)


# ---------------------------------------------------------------------------
# Test 3 — Pad positions have value 10 and are consistent with mask
# ---------------------------------------------------------------------------

def test_pad_positions_consistent_with_mask():
    """Padded positions must have value ARC_PAD_TOKEN (10) and mask=False.
    Non-padded positions must have mask=True and value in [0, 9].
    """
    ds = ARCTaskDataset(data_dir=str(FIXTURE_DIR), split="train")

    for i in range(len(ds)):
        sample = ds[i]
        inputs = sample["inputs"]           # [30, 30]
        input_mask = sample["input_mask"]   # [30, 30] bool

        # Where mask is False → padded → value must be ARC_PAD_TOKEN
        pad_positions = ~input_mask
        if pad_positions.any():
            pad_values = inputs[pad_positions]
            assert (pad_values == ARC_PAD_TOKEN).all(), (
                f"Task {sample['task_id']}: padded positions must have value "
                f"{ARC_PAD_TOKEN}, got {pad_values.unique().tolist()}"
            )

        # Where mask is True → real cell → value must be in [0, 9]
        if input_mask.any():
            real_values = inputs[input_mask]
            assert real_values.min().item() >= 0, (
                f"Task {sample['task_id']}: real cells must be >= 0"
            )
            assert real_values.max().item() <= 9, (
                f"Task {sample['task_id']}: real cells must be <= 9 (not pad)"
            )

        # Same checks for output
        labels = sample["labels"]
        output_mask = sample["output_mask"]
        if (~output_mask).any():
            assert (labels[~output_mask] == ARC_PAD_TOKEN).all()
        if output_mask.any():
            assert labels[output_mask].max().item() <= 9


# ---------------------------------------------------------------------------
# Test 4 — Same task ID produces reproducible tensors across __getitem__ calls
# ---------------------------------------------------------------------------

def test_reproducible_across_getitem_calls():
    """Calling __getitem__ twice with the same index must return identical tensors."""
    ds = ARCTaskDataset(data_dir=str(FIXTURE_DIR), split="train")

    for i in range(len(ds)):
        s1 = ds[i]
        s2 = ds[i]

        assert s1["task_id"] == s2["task_id"], (
            f"task_id changed across calls for idx={i}"
        )
        assert torch.equal(s1["inputs"], s2["inputs"]), (
            f"inputs changed across calls for task {s1['task_id']}"
        )
        assert torch.equal(s1["labels"], s2["labels"]), (
            f"labels changed across calls for task {s1['task_id']}"
        )
        assert torch.equal(s1["input_mask"], s2["input_mask"]), (
            f"input_mask changed across calls for task {s1['task_id']}"
        )


# ---------------------------------------------------------------------------
# Additional: demo_pairs structure
# ---------------------------------------------------------------------------

def test_demo_pairs_structure():
    """demo_pairs must be a list of dicts with input/output/mask tensors."""
    ds = ARCTaskDataset(data_dir=str(FIXTURE_DIR), split="train")

    for i in range(len(ds)):
        sample = ds[i]
        demo_pairs = sample["demo_pairs"]
        assert isinstance(demo_pairs, list), "demo_pairs must be a list"
        assert len(demo_pairs) >= 1, "Must have at least 1 demo pair"
        for dp in demo_pairs:
            assert "input" in dp and "output" in dp
            assert dp["input"].shape == (ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE)
            assert dp["output"].shape == (ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE)
            assert dp["input"].dtype == torch.long
            assert "input_mask" in dp and "output_mask" in dp
            assert dp["input_mask"].dtype == torch.bool


def test_task_ids_unique():
    """All task IDs within a split must be unique."""
    ds = ARCTaskDataset(data_dir=str(FIXTURE_DIR), split="train")
    ids = ds.get_task_ids()
    assert len(ids) == len(set(ids)), f"Duplicate task IDs found: {ids}"


def test_invalid_split_raises():
    """Passing an unknown split name must raise ValueError."""
    with pytest.raises(ValueError, match="split must be"):
        ARCTaskDataset(data_dir=str(FIXTURE_DIR), split="test")
