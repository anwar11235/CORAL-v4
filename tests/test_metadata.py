"""Tests for SudokuDatasetMetadata extra-field tolerance."""

import pytest

from coral.data.sudoku_dataset import SudokuDatasetMetadata


def test_metadata_ignores_extra_fields():
    """Full TRM maze dataset.json schema — extra fields must not raise."""
    raw = {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 6,
        "seq_len": 900,
        "num_puzzle_identifiers": 1,
        "total_groups": 1000,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": 1000,   # extra field not declared on the class
        "sets": ["train"],
    }
    meta = SudokuDatasetMetadata(**raw)
    assert meta.vocab_size == 6
    assert meta.seq_len == 900
    assert meta.total_groups == 1000
    assert meta.sets == ["train"]
    # total_puzzles is silently dropped — must not be stored as an attribute
    assert not hasattr(meta, "total_puzzles")


def test_metadata_sudoku_defaults_unchanged():
    """Instantiating with no args preserves Sudoku defaults."""
    meta = SudokuDatasetMetadata()
    assert meta.vocab_size == 10
    assert meta.seq_len == 81
    assert meta.sets == ["train"]
