"""Sudoku dataset for CORAL v4.

Loads HRM-format puzzle datasets (produced by build_sudoku_dataset.py).
Each dataset split contains .npy memory-mapped arrays:
    {set_name}__inputs.npy               — token ids [N, 81]
    {set_name}__labels.npy               — target ids [N, 81]
    {set_name}__puzzle_identifiers.npy   — per-puzzle IDs [num_puzzles]
    {set_name}__puzzle_indices.npy       — CSR-style start indices into examples
    {set_name}__group_indices.npy        — CSR-style start indices into puzzle groups

A dataset.json file in the split directory contains SudokuDatasetMetadata.

Training iteration: randomly shuffle puzzle groups, sample one example per group
per step, pack into global_batch_size batches.
Eval iteration: iterate all examples sequentially without shuffling.
"""

import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

IGNORE_LABEL_ID: int = -100
SUDOKU_SEQ_LEN: int = 81
SUDOKU_VOCAB_SIZE: int = 10  # digits 0–9, 0 = empty


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------


class SudokuDatasetMetadata:
    """Metadata stored alongside each dataset split."""

    def __init__(
        self,
        pad_id: int = 0,
        ignore_label_id: Optional[int] = None,
        blank_identifier_id: int = 0,
        vocab_size: int = SUDOKU_VOCAB_SIZE,
        seq_len: int = SUDOKU_SEQ_LEN,
        num_puzzle_identifiers: int = 1000,
        total_groups: int = 1000,
        mean_puzzle_examples: float = 1000.0,
        sets: Optional[List[str]] = None,
    ):
        self.pad_id = pad_id
        self.ignore_label_id = ignore_label_id
        self.blank_identifier_id = blank_identifier_id
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_puzzle_identifiers = num_puzzle_identifiers
        self.total_groups = total_groups
        self.mean_puzzle_examples = mean_puzzle_examples
        self.sets = sets or ["train"]

    @classmethod
    def from_json(cls, path: str) -> "SudokuDatasetMetadata":
        with open(path, "r") as f:
            d = json.load(f)
        return cls(**d)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _sample_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    global_batch_size: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Pack examples from groups into a batch of size <= global_batch_size."""
    batch: List[np.ndarray] = []
    batch_puzzle_indices: List[np.ndarray] = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)
        append_size = min(puzzle_size, global_batch_size - current_size)

        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))
        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SudokuDataset(IterableDataset):
    """Iterable dataset for HRM-format Sudoku puzzle datasets.

    Yields dicts with keys: "inputs" [B, 81], "labels" [B, 81].
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        global_batch_size: int = 64,
        rank: int = 0,
        num_replicas: int = 1,
        seed: int = 0,
        test_set_mode: bool = False,
        epochs_per_iter: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.global_batch_size = global_batch_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.seed = seed
        self.test_set_mode = test_set_mode
        self.epochs_per_iter = epochs_per_iter

        assert global_batch_size % num_replicas == 0, (
            f"global_batch_size {global_batch_size} must be divisible by "
            f"num_replicas {num_replicas}."
        )
        self.local_batch_size = global_batch_size // num_replicas

        self._data: Optional[Dict] = None
        self._iters: int = 0
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> SudokuDatasetMetadata:
        metadata_path = os.path.join(self.dataset_path, self.split, "dataset.json")
        if os.path.exists(metadata_path):
            return SudokuDatasetMetadata.from_json(metadata_path)
        return SudokuDatasetMetadata()

    def _lazy_load(self) -> None:
        if self._data is not None:
            return

        mmap_modes = {
            "inputs": "r",
            "labels": "r",
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None,
        }

        self._data = {}
        for set_name in self.metadata.sets:
            self._data[set_name] = {
                field: np.load(
                    os.path.join(
                        self.dataset_path, self.split, f"{set_name}__{field}.npy"
                    ),
                    mmap_mode=mmap_mode,
                )
                for field, mmap_mode in mmap_modes.items()
            }

    def _collate(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Remap ignore labels, pad to local_batch_size, tensorize."""
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_vals = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
            }
            batch = {
                k: np.pad(
                    v,
                    ((0, pad),) + ((0, 0),) * (v.ndim - 1),
                    constant_values=pad_vals[k],
                )
                for k, v in batch.items()
            }

        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_test(self) -> Iterator:
        for set_name, dataset in self._data.items():  # type: ignore[union-attr]
            total = len(dataset["inputs"])
            start = 0
            while start < total:
                end = min(total, start + self.global_batch_size)
                local_start = start + self.rank * self.local_batch_size
                local_end = min(
                    start + (self.rank + 1) * self.local_batch_size, end
                )

                puzzle_ids = []
                puzzle_idx = (
                    np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                )
                for i in range(local_start, local_end):
                    while (
                        puzzle_idx + 1 < len(dataset["puzzle_indices"])
                        and i >= dataset["puzzle_indices"][puzzle_idx + 1]
                    ):
                        puzzle_idx += 1
                    puzzle_ids.append(puzzle_idx)

                yield self._collate({
                    "inputs": dataset["inputs"][local_start:local_end],
                    "labels": dataset["labels"][local_start:local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_ids],
                })
                start += self.global_batch_size

    def _iter_train(self) -> Iterator:
        for set_name, dataset in self._data.items():  # type: ignore[union-attr]
            self._iters += 1
            rng = np.random.Generator(
                np.random.Philox(seed=self.seed + self._iters)
            )

            group_order = np.concatenate([
                rng.permutation(dataset["group_indices"].size - 1)
                for _ in range(self.epochs_per_iter)
            ])
            start = 0

            while start < group_order.size:
                start, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start,
                    global_batch_size=self.global_batch_size,
                )

                effective_bs = batch_puzzle_indices.size
                if effective_bs < self.global_batch_size:
                    break

                r = self.rank
                bs = self.local_batch_size
                yield self._collate({
                    "inputs": dataset["inputs"][batch_indices[r * bs:(r + 1) * bs]],
                    "labels": dataset["labels"][batch_indices[r * bs:(r + 1) * bs]],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][
                        batch_puzzle_indices[r * bs:(r + 1) * bs]
                    ],
                })

    def __iter__(self) -> Iterator:
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, (
            "SudokuDataset does not support num_workers > 1."
        )
        self._lazy_load()
        if self.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_sudoku_dataloader(
    dataset_path: str,
    split: str,
    global_batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    test_set_mode: bool = False,
    epochs_per_iter: int = 1,
) -> Tuple[DataLoader, SudokuDatasetMetadata]:
    """Build a DataLoader for a single Sudoku split."""
    dataset = SudokuDataset(
        dataset_path=dataset_path,
        split=split,
        global_batch_size=global_batch_size,
        rank=rank,
        num_replicas=world_size,
        seed=seed,
        test_set_mode=test_set_mode,
        epochs_per_iter=epochs_per_iter,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader, dataset.metadata
