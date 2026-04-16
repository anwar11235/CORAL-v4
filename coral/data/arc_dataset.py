"""CORAL — ARC-AGI-1 data loader.

Loads the public ARC-AGI-1 benchmark (Francois Chollet, 2019) into
PyTorch-compatible tensors for use with CORAL's GridAdapter.

Source: https://github.com/fchollet/ARC-AGI
  data/training/  — 400 tasks (official training set)
  data/evaluation/ — 400 tasks (official evaluation set, held out)

Each task is a JSON file with 2-10 demonstration (input, output) pairs
plus 1 test pair whose output is the prediction target.

==========================================================================
Design decisions (D1–D4 per prompt spec)
==========================================================================

D1. PADDING STRATEGY — pad to 30×30 with token 10.

  ARC grids range from 1×1 to 30×30. Three options were considered:
    (a) Pad all samples to 30×30 with a dedicated pad token.     ← CHOSEN
    (b) Pad per-batch to the max size in that batch.
    (c) No padding; return variable-size tensors.

  Option (a) is chosen for compatibility with CORAL's GridAdapter, which
  uses a fixed-size 2D positional embedding (nn.Embedding(H*W, d_model)).
  With option (a), the adapter can be instantiated as GridAdapter(…,
  grid_height=30, grid_width=30), producing seq_len=900 — identical
  to the Maze-Hard adapter already in the codebase. Padded positions
  carry token 10 and are excluded from the loss via output_mask.

  Option (b) would require dynamic adapter re-instantiation per batch,
  which is incompatible with the current architecture.
  Option (c) is incompatible with batching in PyTorch without custom
  collate functions.

  Pad token: 10. This keeps the vocab range [0, 10] and coincidentally
  matches CORAL's existing vocab_size=11 (but see D4 for semantics).

D2. FEW-SHOT REPRESENTATION — one sample = test pair + demos as context.

  ARC tasks have 2–10 demonstration pairs ("train" in the JSON). Options:
    (a) One sample = all demos concatenated into one long sequence.
    (b) One sample = one (task, single demo pair).
    (c) One sample = test pair, with demos as a separate context tensor.
                                                                  ← CHOSEN

  Option (c) is chosen because it preserves the few-shot structure:
  the model sees "given these examples, predict the test output." The
  demonstration pairs are stored in demo_pairs as a list of (input,
  output) padded 30×30 tensors. The test input/output are the primary
  "inputs"/"labels" tensors.

  This matches ARC's intended evaluation protocol (Chollet 2019):
  generalise from 2–10 demonstrations to a held-out test pair.

  Option (a) would lose the test/demo distinction.
  Option (b) would discard inter-demo relationships within a task.

  Note: CORAL does not yet have a mechanism to consume demo_pairs as
  additional context. The field is present in the batch dict for future
  use. A baseline training run would simply ignore demo_pairs.

D3. TRAIN/EVAL SPLIT — use the official ARC-AGI-1 split.

  ARC-AGI-1 provides two separate directories:
    data/training/   — 400 tasks → split="train"
    data/evaluation/ — 400 tasks → split="eval"

  The official split is used as-is. No custom train/val split from the
  training set. The evaluation set is held out for final evaluation only.
  This is consistent with ARC's intended benchmarking protocol.

D4. INPUT/OUTPUT REPRESENTATION — integers 0–9 plus pad token 10.

  ARC grid cells are integers 0–9 (10 colors). Token 10 is the pad token
  introduced by this loader (not part of the original ARC format).
  Final vocab size = 11.

  IMPORTANT: This coincidentally matches Sudoku's vocab_size=11, but
  the semantics are entirely different:
    Sudoku:  0 = empty cell, 1–9 = digit, 10 = padding (never in data)
    ARC:     0–9 = colors (all valid grid values), 10 = padding only
  Do not share embedding weights between Sudoku and ARC adapters.

==========================================================================
Download behavior
==========================================================================

If data_dir does not exist or is empty, the loader attempts to download
the dataset from GitHub to ~/.cache/coral/arc-agi-1/. The download
uses only the Python standard library (urllib) to avoid adding
dependencies.

If the download fails (no network, rate limit), a FileNotFoundError is
raised with a message pointing to the manual download instructions.
"""

import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARC_MAX_GRID_SIZE: int = 30
ARC_PAD_TOKEN: int = 10      # Values 0–9 are colors; 10 is pad
ARC_VOCAB_SIZE: int = 11     # 0–9 colors + 1 pad token

# Official ARC-AGI-1 GitHub release (zip archive of the full repository)
_ARC_GITHUB_ZIP = (
    "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
)
_ARC_CACHE_DIR = Path.home() / ".cache" / "coral" / "arc-agi-1"

# Subdirectory names within the ARC repo for each split
_SPLIT_SUBDIR = {
    "train": "training",
    "eval": "evaluation",
}


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _ensure_arc_data(data_dir: str) -> str:
    """Ensure ARC data exists at data_dir, downloading if necessary.

    Args:
        data_dir: Path to directory expected to contain training/ and
                  evaluation/ subdirectories with JSON task files.

    Returns:
        Validated data_dir path (the root containing training/ and evaluation/).

    Raises:
        FileNotFoundError: If data_dir doesn't exist and download fails.
    """
    data_path = Path(data_dir)

    # Check if at least one split directory has JSON files
    for subdir in _SPLIT_SUBDIR.values():
        candidate = data_path / subdir
        if candidate.is_dir() and any(candidate.glob("*.json")):
            return str(data_path)

    # Try the cache directory
    cache_root = _ARC_CACHE_DIR
    for subdir in _SPLIT_SUBDIR.values():
        candidate = cache_root / subdir
        if candidate.is_dir() and any(candidate.glob("*.json")):
            return str(cache_root)

    # Attempt download
    print(f"ARC-AGI-1 data not found at {data_dir}. Attempting download from GitHub...")
    try:
        _download_arc(cache_root)
        return str(cache_root)
    except Exception as e:
        raise FileNotFoundError(
            f"ARC-AGI-1 data not found at '{data_dir}' and automatic download failed: {e}\n"
            f"Manual download: git clone https://github.com/fchollet/ARC-AGI "
            f"then set data_dir to the cloned repo root (containing data/training/ and "
            f"data/evaluation/)."
        ) from e


def _download_arc(dest: Path) -> None:
    """Download ARC-AGI-1 from GitHub and extract to dest."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "arc-agi.zip"

    print(f"Downloading {_ARC_GITHUB_ZIP} ...")
    urllib.request.urlretrieve(_ARC_GITHUB_ZIP, zip_path)

    print(f"Extracting to {dest} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()

    # The zip extracts to ARC-AGI-master/data/training and .../evaluation
    # Move the data subdirectories up to dest/training and dest/evaluation
    extracted_root = dest / "ARC-AGI-master" / "data"
    for subdir in _SPLIT_SUBDIR.values():
        src = extracted_root / subdir
        dst = dest / subdir
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))

    # Clean up extracted root
    arc_master = dest / "ARC-AGI-master"
    if arc_master.exists():
        shutil.rmtree(arc_master)

    print(f"ARC-AGI-1 data ready at {dest}")


# ---------------------------------------------------------------------------
# Grid padding helpers
# ---------------------------------------------------------------------------

def _pad_grid(
    grid: List[List[int]],
    max_size: int = ARC_MAX_GRID_SIZE,
    pad_token: int = ARC_PAD_TOKEN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a 2D grid to [max_size, max_size] with pad_token.

    Args:
        grid:      2D list of ints, shape [H, W].
        max_size:  Target size for both dimensions.
        pad_token: Value to fill padded cells.

    Returns:
        padded: [max_size, max_size] int64 tensor.
        mask:   [max_size, max_size] bool tensor — True at non-padded cells.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    padded = torch.full((max_size, max_size), pad_token, dtype=torch.long)
    mask = torch.zeros(max_size, max_size, dtype=torch.bool)

    if h > 0 and w > 0:
        data = torch.tensor(grid, dtype=torch.long)  # [h, w]
        padded[:h, :w] = data
        mask[:h, :w] = True

    return padded, mask


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class ARCTaskDataset(Dataset):
    """Loads ARC-AGI-1 tasks into CORAL-compatible tensors.

    Each sample corresponds to one ARC task. The test pair (input, output)
    is the primary prediction target. Demonstration pairs are available as
    context via the demo_pairs field (D2: one sample = test pair + demos
    as context).

    Args:
        data_dir:      Root directory containing training/ and evaluation/
                       subdirectories. If absent, download is attempted to
                       ~/.cache/coral/arc-agi-1/.
        split:         "train" (400 tasks) or "eval" (400 tasks).
        max_grid_size: Pad all grids to this size × this size. Default 30.
        pad_token:     Token used for padding. Default 10.

    Returns per __getitem__:
        {
            "inputs":      [max_grid_size, max_grid_size] int64 — test input
                           padded to max_grid_size×max_grid_size.
            "labels":      [max_grid_size, max_grid_size] int64 — test output
                           padded. Padded positions have value pad_token.
            "input_mask":  [max_grid_size, max_grid_size] bool — True at
                           non-padded input cells.
            "output_mask": [max_grid_size, max_grid_size] bool — True at
                           non-padded output cells.
            "task_id":     str — filename stem (e.g. "007bbfb7").
            "demo_pairs":  List of {"input": tensor, "output": tensor,
                           "input_mask": tensor, "output_mask": tensor}
                           dicts, one per demonstration pair.
        }
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_grid_size: int = ARC_MAX_GRID_SIZE,
        pad_token: int = ARC_PAD_TOKEN,
    ) -> None:
        if split not in _SPLIT_SUBDIR:
            raise ValueError(f"split must be 'train' or 'eval', got '{split}'")

        self.split = split
        self.max_grid_size = max_grid_size
        self.pad_token = pad_token

        # Resolve data directory (download if needed)
        resolved_root = _ensure_arc_data(data_dir)
        self._split_dir = Path(resolved_root) / _SPLIT_SUBDIR[split]

        if not self._split_dir.is_dir():
            raise FileNotFoundError(
                f"Split directory not found: {self._split_dir}"
            )

        # Enumerate task files, sorted for reproducibility
        self._task_files: List[Path] = sorted(self._split_dir.glob("*.json"))
        if len(self._task_files) == 0:
            raise FileNotFoundError(
                f"No JSON task files found in {self._split_dir}"
            )

    def __len__(self) -> int:
        return len(self._task_files)

    def __getitem__(self, idx: int) -> Dict:
        task_path = self._task_files[idx]
        task_id = task_path.stem

        with open(task_path, "r", encoding="utf-8") as f:
            task = json.load(f)

        # Test pair — the primary prediction target (D2).
        # Use the first test pair (ARC eval has exactly 1 test pair in the
        # public training set; the hidden evaluation set has 1 per task too).
        test_pair = task["test"][0]
        inputs, input_mask = _pad_grid(
            test_pair["input"], self.max_grid_size, self.pad_token
        )
        labels, output_mask = _pad_grid(
            test_pair["output"], self.max_grid_size, self.pad_token
        )

        # Demonstration pairs — context for few-shot conditioning (D2).
        demo_pairs = []
        for demo in task["train"]:
            d_in, d_in_mask = _pad_grid(
                demo["input"], self.max_grid_size, self.pad_token
            )
            d_out, d_out_mask = _pad_grid(
                demo["output"], self.max_grid_size, self.pad_token
            )
            demo_pairs.append({
                "input": d_in,
                "output": d_out,
                "input_mask": d_in_mask,
                "output_mask": d_out_mask,
            })

        return {
            "inputs": inputs,
            "labels": labels,
            "input_mask": input_mask,
            "output_mask": output_mask,
            "task_id": task_id,
            "demo_pairs": demo_pairs,
        }

    def get_task_ids(self) -> List[str]:
        """Return all task ID strings in dataset order."""
        return [p.stem for p in self._task_files]
