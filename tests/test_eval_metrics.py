"""Tests for precision/recall@5 metric computation (Session E).

Verifies compute_precision_recall_at_5 is:
  - correct on a hand-constructed batch
  - non-NaN when the model never predicts the token
"""

import torch
import pytest

from coral.evaluation.evaluator import compute_precision_recall_at_5


def test_precision_recall_at_5():
    """Aggregate TP/FP/FN over two sequences and check precision == recall == 0.6.

    labels row 0: path cells at positions 2, 3   (2 ground-truth positives)
    labels row 1: path cells at positions 0, 3, 4 (3 ground-truth positives)

    preds row 0: [1,5,5,2,2,1]  → TP at pos 2, FN at pos 3, FP at pos 1
    preds row 1: [5,2,5,5,1,2]  → TP at pos 0,3, FN at pos 4, FP at pos 2

    Aggregate: TP=3, FP=2, FN=2
    precision = 3 / (3+2) = 0.6
    recall    = 3 / (3+2) = 0.6
    """
    labels = torch.tensor([
        [1, 2, 5, 5, 2, 1],   # 2 path cells
        [5, 2, 1, 5, 5, 2],   # 3 path cells
    ])
    preds = torch.tensor([
        [1, 5, 5, 2, 2, 1],   # TP=1 (pos 2), FN=1 (pos 3), FP=1 (pos 1)
        [5, 2, 5, 5, 1, 2],   # TP=2 (pos 0, 3), FN=1 (pos 4), FP=1 (pos 2)
    ])
    # Aggregate: TP=3, FP=2, FN=2
    # precision = 3/5 = 0.6
    # recall    = 3/5 = 0.6
    prec, rec = compute_precision_recall_at_5(preds, labels, token=5)
    assert abs(prec - 0.6) < 1e-6, f"Expected precision=0.6, got {prec}"
    assert abs(rec - 0.6) < 1e-6, f"Expected recall=0.6, got {rec}"


def test_precision_at_5_never_predicts_token():
    """When the model never predicts token 5, precision must be 0.0, not NaN."""
    labels = torch.tensor([[1, 2, 5, 5, 2, 1]])
    preds = torch.zeros_like(labels)  # model always predicts 0, never 5

    prec, rec = compute_precision_recall_at_5(preds, labels, token=5)

    assert not torch.isnan(torch.tensor(prec)), "precision must not be NaN"
    assert prec == pytest.approx(0.0), f"Expected 0.0, got {prec}"
    # recall: 0 TP, 2 FN → 0/2 = 0.0
    assert rec == pytest.approx(0.0), f"Expected recall=0.0, got {rec}"


def test_precision_recall_at_5_perfect():
    """Perfect predictions: precision == recall == 1.0."""
    labels = torch.tensor([[1, 5, 2, 5, 1, 5]])
    preds = labels.clone()

    prec, rec = compute_precision_recall_at_5(preds, labels, token=5)
    assert prec == pytest.approx(1.0)
    assert rec == pytest.approx(1.0)
