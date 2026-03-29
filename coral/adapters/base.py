"""CORAL v4 — Abstract adapter interface.

Adapters bridge the amodal reasoning core (which operates on d_1=512
embedding vectors) with task-specific input/output spaces.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseAdapter(ABC, nn.Module):
    """Abstract base class for CORAL adapters.

    Subclasses must implement encode() and decode().
    """

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw task input to d_1=512 embeddings.

        Args:
            x: Task-specific input tensor.

        Returns:
            [B, L, d_1] float tensor of embeddings.
        """
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode d_1=512 embeddings to task-specific output logits.

        Args:
            z: [B, L, d_1] embedding tensor from the reasoning core.

        Returns:
            Task-specific output tensor (e.g., [B, L, vocab_size] logits).
        """
        ...
