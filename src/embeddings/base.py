"""Adapter interface for turning text into embedding vectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    """Common interface for any text embedding backend."""

    @abstractmethod
    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of long documents (e.g. movie plots).

        Args:
            texts: Documents to embed.

        Returns:
            Array of shape `(len(texts), embedding_dim)`, L2-normalized so
            dot product equals cosine similarity.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        """Embed a single short search query.

        Some models (like BGE) use an asymmetric scheme where queries and
        passages are encoded differently, so this is kept as a separate
        method rather than reusing `encode_passages`.

        Args:
            text: Free-text search query.

        Returns:
            Array of shape `(embedding_dim,)`, L2-normalized.
        """
        raise NotImplementedError
