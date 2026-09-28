"""BGE (BAAI General Embedding) implementation of `EmbeddingModel`."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings.base import EmbeddingModel

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbedder(EmbeddingModel):
    """Sentence-embedding adapter backed by `BAAI/bge-small-en-v1.5`.

    Args:
        model_name: HuggingFace model id to load.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model = SentenceTransformer(model_name)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """See `EmbeddingModel.encode_passages`."""
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )

    def encode_query(self, text: str) -> np.ndarray:
        """See `EmbeddingModel.encode_query`."""
        vector = self._model.encode(
            _QUERY_INSTRUCTION + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector
