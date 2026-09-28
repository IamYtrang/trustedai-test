"""Embedding layer: text-to-vector adapter plus its on-disk cache."""

from embeddings.base import EmbeddingModel
from embeddings.bge_embedder import BGEEmbedder
from embeddings.cache import EmbeddingCache

__all__ = ["BGEEmbedder", "EmbeddingCache", "EmbeddingModel"]
