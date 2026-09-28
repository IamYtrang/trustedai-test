"""On-disk cache for precomputed plot embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class EmbeddingCache:
    """Reads/writes a matrix of plot embeddings aligned to movie ids.

    Args:
        cache_dir: Directory where `plot_embeddings.npy` and
            `plot_embeddings_movie_ids.json` are stored.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._matrix_path = cache_dir / "plot_embeddings.npy"
        self._ids_path = cache_dir / "plot_embeddings_movie_ids.json"

    def exists(self) -> bool:
        """Check whether a cached embedding matrix is present on disk.

        Returns:
            `True` if both the matrix and its id index exist.
        """
        return self._matrix_path.exists() and self._ids_path.exists()

    def save(self, movie_ids: list[int], embeddings: np.ndarray) -> None:
        """Persist an embedding matrix and its aligned movie-id index.

        Args:
            movie_ids: Movie id for each row of `embeddings`, same order.
            embeddings: Array of shape `(len(movie_ids), embedding_dim)`.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self._matrix_path, embeddings)
        self._ids_path.write_text(json.dumps(movie_ids))

    def load(self) -> tuple[list[int], np.ndarray]:
        """Load the cached embedding matrix and its movie-id index.

        Returns:
            Tuple of `(movie_ids, embeddings)`, with `embeddings` shaped
            `(len(movie_ids), embedding_dim)`.

        Raises:
            FileNotFoundError: If the cache has not been built yet — run
                `scripts/build_embeddings.py` first.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"No embedding cache at {self._cache_dir}. Run scripts/build_embeddings.py first."
            )
        movie_ids = json.loads(self._ids_path.read_text())
        embeddings = np.load(self._matrix_path)
        return movie_ids, embeddings
