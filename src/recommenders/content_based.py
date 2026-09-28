"""Content-based recommendation via plot-embedding similarity."""

from __future__ import annotations

import numpy as np

from data.models import ScoredMovie
from data.repository import DataRepository
from embeddings.base import EmbeddingModel


class ContentBasedRecommender:
    """Ranks movies by embedding similarity between their plot and a query.

    Not a `RecommenderStrategy` — it requires a `query`, so it's only used
    composed into `HybridRecommender` (a bare `query`, no `user_id`, routes
    here for the "find me a movie like X" case).

    Args:
        repository: Data access layer used to resolve movie metadata.
        embedder: Embedding backend used to encode the search query.
        movie_ids: Movie ids aligned to the rows of `embeddings`.
        embeddings: Precomputed, L2-normalized plot embeddings, shape
            `(len(movie_ids), embedding_dim)`.
    """

    def __init__(
        self,
        repository: DataRepository,
        embedder: EmbeddingModel,
        movie_ids: list[int],
        embeddings: np.ndarray,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._movie_ids = movie_ids
        self._embeddings = embeddings

    def search(
        self,
        query: str,
        exclude_genres: list[str] | None = None,
        min_avg_rating: float | None = None,
        k: int = 10,
    ) -> list[ScoredMovie]:
        """Find movies whose plot is semantically closest to a query.

        Args:
            query: Free-text description, e.g. "dark psychological thriller
                with a twist".
            exclude_genres: Genres to filter out of the results.
            min_avg_rating: If set, drop movies whose average rating (among
                movies that have any ratings) is below this value.
            k: Maximum number of results.

        Returns:
            Up to `k` `ScoredMovie` results, best cosine-similarity first.
        """
        query_vector = self._embedder.encode_query(query)
        similarities = self._embeddings @ query_vector

        exclude_genres = set(exclude_genres or [])
        order = np.argsort(similarities)[::-1]

        results: list[ScoredMovie] = []
        for i in order:
            movie = self._repository.get_movie(self._movie_ids[i])
            if movie is None or exclude_genres & set(movie.genres):
                continue
            if min_avg_rating is not None:
                movie_ratings = self._repository.get_movie_ratings(movie.movie_id)
                if not movie_ratings:
                    continue
                avg = sum(r.rating for r in movie_ratings) / len(movie_ratings)
                if avg < min_avg_rating:
                    continue
            results.append(
                ScoredMovie(
                    movie=movie,
                    score=float(similarities[i]),
                    reason=f"Plot matches '{query}' (similarity {similarities[i]:.2f})",
                )
            )
            if len(results) >= k:
                break

        return results
