"""User-based collaborative filtering."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data.models import ScoredMovie
from data.repository import DataRepository
from recommenders.base import RecommenderStrategy

# A recommendation only counts if at least this many similar users rated the
# movie — otherwise a single enthusiastic neighbor can dominate the score.
_MIN_SUPPORTING_USERS = 2
# How many neighbors to look at when aggregating recommendations. Larger
# than the k typically requested by `find_similar_users` callers so the
# recommendation pool has enough breadth.
_NEIGHBORHOOD_SIZE = 30


class UserBasedCollaborativeRecommender(RecommenderStrategy):
    """Collaborative filtering over the full user-item rating matrix.

    Args:
        repository: Data access layer used to resolve movie metadata and
            to read the raw ratings table.
    """

    def __init__(self, repository: DataRepository) -> None:
        self._repository = repository
        self._build_matrix()

    def _build_matrix(self) -> None:
        """Build the dense user x movie rating matrix and its centered form."""
        ratings_df = self._repository.ratings_dataframe()

        self._user_ids = sorted(ratings_df["userId"].unique().tolist())
        self._movie_ids = sorted(ratings_df["movieId"].unique().tolist())
        self._user_index = {uid: i for i, uid in enumerate(self._user_ids)}
        self._movie_index = {mid: i for i, mid in enumerate(self._movie_ids)}

        matrix = np.zeros((len(self._user_ids), len(self._movie_ids)), dtype=np.float32)
        rows = ratings_df["userId"].map(self._user_index).to_numpy()
        cols = ratings_df["movieId"].map(self._movie_index).to_numpy()
        matrix[rows, cols] = ratings_df["rating"].to_numpy(dtype=np.float32)

        self._matrix = matrix
        mask = matrix != 0
        with np.errstate(invalid="ignore"):
            user_means = np.divide(
                matrix.sum(axis=1), mask.sum(axis=1), out=np.zeros(len(self._user_ids)), where=mask.sum(axis=1) > 0
            )
        self._centered = np.where(mask, matrix - user_means[:, None], 0.0)

    def find_similar_users(self, user_id: int, k: int = 10) -> list[tuple[int, float]]:
        """Find the users with the most similar taste to the given user.

        Similarity is cosine similarity over mean-centered ratings, so it
        reflects agreement on *relative* preference (rating movies higher
        or lower than one's own average) rather than raw rating level.

        Args:
            user_id: Target user id.
            k: Number of neighbors to return.

        Returns:
            Up to `k` `(other_user_id, similarity)` pairs, best first, with
            similarity > 0 only. Empty if `user_id` is unknown or has no
            positively-similar neighbors.
        """
        if user_id not in self._user_index:
            return []

        idx = self._user_index[user_id]
        sims = cosine_similarity(self._centered[idx : idx + 1], self._centered)[0]
        sims[idx] = -np.inf

        top_indices = np.argsort(sims)[::-1][:k]
        return [
            (self._user_ids[i], float(sims[i]))
            for i in top_indices
            if sims[i] > 0
        ]

    def score_candidates(self, user_id: int, movie_ids: list[int], k: int = _NEIGHBORHOOD_SIZE) -> dict[int, float]:
        """Predict a taste-neighborhood rating for a specific set of movies.

        Cheaper than `recommend` when the caller already has a candidate
        list (e.g. from content-based search) and only wants a
        collaborative-filtering signal to blend in, not a full ranking.

        Args:
            user_id: Target user id.
            movie_ids: Candidate movies to score.
            k: Number of taste-neighbors to consider.

        Returns:
            Mapping of movie id to predicted rating, for the subset of
            `movie_ids` that at least one neighbor has rated. Empty if the
            user has no neighbors.
        """
        neighbors = self.find_similar_users(user_id, k=k)
        if not neighbors:
            return {}

        candidate_cols = {mid: self._movie_index[mid] for mid in movie_ids if mid in self._movie_index}
        weighted_sum: dict[int, float] = {}
        weight_sum: dict[int, float] = {}

        for other_user_id, sim in neighbors:
            other_idx = self._user_index[other_user_id]
            for movie_id, col in candidate_cols.items():
                rating = self._matrix[other_idx, col]
                if rating > 0:
                    weighted_sum[movie_id] = weighted_sum.get(movie_id, 0.0) + sim * rating
                    weight_sum[movie_id] = weight_sum.get(movie_id, 0.0) + sim

        return {
            mid: float(weighted_sum[mid] / weight_sum[mid]) for mid in weighted_sum if weight_sum[mid] > 0
        }

    def recommend(
        self,
        user_id: int | None = None,
        query: str | None = None,
        exclude_genres: list[str] | None = None,
        k: int = 10,
    ) -> list[ScoredMovie]:
        """See `RecommenderStrategy.recommend`. `query` is ignored (unsupported)."""
        if user_id is None or user_id not in self._user_index:
            return []

        user_idx = self._user_index[user_id]
        already_rated = set(np.nonzero(self._matrix[user_idx])[0].tolist())
        exclude_genres = set(exclude_genres or [])

        neighbors = self.find_similar_users(user_id, k=_NEIGHBORHOOD_SIZE)
        if not neighbors:
            return []

        weighted_sum = np.zeros(len(self._movie_ids))
        weight_sum = np.zeros(len(self._movie_ids))
        support = np.zeros(len(self._movie_ids), dtype=int)

        for other_user_id, sim in neighbors:
            other_idx = self._user_index[other_user_id]
            row = self._matrix[other_idx]
            rated_cols = np.nonzero(row)[0]
            weighted_sum[rated_cols] += sim * row[rated_cols]
            weight_sum[rated_cols] += sim
            support[rated_cols] += 1

        candidate_cols = [
            i
            for i in range(len(self._movie_ids))
            if i not in already_rated and support[i] >= _MIN_SUPPORTING_USERS
        ]
        candidate_cols.sort(key=lambda i: weighted_sum[i] / weight_sum[i], reverse=True)

        results: list[ScoredMovie] = []
        for i in candidate_cols:
            movie = self._repository.get_movie(self._movie_ids[i])
            if movie is None or exclude_genres & set(movie.genres):
                continue
            predicted_rating = weighted_sum[i] / weight_sum[i]
            results.append(
                ScoredMovie(
                    movie=movie,
                    score=predicted_rating,
                    reason=f"Liked by {support[i]} of your top {len(neighbors)} taste-neighbors "
                    f"(predicted rating {predicted_rating:.1f}/5)",
                )
            )
            if len(results) >= k:
                break

        return results
