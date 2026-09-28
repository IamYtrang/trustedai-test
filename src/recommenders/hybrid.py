"""Hybrid recommender: composes collaborative + content-based strategies."""

from __future__ import annotations

from data.models import ScoredMovie
from data.repository import DataRepository
from recommenders.base import RecommenderStrategy
from recommenders.collaborative import UserBasedCollaborativeRecommender
from recommenders.content_based import ContentBasedRecommender

# Movies with fewer ratings than this are excluded from the popularity
# fallback — a 5.0 average from a single rating is not a real signal.
_MIN_RATINGS_FOR_POPULARITY = 5
# How many content candidates to pull before re-ranking with a CF signal,
# so the blend has enough breadth to actually move the ranking.
_CONTENT_CANDIDATE_MULTIPLIER = 4


class HybridRecommender(RecommenderStrategy):
    """Blends collaborative filtering and content-based search as needed.

    Args:
        repository: Data access layer, used for the popularity fallback.
        collaborative: Collaborative filtering strategy.
        content_based: Content-based (plot embedding) strategy.
    """

    def __init__(
        self,
        repository: DataRepository,
        collaborative: UserBasedCollaborativeRecommender,
        content_based: ContentBasedRecommender,
    ) -> None:
        self._repository = repository
        self._collaborative = collaborative
        self._content_based = content_based

    def recommend(
        self,
        user_id: int | None = None,
        query: str | None = None,
        exclude_genres: list[str] | None = None,
        k: int = 10,
    ) -> list[ScoredMovie]:
        """See `RecommenderStrategy.recommend`."""
        exclude_genres = exclude_genres or []

        if query:
            return self._recommend_from_query(user_id, query, exclude_genres, k)

        if user_id is not None:
            cf_results = self._collaborative.recommend(user_id=user_id, exclude_genres=exclude_genres, k=k)
            if cf_results:
                return cf_results
            already_rated = {r.movie_id for r in self._repository.get_user_ratings(user_id)}
            return self._popularity_fallback(exclude_genres, k, exclude_movie_ids=already_rated)

        return self._popularity_fallback(exclude_genres, k)

    def _recommend_from_query(
        self, user_id: int | None, query: str, exclude_genres: list[str], k: int
    ) -> list[ScoredMovie]:
        """Content search, re-ranked with a collaborative signal when available."""
        candidates = self._content_based.search(
            query, exclude_genres=exclude_genres, k=max(k * _CONTENT_CANDIDATE_MULTIPLIER, 20)
        )
        if not candidates:
            return []
        if user_id is None:
            return candidates[:k]

        cf_scores = self._collaborative.score_candidates(user_id, [c.movie.movie_id for c in candidates])
        if not cf_scores:
            return candidates[:k]

        blended: list[ScoredMovie] = []
        for candidate in candidates:
            cf_score = cf_scores.get(candidate.movie.movie_id)
            if cf_score is None:
                blended.append(candidate)
                continue
            blended.append(
                ScoredMovie(
                    movie=candidate.movie,
                    score=0.5 * candidate.score + 0.5 * (cf_score / 5.0),
                    reason=f"{candidate.reason}; taste-neighbors also rate it ~{cf_score:.1f}/5",
                )
            )
        blended.sort(key=lambda s: s.score, reverse=True)
        return blended[:k]

    def _popularity_fallback(
        self, exclude_genres: list[str], k: int, exclude_movie_ids: set[int] | None = None
    ) -> list[ScoredMovie]:
        """Rank by average rating among movies with enough ratings to trust."""
        ratings_df = self._repository.ratings_dataframe()
        stats = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"])
        stats = stats[stats["count"] >= _MIN_RATINGS_FOR_POPULARITY].sort_values("mean", ascending=False)

        exclude_genres_set = set(exclude_genres)
        exclude_movie_ids = exclude_movie_ids or set()

        results: list[ScoredMovie] = []
        for movie_id, row in stats.iterrows():
            if movie_id in exclude_movie_ids:
                continue
            movie = self._repository.get_movie(int(movie_id))
            if movie is None or exclude_genres_set & set(movie.genres):
                continue
            results.append(
                ScoredMovie(
                    movie=movie,
                    score=float(row["mean"]),
                    reason=f"Popular pick: {int(row['count'])} ratings, avg {row['mean']:.1f}/5",
                )
            )
            if len(results) >= k:
                break
        return results
