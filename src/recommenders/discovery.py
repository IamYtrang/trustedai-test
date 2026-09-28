"""Discovery recommender: deliberately breaks the collaborative-filtering filter bubble.

"""

from __future__ import annotations

from data.models import ScoredMovie
from data.repository import DataRepository
from recommenders.base import RecommenderStrategy

_MIN_RATINGS_FOR_QUALITY = 5
_MIN_CATALOG_GENRE_COUNT = 30
_NUM_TARGET_GENRES = 3


class DiscoveryRecommender(RecommenderStrategy):
    """Recommends well-regarded movies in genres a user rarely explores.

    Args:
        repository: Data access layer.
        min_ratings_for_quality: Minimum ratings a movie needs to be
            trusted as "well-regarded" rather than a single enthusiastic
            rater. Exposed mainly so tests can use a small fixture dataset;
            production code should use the default.
        min_catalog_genre_count: Minimum catalog movies a genre needs to be
            considered a valid discovery target (see
            `DataRepository.genre_gaps`). Same testability note applies.
    """

    def __init__(
        self,
        repository: DataRepository,
        min_ratings_for_quality: int = _MIN_RATINGS_FOR_QUALITY,
        min_catalog_genre_count: int = _MIN_CATALOG_GENRE_COUNT,
    ) -> None:
        self._repository = repository
        self._min_ratings_for_quality = min_ratings_for_quality
        self._min_catalog_genre_count = min_catalog_genre_count

    def recommend(
        self,
        user_id: int | None = None,
        query: str | None = None,
        exclude_genres: list[str] | None = None,
        k: int = 10,
    ) -> list[ScoredMovie]:
        """See `RecommenderStrategy.recommend`. `query` is ignored (unsupported).

        Without a `user_id`, there is no taste profile to diverge from, so
        this falls back to general well-regarded picks (no genre targeting).
        """
        exclude_genres_set = set(exclude_genres or [])
        target_genres = self._target_genres(user_id, exclude_genres_set) if user_id is not None else set()
        already_rated = (
            {r.movie_id for r in self._repository.get_user_ratings(user_id)} if user_id is not None else set()
        )

        ratings_df = self._repository.ratings_dataframe()
        stats = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"])
        stats = stats[stats["count"] >= self._min_ratings_for_quality].sort_values("mean", ascending=False)

        results: list[ScoredMovie] = []
        for movie_id, row in stats.iterrows():
            if movie_id in already_rated:
                continue
            movie = self._repository.get_movie(int(movie_id))
            if movie is None or exclude_genres_set & set(movie.genres):
                continue
            matched_genres = target_genres & set(movie.genres)
            if target_genres and not matched_genres:
                continue  # only surface movies that actually address the gap we found

            genre_note = ", ".join(sorted(matched_genres)) if matched_genres else "general discovery"
            results.append(
                ScoredMovie(
                    movie=movie,
                    score=float(row["mean"]),
                    reason=(
                        f"Outside your usual taste ({genre_note}) but well-regarded: "
                        f"avg {row['mean']:.1f}/5 across {int(row['count'])} ratings"
                    ),
                )
            )
            if len(results) >= k:
                break

        return results

    def _target_genres(self, user_id: int, exclude_genres: set[str]) -> set[str]:
        """Pick the user's most under-explored genres to recommend from."""
        gaps = self._repository.genre_gaps(user_id, min_catalog_count=self._min_catalog_genre_count)
        if not gaps:
            return set()
        candidates = [g for g in gaps if g["genre"] not in exclude_genres]
        return {g["genre"] for g in candidates[:_NUM_TARGET_GENRES]}
