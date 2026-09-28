"""Tests for the recommender Strategy implementations.

Uses a small deterministic fake embedding model (bag-of-words over a fixed
vocabulary) instead of loading the real BGE model, so these tests run fast
and don't depend on network/model downloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.repository import DataRepository
from embeddings.base import EmbeddingModel
from recommenders.collaborative import UserBasedCollaborativeRecommender
from recommenders.content_based import ContentBasedRecommender
from recommenders.discovery import DiscoveryRecommender
from recommenders.hybrid import HybridRecommender

_VOCAB = ["cowboy", "spaceman", "detectives", "killer", "sins", "cop", "thieves", "cartoon", "children"]


class FakeEmbeddingModel(EmbeddingModel):
    """Deterministic bag-of-words embedding for tests, no model download."""

    def _vectorize(self, text: str) -> np.ndarray:
        lower = text.lower()
        vec = np.array([1.0 if word in lower else 0.0 for word in _VOCAB])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vectorize(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vectorize(text)


@pytest.fixture
def collaborative(repository: DataRepository) -> UserBasedCollaborativeRecommender:
    return UserBasedCollaborativeRecommender(repository)


@pytest.fixture
def content_based(repository: DataRepository) -> ContentBasedRecommender:
    embedder = FakeEmbeddingModel()
    movies = repository.all_movies()
    embeddings = embedder.encode_passages([m.plot for m in movies])
    return ContentBasedRecommender(repository, embedder, [m.movie_id for m in movies], embeddings)


class TestCollaborativeRecommender:
    def test_find_similar_users_excludes_self(self, collaborative: UserBasedCollaborativeRecommender) -> None:
        neighbors = collaborative.find_similar_users(1, k=5)
        assert all(user_id != 1 for user_id, _ in neighbors)

    def test_find_similar_users_ranks_user_2_above_user_3_for_user_1(
        self, collaborative: UserBasedCollaborativeRecommender
    ) -> None:
        # User 1 and user 2 both like movies 1 and 2; user 3 only overlaps on movie 4.
        neighbors = collaborative.find_similar_users(1, k=5)
        neighbor_ids = [uid for uid, _ in neighbors]
        assert neighbor_ids and neighbor_ids[0] == 2

    def test_find_similar_users_returns_empty_for_unknown_user(
        self, collaborative: UserBasedCollaborativeRecommender
    ) -> None:
        assert collaborative.find_similar_users(999, k=5) == []

    def test_recommend_excludes_already_rated_movies(
        self, collaborative: UserBasedCollaborativeRecommender
    ) -> None:
        results = collaborative.recommend(user_id=1, k=10)
        recommended_ids = {r.movie.movie_id for r in results}
        assert 1 not in recommended_ids  # user 1 already rated movie 1
        assert 2 not in recommended_ids  # user 1 already rated movie 2

    def test_recommend_respects_exclude_genres(self, collaborative: UserBasedCollaborativeRecommender) -> None:
        results = collaborative.recommend(user_id=1, exclude_genres=["Action"], k=10)
        assert all("Action" not in r.movie.genres for r in results)


class TestContentBasedRecommender:
    def test_search_ranks_matching_plot_first(self, content_based: ContentBasedRecommender) -> None:
        results = content_based.search("detectives hunt a killer with sins", k=3)
        assert results
        assert results[0].movie.movie_id == 2  # Seven

    def test_search_respects_exclude_genres(self, content_based: ContentBasedRecommender) -> None:
        results = content_based.search("cartoon for children", exclude_genres=["Animation"], k=5)
        assert all("Animation" not in r.movie.genres for r in results)

    def test_search_respects_min_avg_rating(self, content_based: ContentBasedRecommender) -> None:
        # Movie 5 has no ratings at all, so any min_avg_rating filter excludes it.
        results = content_based.search("cartoon for children", min_avg_rating=1.0, k=5)
        assert all(r.movie.movie_id != 5 for r in results)


class TestHybridRecommender:
    @pytest.fixture
    def hybrid(
        self,
        repository: DataRepository,
        collaborative: UserBasedCollaborativeRecommender,
        content_based: ContentBasedRecommender,
    ) -> HybridRecommender:
        return HybridRecommender(repository, collaborative, content_based)

    def test_falls_back_to_popularity_without_crashing_when_no_user_or_query(
        self, hybrid: HybridRecommender
    ) -> None:
        # The popularity fallback requires >=5 ratings per movie; this tiny
        # fixture never reaches that, so the meaningful assertion is that it
        # degrades to an empty list instead of raising.
        results = hybrid.recommend(k=5)
        assert results == []

    def test_uses_content_search_when_query_given_without_user(self, hybrid: HybridRecommender) -> None:
        results = hybrid.recommend(query="detectives hunt a killer", k=3)
        assert results
        assert results[0].movie.movie_id == 2

    def test_falls_back_to_popularity_for_unknown_user(self, hybrid: HybridRecommender) -> None:
        # Unknown user_id has no collaborative signal; should not raise.
        results = hybrid.recommend(user_id=999, k=5)
        assert isinstance(results, list)


class TestDiscoveryRecommender:
    """User 1 rated Toy Story (Adventure/Animation/Children/Comedy/Fantasy),
    Seven (Mystery/Thriller), and Heat (Action/Crime/Thriller) — never a Drama
    movie, even though the catalog fixture has one (Seven Pounds). That makes
    Drama their biggest genre gap, which is exactly what discovery mode should
    target.
    """

    @pytest.fixture
    def discovery(self, repository: DataRepository) -> DiscoveryRecommender:
        # Thresholds lowered to fit the tiny fixture (production defaults are
        # tuned for the real ~5,000-movie catalog).
        return DiscoveryRecommender(repository, min_ratings_for_quality=1, min_catalog_genre_count=1)

    def test_targets_users_biggest_genre_gap(self, discovery: DiscoveryRecommender) -> None:
        results = discovery.recommend(user_id=1, k=5)
        assert len(results) == 1
        assert results[0].movie.movie_id == 3  # Seven Pounds — the only rated Drama movie
        assert "Drama" in results[0].reason

    def test_excludes_movies_the_user_already_rated(self, discovery: DiscoveryRecommender) -> None:
        results = discovery.recommend(user_id=1, k=5)
        recommended_ids = {r.movie.movie_id for r in results}
        assert recommended_ids.isdisjoint({1, 2, 4})  # user 1's own rated movies

    def test_respects_exclude_genres(self, discovery: DiscoveryRecommender) -> None:
        # Excluding Drama removes the only movie that would otherwise match;
        # the fixture has no rated movie in the next-best gap genres either.
        results = discovery.recommend(user_id=1, exclude_genres=["Drama"], k=5)
        assert results == []

    def test_falls_back_to_general_quality_ranking_for_unknown_user(
        self, discovery: DiscoveryRecommender
    ) -> None:
        # No profile to compute a gap from; should degrade to "well-regarded
        # overall" rather than raise or return nothing.
        results = discovery.recommend(user_id=999, k=5)
        assert results
        assert results[0].movie.movie_id == 1  # highest avg rating (4.75 from 2 ratings)
