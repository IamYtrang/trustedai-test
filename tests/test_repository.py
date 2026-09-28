"""Tests for `DataRepository`."""

from __future__ import annotations

from data.repository import DataRepository


def test_get_movie_returns_known_movie(repository: DataRepository) -> None:
    movie = repository.get_movie(1)
    assert movie is not None
    assert movie.title == "Toy Story"
    assert "Animation" in movie.genres


def test_get_movie_returns_none_for_unknown_id(repository: DataRepository) -> None:
    assert repository.get_movie(9999) is None


def test_build_user_profile_aggregates_ratings_and_genres(repository: DataRepository) -> None:
    profile = repository.build_user_profile(1)
    assert profile is not None
    assert profile.num_ratings == 3
    assert profile.avg_rating == (5.0 + 5.0 + 4.0) / 3
    # Both movie 2 (Seven, Mystery|Thriller) and movie 4 (Heat, Action|Crime|Thriller)
    # carry the Thriller genre.
    assert profile.genre_counts["Thriller"] == 2


def test_build_user_profile_returns_none_for_user_with_no_ratings(repository: DataRepository) -> None:
    assert repository.build_user_profile(999) is None


def test_catalog_genre_distribution_counts_every_movie(repository: DataRepository) -> None:
    dist = repository.catalog_genre_distribution()
    assert dist["Animation"] == 2  # Toy Story + Kids Movie
    assert dist["Thriller"] == 2  # Seven + Heat


def test_known_user_ids_is_sorted(repository: DataRepository) -> None:
    assert repository.known_user_ids() == [1, 2, 3]


def test_tags_dataframe_matches_source_tags(repository: DataRepository) -> None:
    tags_df = repository.tags_dataframe()
    assert len(tags_df) == 2
    assert set(tags_df["tag"]) == {"twist ending", "dark"}
