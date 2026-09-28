"""Tests for `SqlDataStore`: schema correctness and the read-only boundary."""

from __future__ import annotations

import pytest

from data.repository import DataRepository
from data.sql_store import SqlDataStore
from recommenders.collaborative import UserBasedCollaborativeRecommender


@pytest.fixture
def sql_store(repository: DataRepository) -> SqlDataStore:
    return SqlDataStore(repository, UserBasedCollaborativeRecommender(repository))


def test_movies_table_matches_catalog(sql_store: SqlDataStore) -> None:
    result = sql_store.execute_readonly_query("SELECT COUNT(*) AS n FROM movies")
    assert result["rows"][0]["n"] == 5


def test_movie_genres_is_normalized_one_row_per_genre(sql_store: SqlDataStore) -> None:
    # Toy Story has 5 genres -> 5 rows in the junction table.
    result = sql_store.execute_readonly_query("SELECT COUNT(*) AS n FROM movie_genres WHERE movie_id = 1")
    assert result["rows"][0]["n"] == 5


def test_ratings_table_matches_fixture_count(sql_store: SqlDataStore) -> None:
    result = sql_store.execute_readonly_query("SELECT COUNT(*) AS n FROM ratings")
    assert result["rows"][0]["n"] == 8


def test_user_similarity_table_excludes_self(sql_store: SqlDataStore) -> None:
    result = sql_store.execute_readonly_query(
        "SELECT COUNT(*) AS n FROM user_similarity WHERE user_id = 1 AND other_user_id = 1"
    )
    assert result["rows"][0]["n"] == 0


def test_query_only_pragma_rejects_writes(sql_store: SqlDataStore) -> None:
    result = sql_store.execute_readonly_query("UPDATE ratings SET rating = 1.0 WHERE user_id = 1")
    assert "error" in result
    # And the data is actually untouched.
    check = sql_store.execute_readonly_query("SELECT rating FROM ratings WHERE user_id = 1 AND movie_id = 1")
    assert check["rows"][0]["rating"] == 5.0


def test_max_rows_truncates_and_flags_it(repository: DataRepository) -> None:
    sql_store = SqlDataStore(repository, UserBasedCollaborativeRecommender(repository))
    result = sql_store.execute_readonly_query("SELECT * FROM ratings", max_rows=2)
    assert len(result["rows"]) == 2
    assert result["truncated"] is True


def test_invalid_sql_returns_sqlite_error_message(sql_store: SqlDataStore) -> None:
    result = sql_store.execute_readonly_query("SELEKT * FROM movies")
    assert "error" in result
