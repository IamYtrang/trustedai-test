"""Read-only SQL view over the dataset, backing the agent's `query_dataset` tool.

Tables:
    movies(movie_id, title, year, plot)
    movie_genres(movie_id, genre)         -- one row per (movie, genre)
    ratings(user_id, movie_id, rating, timestamp)
    tags(user_id, movie_id, tag, timestamp)
    user_similarity(user_id, other_user_id, similarity)  -- precomputed top-K CF neighbors per user
"""

from __future__ import annotations

import sqlite3
from typing import Any

from data.repository import DataRepository
from recommenders.collaborative import UserBasedCollaborativeRecommender

# Neighbors precomputed per user; generous enough that "similar users"
# questions rarely need more than this, without recomputing at query time.
_NEIGHBORS_PER_USER = 30


class SqlDataStore:
    """Builds and serves a read-only, in-memory SQL view of the dataset.

    Args:
        repository: Data access layer, source of movies/ratings/tags.
        collaborative: Used once at construction time to precompute each
            user's taste-neighbors into the `user_similarity` table.
    """

    def __init__(self, repository: DataRepository, collaborative: UserBasedCollaborativeRecommender) -> None:
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._build_schema(repository, collaborative)
        # Enforced from here on: any write attempt raises rather than
        # silently succeeding. This is the real safety boundary, not just
        # the tool-level "does it start with SELECT" check.
        self._connection.execute("PRAGMA query_only = ON;")

    def _build_schema(self, repository: DataRepository, collaborative: UserBasedCollaborativeRecommender) -> None:
        cur = self._connection.cursor()

        cur.execute("CREATE TABLE movies (movie_id INTEGER PRIMARY KEY, title TEXT, year INTEGER, plot TEXT)")
        cur.execute("CREATE TABLE movie_genres (movie_id INTEGER, genre TEXT)")
        movies = repository.all_movies()
        cur.executemany("INSERT INTO movies VALUES (?, ?, ?, ?)", [(m.movie_id, m.title, m.year, m.plot) for m in movies])
        cur.executemany("INSERT INTO movie_genres VALUES (?, ?)", [(m.movie_id, g) for m in movies for g in m.genres])

        cur.execute("CREATE TABLE ratings (user_id INTEGER, movie_id INTEGER, rating REAL, timestamp INTEGER)")
        ratings_df = repository.ratings_dataframe()
        cur.executemany(
            "INSERT INTO ratings VALUES (?, ?, ?, ?)",
            ratings_df[["userId", "movieId", "rating", "timestamp"]].values.tolist(),
        )

        cur.execute("CREATE TABLE tags (user_id INTEGER, movie_id INTEGER, tag TEXT, timestamp INTEGER)")
        tags_df = repository.tags_dataframe()
        cur.executemany(
            "INSERT INTO tags VALUES (?, ?, ?, ?)",
            tags_df[["userId", "movieId", "tag", "timestamp"]].values.tolist(),
        )

        cur.execute("CREATE TABLE user_similarity (user_id INTEGER, other_user_id INTEGER, similarity REAL)")
        neighbor_rows = [
            (user_id, other_id, similarity)
            for user_id in repository.known_user_ids()
            for other_id, similarity in collaborative.find_similar_users(user_id, k=_NEIGHBORS_PER_USER)
        ]
        cur.executemany("INSERT INTO user_similarity VALUES (?, ?, ?)", neighbor_rows)

        cur.execute("CREATE INDEX idx_ratings_movie ON ratings(movie_id)")
        cur.execute("CREATE INDEX idx_ratings_user ON ratings(user_id)")
        cur.execute("CREATE INDEX idx_genres_movie ON movie_genres(movie_id)")
        cur.execute("CREATE INDEX idx_similarity_user ON user_similarity(user_id)")
        self._connection.commit()

    def execute_readonly_query(self, sql: str, max_rows: int = 200) -> dict[str, Any]:
        """Run a read-only SQL query and return its rows.

        Args:
            sql: A single SQL statement. Only reads succeed — the
                connection is opened with `PRAGMA query_only = ON`, so any
                write attempt raises rather than silently no-op'ing.
            max_rows: Caps how many rows are returned, to keep results
                small enough for the LLM's context window.

        Returns:
            `{"columns": [...], "rows": [...], "truncated": bool}` on
            success, or `{"error": "..."}` with SQLite's own message on
            failure. The error is surfaced verbatim, not swallowed, so the
            LLM can see *why* a query failed (e.g. "no such column: foo")
            and correct it itself — the same self-correcting loop a shell
            gives a `bash` tool.
        """
        try:
            cursor = self._connection.execute(sql)
        except sqlite3.Error as exc:
            return {"error": str(exc)}

        columns = [description[0] for description in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        return {
            "columns": columns,
            "rows": [dict(row) for row in rows[:max_rows]],
            "truncated": truncated,
        }
