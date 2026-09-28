"""Repository pattern: the single place that touches pandas/CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.models import Movie, Rating, UserProfile


class DataRepository:
    """Indexes movies/ratings/tags DataFrames for fast in-memory lookups.

    Args:
        movies_df: Columns `movieId, title, year, genres, plot`.
        ratings_df: Columns `userId, movieId, rating, timestamp`.
        tags_df: Columns `userId, movieId, tag, timestamp`.
    """

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, tags_df: pd.DataFrame) -> None:
        self._movies_df = movies_df
        self._ratings_df = ratings_df
        self._tags_df = tags_df

        self._movies_by_id: dict[int, Movie] = {
            row.movieId: Movie(
                movie_id=row.movieId,
                title=row.title,
                year=int(row.year),
                genres=[g for g in row.genres.split("|") if g != "(no genres listed)"],
                plot=row.plot if isinstance(row.plot, str) else "",
            )
            for row in self._movies_df.itertuples(index=False)
        }

    @classmethod
    def from_csv_dir(cls, data_dir: Path) -> "DataRepository":
        """Build a repository by loading the standard MovieLens CSVs from disk.

        Args:
            data_dir: Directory containing `movies_with_plots.csv`,
                `ratings.csv`, and `tags.csv`.

        Returns:
            A ready-to-use `DataRepository`.
        """
        return cls(
            movies_df=pd.read_csv(data_dir / "movies_with_plots.csv"),
            ratings_df=pd.read_csv(data_dir / "ratings.csv"),
            tags_df=pd.read_csv(data_dir / "tags.csv"),
        )

    def with_ratings(self, ratings_df: pd.DataFrame) -> "DataRepository":
        """Build a copy of this repository with a different ratings table.

        Lets the evaluation harness train a recommender on a holdout split
        of ratings while reusing the same movie/tag catalog, without
        re-reading CSVs from disk.

        Args:
            ratings_df: Replacement ratings table.

        Returns:
            A new `DataRepository` sharing this instance's movies/tags.
        """
        return DataRepository(movies_df=self._movies_df, ratings_df=ratings_df, tags_df=self._tags_df)

    # -- Movies ---------------------------------------------------------

    def get_movie(self, movie_id: int) -> Movie | None:
        """Look up a single movie by id.

        Args:
            movie_id: MovieLens movie identifier.

        Returns:
            The `Movie`, or `None` if no movie with that id exists.
        """
        return self._movies_by_id.get(movie_id)

    def all_movies(self) -> list[Movie]:
        """Return every movie in the catalog.

        Returns:
            All `Movie` objects, in the order they appear in the source CSV.
        """
        return list(self._movies_by_id.values())

    # -- Ratings ----------------------------------------------------------

    def ratings_dataframe(self) -> pd.DataFrame:
        """Expose the raw ratings table for recommenders that need bulk access.

        Returns:
            A copy-free view of the `userId, movieId, rating, timestamp`
            DataFrame. Callers must treat it as read-only.
        """
        return self._ratings_df

    def tags_dataframe(self) -> pd.DataFrame:
        """Expose the raw tags table for bulk access (e.g. building the SQL store).

        Returns:
            A copy-free view of the `userId, movieId, tag, timestamp`
            DataFrame. Callers must treat it as read-only.
        """
        return self._tags_df

    def get_user_ratings(self, user_id: int) -> list[Rating]:
        """Get every rating a user has made.

        Args:
            user_id: MovieLens user identifier.

        Returns:
            The user's ratings, unsorted. Empty list if the user is unknown.
        """
        rows = self._ratings_df[self._ratings_df["userId"] == user_id]
        return [
            Rating(user_id=r.userId, movie_id=r.movieId, rating=r.rating, timestamp=r.timestamp)
            for r in rows.itertuples(index=False)
        ]

    def get_movie_ratings(self, movie_id: int) -> list[Rating]:
        """Get every rating a movie has received.

        Args:
            movie_id: MovieLens movie identifier.

        Returns:
            The movie's ratings, unsorted. Empty list if unrated.
        """
        rows = self._ratings_df[self._ratings_df["movieId"] == movie_id]
        return [
            Rating(user_id=r.userId, movie_id=r.movieId, rating=r.rating, timestamp=r.timestamp)
            for r in rows.itertuples(index=False)
        ]

    def known_user_ids(self) -> list[int]:
        """List every user id present in the ratings table.

        Returns:
            Sorted list of user ids.
        """
        return sorted(self._ratings_df["userId"].unique().tolist())

    def build_user_profile(self, user_id: int) -> UserProfile | None:
        """Aggregate a user's rating history into a `UserProfile`.

        Args:
            user_id: MovieLens user identifier.

        Returns:
            The user's profile (genre counts, average rating), or `None` if
            the user has no ratings.
        """
        ratings = self.get_user_ratings(user_id)
        if not ratings:
            return None

        genre_counts: dict[str, int] = {}
        for rating in ratings:
            movie = self.get_movie(rating.movie_id)
            if movie is None:
                continue
            for genre in movie.genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        return UserProfile(
            user_id=user_id,
            num_ratings=len(ratings),
            avg_rating=sum(r.rating for r in ratings) / len(ratings),
            genre_counts=genre_counts,
        )

    # -- Tags ---------------------------------------------------------------

    def catalog_genre_distribution(self) -> dict[str, int]:
        """Count how many movies in the whole catalog carry each genre.

        Used as the baseline to compare a single user's genre profile
        against, to surface "blind spot" genres.

        Returns:
            Mapping of genre name to number of movies with that genre.
        """
        counts: dict[str, int] = {}
        for movie in self._movies_by_id.values():
            for genre in movie.genres:
                counts[genre] = counts.get(genre, 0) + 1
        return counts

    def genre_gaps(self, user_id: int, min_catalog_count: int = 30) -> list[dict] | None:
        """Compare a user's genre profile against the catalog to find under-explored genres.

        Used by `DiscoveryRecommender` to pick which under-explored genres
        to recommend from. There is no dedicated "blind spots" tool — when
        a user asks about their blind spots directly, the assistant answers
        by having the LLM write SQL against `query_dataset` instead (see the
        worked example in `tools/query_tool.py`), not through this method.

        Args:
            user_id: MovieLens user identifier.
            min_catalog_count: Minimum number of catalog movies a genre must
                have to be considered — filters out rare genres that would
                otherwise dominate by having a tiny, noisy denominator.

        Returns:
            Per-genre dicts with `genre`, `user_share_of_ratings`,
            `catalog_share_of_movies`, and `gap` (catalog share minus user
            share — higher means more under-explored), sorted by `gap`
            descending. `None` if the user has no ratings.
        """
        profile = self.build_user_profile(user_id)
        if profile is None:
            return None

        catalog_dist = self.catalog_genre_distribution()
        total_user_ratings = sum(profile.genre_counts.values()) or 1
        total_catalog = sum(catalog_dist.values()) or 1

        gaps = []
        for genre, catalog_count in catalog_dist.items():
            if catalog_count < min_catalog_count:
                continue
            catalog_share = catalog_count / total_catalog
            user_share = profile.genre_counts.get(genre, 0) / total_user_ratings
            gaps.append(
                {
                    "genre": genre,
                    "user_share_of_ratings": user_share,
                    "catalog_share_of_movies": catalog_share,
                    "gap": catalog_share - user_share,
                }
            )

        gaps.sort(key=lambda g: g["gap"], reverse=True)
        return gaps
