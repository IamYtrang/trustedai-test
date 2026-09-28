"""Plain data containers shared across the application."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Movie:
    """A single movie and its metadata.

    Attributes:
        movie_id: MovieLens movie identifier.
        title: Movie title (may include the release year in parentheses).
        year: Release year.
        genres: Genre labels, e.g. ["Action", "Sci-Fi"].
        plot: Full plot summary text, empty string if unavailable.
    """

    movie_id: int
    title: str
    year: int
    genres: list[str]
    plot: str = ""


@dataclass(frozen=True)
class Rating:
    """A single user rating of a movie.

    Attributes:
        user_id: MovieLens user identifier.
        movie_id: MovieLens movie identifier.
        rating: Rating value, from 0.5 to 5.0.
        timestamp: Unix timestamp of when the rating was made.
    """

    user_id: int
    movie_id: int
    rating: float
    timestamp: int


@dataclass(frozen=True)
class UserProfile:
    """Aggregated view of a user's rating history.

    Attributes:
        user_id: MovieLens user identifier.
        num_ratings: Total number of ratings the user has made.
        avg_rating: Mean of all ratings given by the user.
        genre_counts: Mapping of genre name to number of rated movies in
            that genre (a movie with multiple genres counts once per genre).
    """

    user_id: int
    num_ratings: int
    avg_rating: float
    genre_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredMovie:
    """A movie with a relevance/preference score attached.

    Attributes:
        movie: The scored movie.
        score: Higher is more relevant/preferred. Scale depends on the
            recommender strategy that produced it (not directly comparable
            across strategies unless explicitly normalized).
        reason: Short human-readable explanation of why this movie was
            scored this way (e.g. which signal drove it).
    """

    movie: Movie
    score: float
    reason: str = ""
