"""Shared helper for turning a `ScoredMovie` into a JSON-serializable dict."""

from __future__ import annotations

from data.models import ScoredMovie


def scored_movie_to_dict(scored: ScoredMovie) -> dict:
    """Convert a `ScoredMovie` into a compact dict for LLM/JSON consumption.

    Args:
        scored: Scored movie to serialize.

    Returns:
        Dict with movie fields (a truncated `plot_excerpt`, since full
        plots are too long to spend context tokens on), plus `score` and
        the recommender's plain-English `reason`.
    """
    movie = scored.movie
    return {
        "movie_id": movie.movie_id,
        "title": movie.title,
        "year": movie.year,
        "genres": movie.genres,
        "plot_excerpt": movie.plot[:400] + ("..." if len(movie.plot) > 400 else ""),
        "score": round(float(scored.score), 3),
        "reason": scored.reason,
    }
