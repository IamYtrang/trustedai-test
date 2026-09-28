"""Tool for the one capability that genuinely isn't plain SQL: ranking."""

from __future__ import annotations

from typing import Any

from agent.prompt import RECOMMEND_FOR_USER_DESCRIPTION, RECOMMEND_MODE_PARAM_DESCRIPTION
from data.repository import DataRepository
from recommenders.base import RecommenderStrategy
from tools._serialization import scored_movie_to_dict
from tools.base import BaseTool

# Every strategy truncates to the requested k internally, with no idea a
# rating filter is coming next — so when min_avg_rating is set, ask for more
# candidates than needed and filter those, instead of filtering an
# already-truncated top-k (which could silently drop from k results to 0
# even though qualifying movies exist further down the ranking).
_MIN_RATING_OVERSAMPLE_MULTIPLIER = 30


class RecommendForUserTool(BaseTool):
    """Produces a ranked recommendation list using the selected strategy.

    `mode` picks a `RecommenderStrategy` from a dict injected at
    construction time, so a new mode is just a new dict entry in
    `api/main.py` — this class doesn't change (Open/Closed). Also covers
    plain content search ("find me a movie like X"): a `query` with no
    `user_id` resolves to the same content-based ranking a dedicated search
    tool would do, so there's no separate tool for that case.
    """

    name = "recommend_movies"
    description = RECOMMEND_FOR_USER_DESCRIPTION
    parameters_schema = {
        "type": "object",
        "properties": {
            "user_id": {"type": "integer", "description": "User id to personalize for."},
            "query": {"type": "string", "description": "Free-text theme/plot description to match."},
            "exclude_genres": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Genres to exclude, e.g. ['Animation'].",
            },
            "min_avg_rating": {
                "type": "number",
                "description": "Minimum average rating (0.5-5.0) a result must have, across all its raters. "
                "Movies with no ratings are excluded when this is set.",
            },
            "mode": {
                "type": "string",
                "enum": ["personalized", "discover"],
                "description": RECOMMEND_MODE_PARAM_DESCRIPTION,
                "default": "personalized",
            },
            "k": {"type": "integer", "description": "Max results to return.", "default": 10},
        },
        "required": [],
    }

    def __init__(self, strategies: dict[str, RecommenderStrategy], repository: DataRepository) -> None:
        self._strategies = strategies
        self._repository = repository

    def run(self, **kwargs: Any) -> dict[str, Any]:
        mode = kwargs.get("mode", "personalized")
        strategy = self._strategies.get(mode)
        if strategy is None:
            return {"error": f"Unknown mode '{mode}'. Valid modes: {sorted(self._strategies)}."}

        user_id = kwargs.get("user_id")
        k = int(kwargs.get("k", 10))
        min_avg_rating = kwargs.get("min_avg_rating")
        strategy_k = k * _MIN_RATING_OVERSAMPLE_MULTIPLIER if min_avg_rating is not None else k

        results = strategy.recommend(
            user_id=int(user_id) if user_id is not None else None,
            query=kwargs.get("query"),
            exclude_genres=kwargs.get("exclude_genres"),
            k=strategy_k,
        )
        results = self._apply_min_avg_rating(results, min_avg_rating)[:k]
        if not results:
            return {"error": "No recommendations could be generated for these constraints."}
        return {"results": [scored_movie_to_dict(r) for r in results]}

    def _apply_min_avg_rating(self, results: list, min_avg_rating: float | None) -> list:
        """Drop results below a minimum average rating, regardless of which
        strategy produced them (their `score` field has different meanings
        per strategy, so this re-derives the real average rating instead)."""
        if min_avg_rating is None:
            return results
        kept = []
        for scored in results:
            ratings = self._repository.get_movie_ratings(scored.movie.movie_id)
            if not ratings:
                continue
            avg = sum(r.rating for r in ratings) / len(ratings)
            if avg >= min_avg_rating:
                kept.append(scored)
        return kept
