"""Strategy interface for recommendation algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

from data.models import ScoredMovie


class RecommenderStrategy(ABC):
    """Common interface for anything that can rank movies for a user or query.

    Concrete strategies (collaborative, content-based, hybrid) are
    interchangeable at runtime since they all expose this same method,
    which is what lets `RecommendForUserTool` stay agnostic to which
    algorithm actually produced the ranking.
    """

    @abstractmethod
    def recommend(
        self,
        user_id: int | None = None,
        query: str | None = None,
        exclude_genres: list[str] | None = None,
        k: int = 10,
    ) -> list[ScoredMovie]:
        """Rank movies for a user and/or a free-text query.

        Args:
            user_id: User to personalize for, if personalization is
                supported/available for this strategy.
            query: Free-text description to match against, if content
                matching is supported/available for this strategy.
            exclude_genres: Genres to filter out of the results entirely.
            k: Maximum number of results to return.

        Returns:
            Up to `k` `ScoredMovie` results, best first.
        """
        raise NotImplementedError
