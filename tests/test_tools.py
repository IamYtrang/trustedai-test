"""Tests for tool implementations and the `ToolRegistry` dispatch mechanism."""

from __future__ import annotations

from typing import Any

import pytest

from data.models import ScoredMovie
from data.repository import DataRepository
from data.sql_store import SqlDataStore
from recommenders.base import RecommenderStrategy
from recommenders.collaborative import UserBasedCollaborativeRecommender
from tools.base import BaseTool
from tools.movie_tools import RecommendForUserTool
from tools.query_tool import QueryDatasetTool
from tools.registry import ToolRegistry


class BoomTool(BaseTool):
    """A tool that always raises, to test the registry's error handling."""

    name = "boom"
    description = "Always fails."
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")


class TestToolRegistry:
    def test_dispatch_unknown_tool_returns_error_dict(self) -> None:
        registry = ToolRegistry([BoomTool()])
        result = registry.dispatch("does_not_exist", {})
        assert "error" in result

    def test_dispatch_catches_tool_exceptions(self) -> None:
        registry = ToolRegistry([BoomTool()])
        result = registry.dispatch("boom", {})
        assert "error" in result
        assert "boom" in result["error"]

    def test_rejects_duplicate_tool_names(self) -> None:
        with pytest.raises(ValueError):
            ToolRegistry([BoomTool(), BoomTool()])

    def test_openai_tool_specs_include_every_tool(self, repository: DataRepository) -> None:
        sql_store = SqlDataStore(repository, UserBasedCollaborativeRecommender(repository))
        registry = ToolRegistry([QueryDatasetTool(sql_store), BoomTool()])
        specs = registry.openai_tool_specs()
        names = {spec["function"]["name"] for spec in specs}
        assert names == {"query_dataset", "boom"}


class TestQueryDatasetTool:
    @pytest.fixture
    def tool(self, repository: DataRepository) -> QueryDatasetTool:
        collaborative = UserBasedCollaborativeRecommender(repository)
        return QueryDatasetTool(SqlDataStore(repository, collaborative))

    def test_simple_select_over_movies(self, tool: QueryDatasetTool) -> None:
        result = tool.run(sql="SELECT title FROM movies WHERE title LIKE '%Seven%' ORDER BY title")
        assert "error" not in result
        titles = {row["title"] for row in result["rows"]}
        assert titles == {"Seven (a.k.a. Se7en)", "Seven Pounds"}

    def test_join_movies_and_genres(self, tool: QueryDatasetTool) -> None:
        # A user profile / genre breakdown question, expressed as a join.
        result = tool.run(
            sql=(
                "SELECT g.genre, COUNT(*) AS n FROM ratings r "
                "JOIN movie_genres g ON r.movie_id = g.movie_id "
                "WHERE r.user_id = 1 GROUP BY g.genre ORDER BY n DESC"
            )
        )
        rows = {row["genre"]: row["n"] for row in result["rows"]}
        # user 1 rated movies 1 (Toy Story), 2 (Seven), 4 (Heat).
        assert rows["Thriller"] == 2  # Seven + Heat both carry Thriller

    def test_similar_users_join_composes_correctly(self, tool: QueryDatasetTool) -> None:
        # This is the "what do people like me think of X" flow, expressed as
        # one SQL join instead of two separate bespoke tools.
        result = tool.run(
            sql=(
                "SELECT AVG(r.rating) AS avg_rating, COUNT(*) AS n FROM ratings r "
                "JOIN user_similarity s ON r.user_id = s.other_user_id "
                "WHERE s.user_id = 1 AND r.movie_id = "
                "(SELECT movie_id FROM movies WHERE title LIKE '%Seven (a.k.a%')"
            )
        )
        assert "error" not in result
        assert result["rows"][0]["n"] >= 0  # tiny fixture may have 0-1 neighbors; just must not error

    def test_rejects_write_statements(self, tool: QueryDatasetTool) -> None:
        result = tool.run(sql="DELETE FROM ratings WHERE user_id = 1")
        assert "error" in result

    def test_rejects_write_statements_even_if_enforcement_is_bypassed_at_tool_level(
        self, repository: DataRepository
    ) -> None:
        # Defense in depth: the SqlDataStore itself must reject writes via
        # PRAGMA query_only, independent of the tool's startswith() check.
        sql_store = SqlDataStore(repository, UserBasedCollaborativeRecommender(repository))
        result = sql_store.execute_readonly_query("DELETE FROM ratings")
        assert "error" in result

    def test_surfaces_sql_syntax_errors_for_self_correction(self, tool: QueryDatasetTool) -> None:
        result = tool.run(sql="SELECT * FROM not_a_real_table")
        assert "error" in result
        assert "not_a_real_table" in result["error"]


class FakeStrategy(RecommenderStrategy):
    """Returns one fixed, tagged result — for testing mode dispatch, not ranking logic."""

    def __init__(self, tag: str, movie) -> None:
        self._tag = tag
        self._movie = movie

    def recommend(self, user_id=None, query=None, exclude_genres=None, k=10) -> list[ScoredMovie]:
        return [ScoredMovie(movie=self._movie, score=1.0, reason=self._tag)]


class TestRecommendForUserTool:
    @pytest.fixture
    def tool(self, repository: DataRepository) -> RecommendForUserTool:
        movie = repository.get_movie(1)
        return RecommendForUserTool(
            {
                "personalized": FakeStrategy("personalized", movie),
                "discover": FakeStrategy("discover", movie),
            },
            repository,
        )

    def test_defaults_to_personalized_mode(self, tool: RecommendForUserTool) -> None:
        result = tool.run(user_id=1)
        assert result["results"][0]["reason"] == "personalized"

    def test_discover_mode_selects_the_discovery_strategy(self, tool: RecommendForUserTool) -> None:
        result = tool.run(user_id=1, mode="discover")
        assert result["results"][0]["reason"] == "discover"

    def test_unknown_mode_reports_error_without_crashing(self, tool: RecommendForUserTool) -> None:
        result = tool.run(user_id=1, mode="not_a_real_mode")
        assert "error" in result

    def test_min_avg_rating_keeps_results_at_or_above_the_threshold(self, tool: RecommendForUserTool) -> None:
        # Movie 1 (Toy Story) averages 4.75 in the fixture (ratings 5.0, 4.5).
        result = tool.run(user_id=1, min_avg_rating=4.75)
        assert result["results"][0]["reason"] == "personalized"

    def test_min_avg_rating_filters_out_results_below_the_threshold(self, tool: RecommendForUserTool) -> None:
        result = tool.run(user_id=1, min_avg_rating=5.0)
        assert "error" in result
