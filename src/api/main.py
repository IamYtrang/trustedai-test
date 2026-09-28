from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from agent import ChatLogger, ConversationAgent, SessionStore
from api.schemas import ChatRequest, ChatResponse, ToolTraceEntry, TurnMetrics, UserSummary
from config import settings
from data import DataRepository
from data.sql_store import SqlDataStore
from embeddings import BGEEmbedder, EmbeddingCache
from llm import LLMClientFactory, estimate_cost_usd
from recommenders import (
    ContentBasedRecommender,
    DiscoveryRecommender,
    HybridRecommender,
    UserBasedCollaborativeRecommender,
)
from tools import QueryDatasetTool, RecommendForUserTool, ToolRegistry

STATIC_DIR = Path(__file__).resolve().parent / "static"

SUGGESTED_USER_IDS = [1, 15, 30]

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Everything a request handler needs, built once at startup.

    Attributes:
        repository: Data access layer.
        agent: The conversation agent, or `None` if no LLM API key is
            configured (the UI still loads; chat requests return a clear
            error instead of the process crashing at import time).
        sessions: Per-session message history store.
        chat_logger: Persistent JSONL audit log of chat turns.
    """

    repository: DataRepository
    agent: ConversationAgent | None
    sessions: SessionStore
    chat_logger: ChatLogger


def build_context() -> AppContext:
    """Construct every layer and wire them into an `AppContext`.

    Returns:
        The fully wired application context.
    """
    repository = DataRepository.from_csv_dir(settings.data_dir)
    collaborative = UserBasedCollaborativeRecommender(repository)

    movie_ids, embeddings = EmbeddingCache(settings.embeddings_dir).load()
    embedder = BGEEmbedder(settings.resolve_embedding_model_path())
    content_based = ContentBasedRecommender(repository, embedder, movie_ids, embeddings)
    hybrid = HybridRecommender(repository, collaborative, content_based)
    discovery = DiscoveryRecommender(repository)
    sql_store = SqlDataStore(repository, collaborative)

    tool_registry = ToolRegistry(
        [
            QueryDatasetTool(sql_store),
            RecommendForUserTool({"personalized": hybrid, "discover": discovery}, repository),
        ]
    )

    agent = None
    if settings.openai_api_key:
        llm_client = LLMClientFactory.create("openai", settings.openai_api_key, settings.openai_model)
        agent = ConversationAgent(llm_client, tool_registry, max_tool_turns=settings.max_agent_tool_turns)

    return AppContext(
        repository=repository,
        agent=agent,
        sessions=SessionStore(),
        chat_logger=ChatLogger(settings.logs_dir),
    )


def create_app() -> FastAPI:
    """Build the FastAPI application with routes bound to a fresh context.

    Returns:
        A ready-to-serve FastAPI app.
    """
    app = FastAPI(title="Movie Discovery Assistant")
    context = build_context()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/users", response_model=list[UserSummary])
    def list_suggested_users() -> list[UserSummary]:
        summaries = []
        for user_id in SUGGESTED_USER_IDS:
            profile = context.repository.build_user_profile(user_id)
            if profile is not None:
                summaries.append(
                    UserSummary(user_id=user_id, num_ratings=profile.num_ratings, avg_rating=round(profile.avg_rating, 2))
                )
        return summaries

    @app.get("/api/users/{user_id}", response_model=UserSummary)
    def get_user(user_id: int) -> UserSummary:
        profile = context.repository.build_user_profile(user_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"No user with id {user_id} in the dataset.")
        return UserSummary(user_id=user_id, num_ratings=profile.num_ratings, avg_rating=round(profile.avg_rating, 2))

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        if context.agent is None:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY is not configured. Set it and restart the server.",
            )

        turn_id = str(uuid.uuid4())
        started = time.perf_counter()
        history = context.sessions.get_history(request.session_id)
        if history is None:
            history = context.agent.initial_messages(request.user_id)

        try:
            result = context.agent.handle_message(history, request.message)
        except Exception as exc:
            logger.exception("Chat turn %s failed", turn_id)
            context.chat_logger.log_turn(
                {
                    "turn_id": turn_id,
                    "session_id": request.session_id,
                    "user_id": request.user_id,
                    "model": settings.openai_model,
                    "status": "error",
                    "message": request.message,
                    "error": f"{type(exc).__name__}: {exc}",
                    "total_latency_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            )
            raise

        context.sessions.save_history(request.session_id, result["history"])
        total_latency_ms = (time.perf_counter() - started) * 1000
        cost_usd = estimate_cost_usd(result["usage"], settings.llm_pricing())
        metrics = TurnMetrics(
            model=settings.openai_model,
            total_latency_ms=round(total_latency_ms, 1),
            cost_usd=cost_usd,
            **result["metrics"],
        )

        context.chat_logger.log_turn(
            {
                "turn_id": turn_id,
                "session_id": request.session_id,
                "user_id": request.user_id,
                "model": settings.openai_model,
                "status": "ok",
                "message": request.message,
                "reply": result["reply"],
                "tool_trace": result["tool_trace"],
                "metrics": metrics.model_dump(),
            }
        )

        return ChatResponse(
            turn_id=turn_id,
            reply=result["reply"] or "",
            tool_trace=[ToolTraceEntry(**entry) for entry in result["tool_trace"]],
            metrics=metrics,
        )

    return app


app = create_app()
