#!/usr/bin/env python3
"""Quantitative evaluation: collaborative filtering vs. a popularity baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k
from evaluation.split import time_based_split
from config import settings
from data.repository import DataRepository
from recommenders.collaborative import UserBasedCollaborativeRecommender

# A held-out rating at or above this is considered "liked" — chosen from the
# EDA notebook's rating distribution (positively skewed, mass at 3.5-4.0+),
# not an arbitrary pick.
_LIKED_THRESHOLD = 4.0
_K = 10
# Cohort boundaries: mirrors the EDA notebook's user rating-count analysis.
_SPARSE_USER_MAX_RATINGS = 20
_DENSE_USER_MIN_RATINGS = 100

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "metrics.json"


def _cohort(num_train_ratings: int) -> str:
    if num_train_ratings < _SPARSE_USER_MAX_RATINGS:
        return "sparse (<20 train ratings)"
    if num_train_ratings >= _DENSE_USER_MIN_RATINGS:
        return "dense (>=100 train ratings)"
    return "medium"


def _evaluate_recommender(name: str, recommend_fn, train_df, test_df) -> dict:
    """Evaluate one recommend_fn(user_id) -> list[movie_id] over all test users."""
    per_cohort: dict[str, list[dict]] = {}
    train_counts = train_df.groupby("userId").size()

    for user_id, user_test in test_df.groupby("userId"):
        relevant = set(user_test.loc[user_test["rating"] >= _LIKED_THRESHOLD, "movieId"])
        if not relevant:
            continue  # nothing this user "liked" in the holdout to evaluate against

        recommended_ids = recommend_fn(user_id)
        cohort = _cohort(int(train_counts.get(user_id, 0)))
        per_cohort.setdefault(cohort, []).append(
            {
                "precision": precision_at_k(recommended_ids, relevant, _K),
                "recall": recall_at_k(recommended_ids, relevant, _K),
                "ndcg": ndcg_at_k(recommended_ids, relevant, _K),
            }
        )

    summary = {"strategy": name, "cohorts": {}}
    all_scores: list[dict] = []
    for cohort, scores in per_cohort.items():
        all_scores.extend(scores)
        summary["cohorts"][cohort] = {
            "num_users": len(scores),
            "precision_at_10": round(sum(s["precision"] for s in scores) / len(scores), 4),
            "recall_at_10": round(sum(s["recall"] for s in scores) / len(scores), 4),
            "ndcg_at_10": round(sum(s["ndcg"] for s in scores) / len(scores), 4),
        }
    summary["overall"] = {
        "num_users": len(all_scores),
        "precision_at_10": round(sum(s["precision"] for s in all_scores) / len(all_scores), 4),
        "recall_at_10": round(sum(s["recall"] for s in all_scores) / len(all_scores), 4),
        "ndcg_at_10": round(sum(s["ndcg"] for s in all_scores) / len(all_scores), 4),
    }
    return summary


def run_eval() -> bool:
    print("Loading data and building time-based holdout split...")
    full_repository = DataRepository.from_csv_dir(settings.data_dir)
    split = time_based_split(full_repository.ratings_dataframe())
    print(f"Train ratings: {len(split.train)}, test ratings: {len(split.test)}")

    train_repository = full_repository.with_ratings(split.train)
    collaborative = UserBasedCollaborativeRecommender(train_repository)

    def cf_recommend(user_id: int) -> list[int]:
        return [r.movie.movie_id for r in collaborative.recommend(user_id=user_id, k=_K)]

    def popularity_recommend(user_id: int) -> list[int]:
        stats = split.train.groupby("movieId")["rating"].agg(["mean", "count"])
        stats = stats[stats["count"] >= 5].sort_values("mean", ascending=False)
        already_rated = set(split.train.loc[split.train["userId"] == user_id, "movieId"])
        return [int(mid) for mid in stats.index if mid not in already_rated][:_K]

    print("Evaluating collaborative filtering...")
    cf_summary = _evaluate_recommender("collaborative_filtering", cf_recommend, split.train, split.test)
    print("Evaluating popularity baseline...")
    pop_summary = _evaluate_recommender("popularity_baseline", popularity_recommend, split.train, split.test)

    results = {
        "liked_threshold": _LIKED_THRESHOLD,
        "k": _K,
        "results": [cf_summary, pop_summary],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {RESULTS_PATH}\n")
    print(json.dumps(results, indent=2))
    return True


if __name__ == "__main__":
    sys.exit(0 if run_eval() else 1)
