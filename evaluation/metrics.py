"""Ranking metrics: Precision@k, Recall@k, NDCG@k.
"""

from __future__ import annotations

import math


def precision_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of the top-k recommendations that are relevant.

    Args:
        recommended_ids: Recommended movie ids, ranked best first.
        relevant_ids: Ground-truth "liked" movie ids for this user.
        k: Cutoff.

    Returns:
        Precision@k, in [0, 1]. 0.0 if `recommended_ids` is empty.
    """
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k recommendations.

    Args:
        recommended_ids: Recommended movie ids, ranked best first.
        relevant_ids: Ground-truth "liked" movie ids for this user.
        k: Cutoff.

    Returns:
        Recall@k, in [0, 1]. 0.0 if there are no relevant items.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(recommended_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Rewards relevant items appearing earlier in the ranking, unlike
    precision/recall which treat every position in the top-k equally.

    Args:
        recommended_ids: Recommended movie ids, ranked best first.
        relevant_ids: Ground-truth "liked" movie ids for this user.
        k: Cutoff.

    Returns:
        NDCG@k, in [0, 1]. 0.0 if there are no relevant items.
    """
    if not relevant_ids:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 2)  # rank is 0-indexed; +2 so rank 0 -> log2(2)
        for rank, mid in enumerate(recommended_ids[:k])
        if mid in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
