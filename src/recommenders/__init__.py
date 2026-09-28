"""Recommender layer: the Strategy pattern implementations."""

from recommenders.base import RecommenderStrategy
from recommenders.collaborative import UserBasedCollaborativeRecommender
from recommenders.content_based import ContentBasedRecommender
from recommenders.discovery import DiscoveryRecommender
from recommenders.hybrid import HybridRecommender

__all__ = [
    "ContentBasedRecommender",
    "DiscoveryRecommender",
    "HybridRecommender",
    "RecommenderStrategy",
    "UserBasedCollaborativeRecommender",
]
