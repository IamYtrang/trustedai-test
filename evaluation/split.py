"""Time-based train/test holdout, per user."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Fraction of each user's ratings (by recency) held out for evaluation.
_HOLDOUT_FRACTION = 0.2
# Users need at least this many ratings to get a holdout at all — otherwise
# there isn't enough history left to train on after the split.
_MIN_RATINGS_FOR_SPLIT = 5


@dataclass
class TrainTestSplit:
    """A time-based train/test split of the ratings table.

    Attributes:
        train: Ratings to build the recommender from.
        test: Held-out (most recent) ratings per user, used for evaluation.
    """

    train: pd.DataFrame
    test: pd.DataFrame


def time_based_split(ratings: pd.DataFrame, holdout_fraction: float = _HOLDOUT_FRACTION) -> TrainTestSplit:
    """Hold out each user's most recent ratings for testing.

    Args:
        ratings: Full ratings table with `userId`, `movieId`, `rating`,
            `timestamp` columns.
        holdout_fraction: Fraction of each user's ratings (rounded down) to
            hold out, taken from the end of their rating timeline.

    Returns:
        A `TrainTestSplit`. Users with fewer than `_MIN_RATINGS_FOR_SPLIT`
        ratings are excluded from `test` entirely (all their ratings stay
        in `train`), since there isn't enough history to hold anything out.
    """
    train_parts = []
    test_parts = []

    for _, user_ratings in ratings.groupby("userId"):
        user_ratings = user_ratings.sort_values("timestamp")
        if len(user_ratings) < _MIN_RATINGS_FOR_SPLIT:
            train_parts.append(user_ratings)
            continue

        n_holdout = max(1, int(len(user_ratings) * holdout_fraction))
        train_parts.append(user_ratings.iloc[:-n_holdout])
        test_parts.append(user_ratings.iloc[-n_holdout:])

    train = pd.concat(train_parts, ignore_index=True) if train_parts else ratings.iloc[0:0]
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0]
    return TrainTestSplit(train=train, test=test)
