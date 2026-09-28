"""Shared pytest fixtures: a tiny, deterministic dataset instead of the full CSVs."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.repository import DataRepository

_MOVIES_CSV = """movieId,title,year,genres,plot
1,Toy Story,1995,Adventure|Animation|Children|Comedy|Fantasy,A cowboy doll is jealous of a new spaceman toy.
2,Seven (a.k.a. Se7en),1995,Mystery|Thriller,Two detectives hunt a serial killer who uses the seven deadly sins.
3,Seven Pounds,2008,Drama,A man tries to help seven strangers to atone for a past mistake.
4,Heat,1995,Action|Crime|Thriller,A cop pursues a crew of professional thieves.
5,Kids Movie,2001,Animation|Children,A cheerful cartoon for children.
"""

_RATINGS_CSV = """userId,movieId,rating,timestamp
1,1,5.0,1000
1,2,5.0,1001
1,4,4.0,1002
2,1,4.5,1003
2,2,4.5,1004
2,3,2.0,1005
3,4,5.0,1006
3,2,4.0,1007
"""

_TAGS_CSV = """userId,movieId,tag,timestamp
1,2,twist ending,1000
1,2,dark,1001
"""


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "movies_with_plots.csv").write_text(_MOVIES_CSV)
    (tmp_path / "ratings.csv").write_text(_RATINGS_CSV)
    (tmp_path / "tags.csv").write_text(_TAGS_CSV)
    return tmp_path


@pytest.fixture
def repository(data_dir: Path) -> DataRepository:
    return DataRepository.from_csv_dir(data_dir)
