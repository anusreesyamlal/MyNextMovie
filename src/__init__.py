"""
MyNextMovie - Data loading and recommendation utilities.
"""

from .data_loader import (
    load_and_merge,
    load_movies,
    load_ratings,
    compute_movie_stats,
    parse_genres,
)
from .popularity_recommender import recommend_popular
from .content_recommender import recommend_similar

__all__ = [
    "load_and_merge",
    "load_movies",
    "load_ratings",
    "compute_movie_stats",
    "parse_genres",
    "recommend_popular",
    "recommend_similar",
]
