"""
Unit tests for data_loader module.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_loader import (
    parse_genres,
    load_movies,
    load_ratings,
    compute_movie_stats,
    load_and_merge,
)


class TestDataLoader(unittest.TestCase):
    def test_parse_genres(self):
        self.assertEqual(parse_genres("Action|Adventure|Sci-Fi"), ["Action", "Adventure", "Sci-Fi"])
        self.assertEqual(parse_genres("Comedy"), ["Comedy"])
        self.assertEqual(parse_genres("(no genres listed)"), [])
        self.assertEqual(parse_genres(""), [])
        self.assertEqual(parse_genres(None), [])
        self.assertEqual(parse_genres(np.nan), [])

    def test_load_movies(self):
        movies = load_movies()
        self.assertFalse(movies.empty)
        self.assertIn("movieId", movies.columns)
        self.assertIn("title", movies.columns)
        self.assertIn("genres", movies.columns)
        self.assertTrue(isinstance(movies["genres"].iloc[0], list))
        # Verify invalid rows dropped
        self.assertFalse(movies["movieId"].isna().any())
        self.assertFalse(movies["title"].isna().any())

    def test_load_ratings(self):
        ratings = load_ratings()
        self.assertFalse(ratings.empty)
        self.assertIn("userId", ratings.columns)
        self.assertIn("movieId", ratings.columns)
        self.assertIn("rating", ratings.columns)
        self.assertFalse(ratings["userId"].isna().any())
        self.assertFalse(ratings["movieId"].isna().any())
        self.assertFalse(ratings["rating"].isna().any())

    def test_compute_movie_stats(self):
        ratings = load_ratings()
        stats = compute_movie_stats(ratings)
        self.assertIn("movieId", stats.columns)
        self.assertIn("mean_rating", stats.columns)
        self.assertIn("rating_count", stats.columns)

    def test_load_and_merge(self):
        df = load_and_merge()
        self.assertFalse(df.empty)
        for col in ["movieId", "title", "genres", "mean_rating", "rating_count"]:
            self.assertIn(col, df.columns)
        
        # Verify movie 1 stats: Toy Story ratings: 4.0, 5.0, 4.5, 3.5 -> count: 4, mean: 4.25
        movie_1 = df[df["movieId"] == 1].iloc[0]
        self.assertEqual(movie_1["rating_count"], 4)
        self.assertAlmostEqual(movie_1["mean_rating"], 4.25, places=2)

        # Movies with no ratings should have rating_count = 0
        unrated = df[df["rating_count"] == 0]
        self.assertTrue(len(unrated) > 0)


if __name__ == "__main__":
    unittest.main()
