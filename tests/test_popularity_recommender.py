"""
Unit tests for popularity_recommender module.
"""

import unittest
import pandas as pd
from src.popularity_recommender import recommend_popular


class TestPopularityRecommender(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"movieId": 1, "title": "Movie A", "genres": ["Action", "Sci-Fi"], "mean_rating": 4.5, "rating_count": 10},
            {"movieId": 2, "title": "Movie B", "genres": ["Action"], "mean_rating": 4.5, "rating_count": 25},
            {"movieId": 3, "title": "Movie C", "genres": ["Comedy"], "mean_rating": 4.8, "rating_count": 2},
            {"movieId": 4, "title": "Movie D", "genres": ["Drama"], "mean_rating": 4.2, "rating_count": 50},
            {"movieId": 5, "title": "Movie E", "genres": ["Action", "Comedy"], "mean_rating": 3.9, "rating_count": 100},
        ])

    def test_genre_filter_case_insensitive(self):
        # Case insensitive genre match
        recs = recommend_popular(self.df, genre="action", min_ratings=0, top_n=10)
        self.assertEqual(len(recs), 3)
        self.assertTrue(all("Action" in g for g in recs["genres"]))

    def test_min_ratings_filter(self):
        # Filter with min_ratings = 20
        recs = recommend_popular(self.df, min_ratings=20, top_n=10)
        self.assertTrue(all(c >= 20 for c in recs["rating_count"]))
        self.assertEqual(len(recs), 3)

    def test_sorting_and_tie_breaking(self):
        # Movie B (4.5, 25) vs Movie A (4.5, 10): Movie B should be first
        recs = recommend_popular(self.df, genre="Action", min_ratings=5, top_n=5)
        self.assertEqual(recs.iloc[0]["title"], "Movie B")
        self.assertEqual(recs.iloc[1]["title"], "Movie A")
        self.assertEqual(recs.iloc[2]["title"], "Movie E")

    def test_top_n_truncation(self):
        recs = recommend_popular(self.df, top_n=2)
        self.assertEqual(len(recs), 2)

    def test_no_matches(self):
        recs = recommend_popular(self.df, genre="Horror", min_ratings=0, top_n=5)
        self.assertEqual(len(recs), 0)


if __name__ == "__main__":
    unittest.main()
