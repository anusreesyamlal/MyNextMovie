"""
Unit tests for content_recommender module.
"""

import unittest
import pandas as pd
from src.content_recommender import recommend_similar, _resolve_movie_index


class TestContentRecommender(unittest.TestCase):
    def setUp(self):
        self.movies_df = pd.DataFrame([
            {"movieId": 1, "title": "Toy Story (1995)", "genres": ["Adventure", "Animation", "Children", "Comedy", "Fantasy"]},
            {"movieId": 2, "title": "Jumanji (1995)", "genres": ["Adventure", "Children", "Fantasy"]},
            {"movieId": 3, "title": "Heat (1995)", "genres": ["Action", "Crime", "Thriller"]},
            {"movieId": 4, "title": "GoldenEye (1995)", "genres": ["Action", "Adventure", "Thriller"]},
            {"movieId": 5, "title": "Sudden Death (1995)", "genres": ["Action"]},
        ])

    def test_exclude_query_movie(self):
        recs = recommend_similar(self.movies_df, movie_title_or_id=1, top_n=3)
        # Target movie 1 should not be in recommendations
        self.assertNotIn(1, recs["movieId"].values)
        self.assertEqual(len(recs), 3)

    def test_query_by_title_case_insensitive(self):
        recs = recommend_similar(self.movies_df, movie_title_or_id="toy story (1995)", top_n=2)
        self.assertEqual(recs.iloc[0]["title"], "Jumanji (1995)")
        self.assertIn("similarity_score", recs.columns)
        self.assertGreater(recs.iloc[0]["similarity_score"], 0)

    def test_similarity_ordering(self):
        # Heat (Action, Crime, Thriller) should be most similar to GoldenEye (Action, Adventure, Thriller)
        recs = recommend_similar(self.movies_df, movie_title_or_id="Heat (1995)", top_n=4)
        self.assertEqual(recs.iloc[0]["title"], "GoldenEye (1995)")
        self.assertEqual(recs.iloc[1]["title"], "Sudden Death (1995)")
        # Non-action movies should have 0 similarity
        self.assertEqual(recs.iloc[-1]["similarity_score"], 0.0)

    def test_invalid_movie_raises(self):
        with self.assertRaises(ValueError):
            recommend_similar(self.movies_df, movie_title_or_id="Nonexistent Movie")


if __name__ == "__main__":
    unittest.main()
