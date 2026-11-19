"""
Popularity-based recommendation module for MyNextMovie.
Recommends top-rated movies filtered by genre and minimum ratings count.
"""

from typing import List, Optional, Union
import pandas as pd


def _matches_genre(genres_val: Union[List[str], str, None], target_genre: str) -> bool:
    """Check if the genres container includes the target genre (case-insensitive)."""
    if genres_val is None:
        return False
    target = target_genre.strip().lower()
    if isinstance(genres_val, (list, tuple, set)):
        return any(target == str(g).strip().lower() or target in str(g).strip().lower() for g in genres_val)
    if isinstance(genres_val, str):
        return target in genres_val.lower()
    return False


def recommend_popular(
    df: pd.DataFrame,
    genre: Optional[str] = None,
    min_ratings: int = 0,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Recommend popular movies filtered by genre and minimum rating threshold.
    
    Parameters:
        df: DataFrame containing movie data with ['title', 'genres', 'mean_rating', 'rating_count'].
        genre: Optional genre name to filter by (case-insensitive).
        min_ratings: Minimum number of ratings required (rating_count >= min_ratings).
        top_n: Number of recommendations to return.
        
    Returns:
        pd.DataFrame: Top-N movies sorted by mean_rating DESC, then rating_count DESC.
    """
    if df.empty:
        return df.copy()
    
    # Filter by minimum ratings threshold
    filtered = df[df["rating_count"] >= min_ratings].copy()
    
    # Filter by genre if specified
    if genre and str(genre).strip():
        mask = filtered["genres"].apply(lambda g: _matches_genre(g, genre))
        filtered = filtered[mask]
        
    # Sort by mean_rating descending, tie-breaker rating_count descending
    recommended = filtered.sort_values(
        by=["mean_rating", "rating_count"],
        ascending=[False, False],
    )
    
    return recommended.head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure project root is in sys.path when running as a script
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.data_loader import load_and_merge
    except ImportError:
        from data_loader import load_and_merge

    # Load data
    movies_df = load_and_merge()
    print("=== All Movies Sample ===")
    print(movies_df[["movieId", "title", "genres", "mean_rating", "rating_count"]].head())

    # Test recommendation for Action movies with at least 1 rating
    target_genre = "Action"
    min_rat = 1
    k = 5
    print(f"\n=== Top {k} '{target_genre}' Movies (min_ratings={min_rat}) ===")
    popular_action = recommend_popular(
        movies_df,
        genre=target_genre,
        min_ratings=min_rat,
        top_n=k,
    )
    print(popular_action[["movieId", "title", "genres", "mean_rating", "rating_count"]])

    # Test recommendation without genre filter (overall top)
    print(f"\n=== Top 3 Movies Overall (min_ratings=2) ===")
    top_overall = recommend_popular(movies_df, min_ratings=2, top_n=3)
    print(top_overall[["movieId", "title", "mean_rating", "rating_count"]])
