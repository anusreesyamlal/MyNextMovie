"""
Data loader module for MyNextMovie.
Handles reading movies and ratings datasets, data cleaning,
genre parsing, and computing per-movie rating statistics.
"""

from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd

# Default project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOVIES_PATH = PROJECT_ROOT / "data" / "Movies.csv"
DEFAULT_RATINGS_PATH = PROJECT_ROOT / "data" / "Ratings.csv"


def _resolve_path(path: Union[str, Path], default: Path) -> Path:
    """Resolve file path, falling back to project root if not found in cwd."""
    p = Path(path)
    if p.exists():
        return p
    resolved = PROJECT_ROOT / p
    return resolved if resolved.exists() else p


def parse_genres(genres_str: Union[str, float, None]) -> List[str]:
    """
    Parse a pipe-separated genre string into a clean list of genres.
    
    Handles missing values and placeholders like '(no genres listed)'.
    """
    if pd.isna(genres_str) or not str(genres_str).strip():
        return []
    
    tokens = [g.strip() for g in str(genres_str).split("|") if g.strip()]
    return [g for g in tokens if g.lower() != "(no genres listed)"]


def load_movies(path: Union[str, Path] = DEFAULT_MOVIES_PATH) -> pd.DataFrame:
    """
    Load and clean the movies dataset.
    
    - Drops records with missing movieId or title.
    - Parses pipe-separated genres into lists.
    """
    file_path = _resolve_path(path, DEFAULT_MOVIES_PATH)
    df = pd.read_csv(file_path)
    
    # Clean missing identifiers and titles
    df = df.dropna(subset=["movieId", "title"])
    df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce")
    df = df.dropna(subset=["movieId"])
    df["movieId"] = df["movieId"].astype(int)
    
    # Parse genres
    df["genres"] = df["genres"].apply(parse_genres)
    return df.reset_index(drop=True)


def load_ratings(path: Union[str, Path] = DEFAULT_RATINGS_PATH) -> pd.DataFrame:
    """
    Load and clean the ratings dataset.
    
    - Drops records with missing userId, movieId, or rating.
    - Coerces rating to numeric and ensures valid identifiers.
    """
    file_path = _resolve_path(path, DEFAULT_RATINGS_PATH)
    df = pd.read_csv(file_path)
    
    # Clean missing ratings and identifiers
    df = df.dropna(subset=["userId", "movieId", "rating"])
    df["userId"] = pd.to_numeric(df["userId"], errors="coerce")
    df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["userId", "movieId", "rating"])
    
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    return df.reset_index(drop=True)


def compute_movie_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-movie mean_rating and rating_count from ratings data.
    """
    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(mean_rating="mean", rating_count="count")
        .reset_index()
    )
    stats["mean_rating"] = stats["mean_rating"].round(2)
    stats["rating_count"] = stats["rating_count"].astype(int)
    return stats


def load_and_merge(
    movies_path: Union[str, Path] = DEFAULT_MOVIES_PATH,
    ratings_path: Union[str, Path] = DEFAULT_RATINGS_PATH,
) -> pd.DataFrame:
    """
    Load movies and ratings, clean missing values, parse genres,
    and compute per-movie mean_rating and rating_count.
    
    Returns:
        pd.DataFrame with columns:
            [movieId, title, genres, mean_rating, rating_count]
    """
    movies = load_movies(movies_path)
    ratings = load_ratings(ratings_path)
    stats = compute_movie_stats(ratings)
    
    merged = movies.merge(stats, on="movieId", how="left")
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
    
    return merged


if __name__ == "__main__":
    df_merged = load_and_merge()
    print("Loaded and merged dataset preview:")
    print(df_merged.head())
    print(f"\nTotal movies: {len(df_merged)}")
