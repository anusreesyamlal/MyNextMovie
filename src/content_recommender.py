"""
Content-based recommendation module for MyNextMovie.
Recommends movies similar to a query movie based on genre cosine similarity.
"""

from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _format_genres(genres_val: Union[List[str], str, None]) -> str:
    """Format genre input into a clean pipe-separated string for TF-IDF tokenization."""
    if isinstance(genres_val, (list, tuple, set)):
        clean_tokens = [str(g).strip().lower() for g in genres_val if str(g).strip()]
        return "|".join(clean_tokens)
    if pd.isna(genres_val) or not genres_val:
        return ""
    # If already a string, normalize to lowercase
    tokens = [g.strip().lower() for g in str(genres_val).split("|") if g.strip()]
    return "|".join(tokens)


def _resolve_movie_index(df: pd.DataFrame, movie_title_or_id: Union[int, str]) -> int:
    """
    Find the DataFrame row index for a given movie title or movieId.
    Supports integer IDs, string IDs, exact titles, and partial title matching.
    """
    # 1. Match by numeric movieId
    if isinstance(movie_title_or_id, (int, np.integer)):
        match = df.index[df["movieId"] == movie_title_or_id]
        if not match.empty:
            return match[0]

    if isinstance(movie_title_or_id, str):
        query_str = movie_title_or_id.strip()
        # 1b. Check if string is a numeric ID
        if query_str.isdigit():
            match = df.index[df["movieId"] == int(query_str)]
            if not match.empty:
                return match[0]

        # 2. Match exact title (case-insensitive)
        match = df.index[df["title"].str.strip().str.lower() == query_str.lower()]
        if not match.empty:
            return match[0]

        # 3. Match partial/contains title (case-insensitive)
        partial = df.index[df["title"].str.lower().str.contains(query_str.lower(), regex=False)]
        if not partial.empty:
            return partial[0]

    raise ValueError(f"Movie '{movie_title_or_id}' not found in dataset.")


def recommend_similar(
    movies_df: pd.DataFrame,
    movie_title_or_id: Union[int, str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Recommend the top_n most similar movies to a given movie using genre TF-IDF and cosine similarity.
    
    Parameters:
        movies_df: DataFrame containing movie data with ['movieId', 'title', 'genres'].
        movie_title_or_id: Target movie title (str) or movieId (int/str).
        top_n: Number of recommendations to return (excluding the query movie).
        
    Returns:
        pd.DataFrame: Top-N similar movies with 'similarity_score' column, sorted descending.
    """
    if movies_df.empty:
        return movies_df.copy()

    df = movies_df.reset_index(drop=True)
    query_idx = _resolve_movie_index(df, movie_title_or_id)

    # Prepare genre strings for TfidfVectorizer
    genre_corpus = df["genres"].apply(_format_genres)

    # Compute TF-IDF matrix using pipe as token delimiter
    vectorizer = TfidfVectorizer(token_pattern=r"[^|]+", lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(genre_corpus)

    # Compute cosine similarity between the query movie and all movies
    query_vec = tfidf_matrix[query_idx]
    cosine_sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Create recommendations DataFrame excluding the query movie itself
    candidate_indices = [i for i in range(len(df)) if i != query_idx]
    recs = df.iloc[candidate_indices].copy()
    recs["similarity_score"] = np.round(cosine_sims[candidate_indices], 4)

    # Sort primarily by similarity score descending (secondary: rating_count if present)
    sort_cols = ["similarity_score"]
    ascending_order = [False]
    if "rating_count" in recs.columns:
        sort_cols.append("rating_count")
        ascending_order.append(False)

    sorted_recs = recs.sort_values(by=sort_cols, ascending=ascending_order)
    return sorted_recs.head(top_n).reset_index(drop=True)


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

    # Load dataset
    df = load_and_merge()

    # Test 1: Query by title string "Toy Story (1995)"
    query_title = "Toy Story (1995)"
    print(f"=== Recommendations for: '{query_title}' ===")
    results_by_title = recommend_similar(df, query_title, top_n=5)
    print(results_by_title[["movieId", "title", "genres", "similarity_score"]])

    # Test 2: Query by integer ID (e.g. movieId=6 -> Heat)
    query_id = 6
    query_movie_title = df[df["movieId"] == query_id]["title"].iloc[0]
    print(f"\n=== Recommendations for: '{query_movie_title}' (ID={query_id}) ===")
    results_by_id = recommend_similar(df, query_id, top_n=5)
    print(results_by_id[["movieId", "title", "genres", "similarity_score"]])
