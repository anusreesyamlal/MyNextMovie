"""
MyNextMovie CLI and Interactive Application.

Provides a unified interface for both:
1. Popularity-based Recommender (filtered by genre, min_ratings, top_n)
2. Content-based Recommender (filtered by movie similarity, top_n)

Saves execution results and logs to outputs/sample_results.txt.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.data_loader import load_and_merge
from src.popularity_recommender import recommend_popular
from src.content_recommender import recommend_similar

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "sample_results.txt"


def log_and_print(message: str, output_file: Optional[Path] = None) -> None:
    """Print message to stdout and append to the output file."""
    print(message)
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def run_popularity(
    df,
    genre: Optional[str] = None,
    min_ratings: int = 1,
    top_n: int = 5,
    output_file: Optional[Path] = None,
):
    """Execute popularity-based recommendations and record outputs."""
    header = (
        f"\n{'='*70}\n"
        f" POPULARITY-BASED RECOMMENDATIONS\n"
        f" Filter: genre='{genre or 'ALL'}', min_ratings={min_ratings}, top_n={top_n}\n"
        f"{'='*70}"
    )
    log_and_print(header, output_file)

    recs = recommend_popular(df, genre=genre, min_ratings=min_ratings, top_n=top_n)
    if recs.empty:
        log_and_print("No movies matched the given criteria.", output_file)
        return recs

    cols = ["movieId", "title", "genres", "mean_rating", "rating_count"]
    cols_to_show = [c for c in cols if c in recs.columns]
    log_and_print(recs[cols_to_show].to_string(index=False), output_file)
    return recs


def run_content(
    df,
    movie: str,
    top_n: int = 5,
    output_file: Optional[Path] = None,
):
    """Execute content-based recommendations and record outputs."""
    header = (
        f"\n{'='*70}\n"
        f" CONTENT-BASED RECOMMENDATIONS\n"
        f" Target Movie: '{movie}', top_n={top_n}\n"
        f"{'='*70}"
    )
    log_and_print(header, output_file)

    try:
        recs = recommend_similar(df, movie_title_or_id=movie, top_n=top_n)
        cols = ["movieId", "title", "genres", "similarity_score"]
        cols_to_show = [c for c in cols if c in recs.columns]
        log_and_print(recs[cols_to_show].to_string(index=False), output_file)
        return recs
    except ValueError as e:
        error_msg = f"Error: {e}"
        log_and_print(error_msg, output_file)
        return None


def run_sample_demo(df, output_path: Path):
    """Run predefined sample inputs covering both recommenders and log results."""
    # Reset/initialize output file with header
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"MyNextMovie Sample Execution Log\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    log_and_print(f"Saving sample logs to: {output_path}\n", output_path)

    # Sample 1: Popularity (Action genre, min_ratings=1, top_n=5)
    run_popularity(df, genre="Action", min_ratings=1, top_n=5, output_file=output_path)

    # Sample 2: Popularity (All genres, min_ratings=2, top_n=3)
    run_popularity(df, genre=None, min_ratings=2, top_n=3, output_file=output_path)

    # Sample 3: Content-based for "Toy Story (1995)"
    run_content(df, movie="Toy Story (1995)", top_n=4, output_file=output_path)

    # Sample 4: Content-based by movieId (Heat, ID 6)
    run_content(df, movie="6", top_n=4, output_file=output_path)

    log_and_print(f"\nSample execution completed successfully.", output_path)


def interactive_menu(df, output_path: Path):
    """Interactive CLI menu loop."""
    print("\nWelcome to MyNextMovie Interactive Recommender!")
    while True:
        print("\n" + "=" * 40)
        print("1. Popularity-based Recommendations")
        print("2. Content-based Recommendations")
        print("3. Run Sample Demo (Both)")
        print("4. Exit")
        print("=" * 40)
        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            genre_input = input("Enter genre (leave blank for all): ").strip()
            genre = genre_input if genre_input else None
            min_ratings_str = input("Enter min ratings count (default 1): ").strip()
            min_ratings = int(min_ratings_str) if min_ratings_str.isdigit() else 1
            top_n_str = input("Enter number of recommendations (default 5): ").strip()
            top_n = int(top_n_str) if top_n_str.isdigit() else 5
            run_popularity(df, genre=genre, min_ratings=min_ratings, top_n=top_n, output_file=output_path)

        elif choice == "2":
            movie = input("Enter movie title or ID (e.g. 'Toy Story (1995)' or '1'): ").strip()
            if not movie:
                print("Movie title/ID cannot be blank.")
                continue
            top_n_str = input("Enter number of recommendations (default 5): ").strip()
            top_n = int(top_n_str) if top_n_str.isdigit() else 5
            run_content(df, movie=movie, top_n=top_n, output_file=output_path)

        elif choice == "3":
            run_sample_demo(df, output_path)

        elif choice == "4":
            print("Exiting MyNextMovie. Goodbye!")
            break
        else:
            print("Invalid choice, please select 1-4.")


def main():
    parser = argparse.ArgumentParser(
        description="MyNextMovie: Recommendation Engine CLI & Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "popularity", "content", "interactive"],
        default="demo",
        help="Execution mode (demo runs both with sample inputs)",
    )
    parser.add_argument("--genre", type=str, default=None, help="Genre filter for popularity recommender")
    parser.add_argument("--min-ratings", type=int, default=1, help="Min ratings threshold for popularity recommender")
    parser.add_argument("--movie", type=str, default="Toy Story (1995)", help="Target movie title or ID for content recommender")
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save output log",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive prompt mode",
    )

    args = parser.parse_args()

    # Load dataset
    print("Loading movies and ratings datasets...")
    df = load_and_merge()
    print(f"Successfully loaded {len(df)} movies.")

    output_path = args.output.resolve()

    if args.interactive or args.mode == "interactive":
        interactive_menu(df, output_path)
    elif args.mode == "popularity":
        run_popularity(
            df,
            genre=args.genre,
            min_ratings=args.min_ratings,
            top_n=args.top_n,
            output_file=output_path,
        )
    elif args.mode == "content":
        run_content(
            df,
            movie=args.movie,
            top_n=args.top_n,
            output_file=output_path,
        )
    else:
        # Default 'demo' mode
        run_sample_demo(df, output_path)


if __name__ == "__main__":
    main()
