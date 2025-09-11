import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List
import time
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Datasets
# Paths to the TMDB CSV files we use as our source data
MOVIES_CSV = "dataset/tmdb_5000_movies.csv"
CREDITS_CSV = "dataset/tmdb_5000_credits.csv"

# Stemmer to reduce words to their root for better matching
ps = PorterStemmer()

# Parse JSON-like strings safely into Python objects; fall back to [] on any error
def _safe_literal(value):
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        return ast.literal_eval(value)
    except Exception:
        return []

# From a list of dicts (genres/keywords), extract only the 'name' values as strings
def _names_from_list_of_dicts(s) -> List[str]:
    items = _safe_literal(s)
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        name = (it or {}).get("name")
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out

# From cast, keep only the top k names (we only use top 3 as per plan)
def _top_cast(s, k: int = 3) -> List[str]:
    items = _safe_literal(s)
    if not isinstance(items, list):
        return []
    names = []
    for it in items:
        nm = (it or {}).get("name")
        if isinstance(nm, str) and nm.strip():
            names.append(nm.strip())
        if len(names) >= k:
            break
    return names

# From crew, keep only the Director's name (job == 'Director')
def _director(crew_str) -> List[str]:
    crew = _safe_literal(crew_str)
    if not isinstance(crew, list):
        return []
    for it in crew:
        if isinstance(it, dict) and it.get("job") == "Director":
            nm = it.get("name")
            return [nm] if isinstance(nm, str) and nm.strip() else []
    return []

# Normalize a token: lowercase, strip non-alphanumerics (removes spaces in names), then stem
def _normalize_token(token: str) -> str:
    token = token.lower().strip()
    token = re.sub(r"[^a-z0-9]+", " ", token)
    token = ps.stem(token)
    return token

# Build "tags" by merging overview + genres + keywords + cast(top 3) + director into one normalized paragraph
def _build_tags(row: pd.Series) -> str:
    # Split overview into words (acts like your lambda split) and normalize
    overview = row.get("overview") or ""
    overview_tokens = [_normalize_token(t) for t in overview.split() if t]

    # Convert genres/keywords strings into lists and normalize each name
    genres = [_normalize_token(n) for n in _names_from_list_of_dicts(row.get("genres"))]
    keywords = [_normalize_token(n) for n in _names_from_list_of_dicts(row.get("keywords"))]

    # Use only top 3 cast and only the director from crew; remove spaces via normalization
    cast = [_normalize_token(n) for n in _top_cast(row.get("cast"), 3)]
    director = [_normalize_token(n) for n in _director(row.get("crew"))]

    tokens = overview_tokens + genres + keywords + cast + director
    return " ".join(t for t in tokens if t)


class Recommender:
    def __init__(self, movies_csv: str = MOVIES_CSV, credits_csv: str = CREDITS_CSV, verbose: bool = True) -> None:
        # Merge two CSVs (movies/credits), then build vectors for recommendations
        # Remember verbosity and start a stopwatch for total setup time
        self.verbose = verbose
        t0 = time.perf_counter()  # START TIMER
        movies_csv_path = Path(movies_csv)
        credits_csv_path = Path(credits_csv)

        if self.verbose:
            print(">> Initializing recommender...")
            print(f">> Checking dataset files:\n   - {movies_csv_path}\n   - {credits_csv_path}")
        
        # Guard: ensure both input files exist before continuing
        if not movies_csv_path.exists() or not credits_csv_path.exists():
            raise FileNotFoundError(
                f"Missing dataset files. Expected at:\n- {movies_csv_path}\n- {credits_csv_path}"
            )

        # Read raw CSVs into memory (two datasets to be merged)
        if self.verbose:
            print(">> Reading CSVs into memory...")
        movies_raw = pd.read_csv(movies_csv_path)
        credits_raw = pd.read_csv(credits_csv_path)
        if self.verbose:
            print(f">> Loaded movies: {len(movies_raw)} rows, credits: {len(credits_raw)} rows")

        # Merge datasets on 'title'; we focus on used columns (id/title/overview/genres/keywords/cast/crew)
        if self.verbose:
            print(">> Merging datasets on 'title'...")
        movies = movies_raw.merge(credits_raw, on="title", how="inner")
        if self.verbose:
            print(f">> Merged dataset shape: {movies.shape[0]} rows x {movies.shape[1]} columns")

        # Handle nulls: fill text fields (you could drop rows if missingness is tiny)
        if self.verbose:
            print(">> Cleaning text fields and handling missing values...")
        for col in ("overview", "genres", "keywords", "cast", "crew"):
            if col in movies.columns:
                movies[col] = movies[col].fillna("")
        # Duplicates check (dataset typically has none); call movies.drop_duplicates() if needed        

        # Create a tags paragraph per movie (overview + genres + keywords + cast + director)
        if self.verbose:
            print(">> Building 'tags' (tokenizing, normalizing, stemming)...")
        movies["tags"] = movies.apply(_build_tags, axis=1)
        if self.verbose:
            print(f">> Finished building 'tags' for {len(movies)} movies")
 
         # Vectorize with Bag of Words (CountVectorizer), remove stop words, keep top 5000 features
        if self.verbose:
            print(">> Vectorizing tags with CountVectorizer (max_features=5000, stop_words='english')...")
        vectorizer = CountVectorizer(max_features=5000, stop_words="english")
        vectors = vectorizer.fit_transform(movies["tags"].values.astype("U")).toarray()
        if self.verbose:
            print(f">> Vectors ready. Shape: {vectors.shape[0]} rows x {vectors.shape[1]} features")

        # Compute cosine similarity between all movie vectors (better than Euclidean for high-dim BoW)
        if self.verbose:
            print(">> Computing cosine similarity matrix...")
        similarity = cosine_similarity(vectors)
        if self.verbose:
            print(f">> Similarity matrix shape: {similarity.shape[0]} x {similarity.shape[1]}")

        # Cache lookups (titles -> index) so recommend() is fast; setup timer ends here
        self.movies: pd.DataFrame = movies.reset_index(drop=True)
        self.similarity: np.ndarray = similarity
        self.all_titles: List[str] = self.movies["title"].tolist()
        self.title_to_index: Dict[str, int] = {t: i for i, t in enumerate(self.all_titles)}
        if self.verbose:
            print(f">> Index built for {len(self.all_titles)} titles. Recommender is ready.")
            print(f">> Setup completed in {time.perf_counter() - t0:.2f}s")  # END TIMER for SetupBuild

    def recommend_titles(self, title: str, top_k: int = 5) -> List[str]:
        # Recommend top-k similar movies using cosine similarity; time the call
        t0 = time.perf_counter()  # START TIMER FOR RECOMMENDATION
        if not isinstance(title, str) or title not in self.title_to_index:
            # Guard: unknown title -> empty list (plus a friendly note if verbose)
            if self.verbose:
                print(f"!! Title '{title}' not found in index. Returning empty list.")
            return []
        idx = self.title_to_index[title]
        if self.verbose:
            print(f">> Generating recommendations for '{title}' (top_k={top_k})...")
        # Use enumerate to keep indices, then sort by similarity value (not by index)
        distances = list(enumerate(self.similarity[idx]))
        distances.sort(key=lambda x: x[1], reverse=True)

        picks: List[str] = []
        # Take the next top_k after the movie itself at position 0
        for i, _ in distances[1 : top_k + 1]:
            picks.append(self.movies.iloc[i]["title"])
        if self.verbose:
            print(f">> Recommendations ready for '{title}': {picks}")
            print(f">> Recommendation time: {time.perf_counter() - t0:.3f}s")  # END TIMER for Recommendation Time
        return picks


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    # Cached factory: return a single ready-to-use recommender instance
    return Recommender()
