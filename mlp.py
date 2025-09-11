import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Datasets
MOVIES_CSV = "dataset/tmdb_5000_movies.csv"
CREDITS_CSV = "dataset/tmdb_5000_credits.csv"

ps = PorterStemmer()


def _safe_literal(value):
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        return ast.literal_eval(value)
    except Exception:
        return []


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


def _director(crew_str) -> List[str]:
    crew = _safe_literal(crew_str)
    if not isinstance(crew, list):
        return []
    for it in crew:
        if isinstance(it, dict) and it.get("job") == "Director":
            nm = it.get("name")
            return [nm] if isinstance(nm, str) and nm.strip() else []
    return []


def _normalize_token(token: str) -> str:
    token = token.lower().strip()
    token = re.sub(r"[^a-z0-9]+", " ", token)
    token = ps.stem(token)
    return token


def _build_tags(row: pd.Series) -> str:
    # overview
    overview = row.get("overview") or ""
    overview_tokens = [_normalize_token(t) for t in overview.split() if t]

    # genres / keywords
    genres = [_normalize_token(n) for n in _names_from_list_of_dicts(row.get("genres"))]
    keywords = [_normalize_token(n) for n in _names_from_list_of_dicts(row.get("keywords"))]

    # cast (top 3) and director
    cast = [_normalize_token(n) for n in _top_cast(row.get("cast"), 3)]
    director = [_normalize_token(n) for n in _director(row.get("crew"))]

    tokens = overview_tokens + genres + keywords + cast + director
    return " ".join(t for t in tokens if t)


class Recommender:
    def __init__(self, movies_csv: str = MOVIES_CSV, credits_csv: str = CREDITS_CSV) -> None:
        movies_csv_path = Path(movies_csv)
        credits_csv_path = Path(credits_csv)

        if not movies_csv_path.exists() or not credits_csv_path.exists():
            raise FileNotFoundError(
                f"Missing dataset files. Expected at:\n- {movies_csv_path}\n- {credits_csv_path}"
            )

        # Load & merge
        movies_raw = pd.read_csv(movies_csv_path)
        credits_raw = pd.read_csv(credits_csv_path)
        movies = movies_raw.merge(credits_raw, on="title", how="inner")

        # Fill missing text fields to avoid NaNs during tokenization
        for col in ("overview", "genres", "keywords", "cast", "crew"):
            if col in movies.columns:
                movies[col] = movies[col].fillna("")

        # Build tags and vectorize
        movies["tags"] = movies.apply(_build_tags, axis=1)

        vectorizer = CountVectorizer(max_features=5000, stop_words="english")
        vectors = vectorizer.fit_transform(movies["tags"].values.astype("U")).toarray()
        similarity = cosine_similarity(vectors)

        # Store attributes
        self.movies: pd.DataFrame = movies.reset_index(drop=True)
        self.similarity: np.ndarray = similarity
        self.all_titles: List[str] = self.movies["title"].tolist()
        self.title_to_index: Dict[str, int] = {t: i for i, t in enumerate(self.all_titles)}

    def recommend_titles(self, title: str, top_k: int = 5) -> List[str]:
        if not isinstance(title, str) or title not in self.title_to_index:
            return []
        idx = self.title_to_index[title]
        distances = list(enumerate(self.similarity[idx]))
        distances.sort(key=lambda x: x[1], reverse=True)

        picks: List[str] = []
        for i, _ in distances[1 : top_k + 1]:
            picks.append(self.movies.iloc[i]["title"])
        return picks


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    return Recommender()
