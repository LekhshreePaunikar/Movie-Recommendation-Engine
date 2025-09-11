
# Movie Recommender Engine

Built an end-to-end recommendation engine using the TMDB 5000 dataset, applying text preprocessing, **vectorization (Bag-of-Words)**, **Similarity Computation (Cosine)**, and **K-Nearest-Neighbor (k‑NN)** search on metadata (genres, keywords, cast). Demonstrated skills in **Unsupervised Learning**, **Information Retrieval (IR)**, and **Recommendation Systems**.

**Skills:** Python, Scikit-learn, Pandas, NumPy, Recommendation systems, Unsupervised learning, Information Retrieval (IR), Machine Learning Algorithms and Techniques
I built a content‑based movie recommendation engine on the TMDB 5000 dataset, vectorized it with scikit‑learn’s `CountVectorizer`, compared movies using cosine similarity, and wrapped it in a Flask web app.

---

## Purpose

My goal was to learn, build, and deploy a **content‑based** recommendation engine that:  
- Ingests real‑world movie metadata,  
- Cleans and normalizes text,  
- Turns text into vectors (Bag‑of‑Words),  
- Finds the **top‑5** most similar movies for any title a user types,  
- Serves the results through a simple, stylish web UI.

---

## Dataset Used

- **Source:** TMDB 5000 Movies & Credits (commonly available via Kaggle).  
- **Files:** `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`  

The movies file contains per‑title metadata such as `id`, `title`, `overview`, `genres`, `keywords`, `original_language`, etc. The credits file holds `cast` and `crew` details per movie.

### Columns I Focused On

- **Used columns:** `id`, `title`, `overview`, `genres`, `keywords`, `cast` (top 3 only), `crew` (director only).  
- **De‑prioritized / redundant for this project:** things like `budget`, `homepage`, `tagline`, etc. (these don’t help the text‑based similarity signal I needed).

> I merged the two datasets on **`title`** and kept only the columns that matter for content similarity.

---

## Recommendation System Types (My Framing)

- **Content‑based** *(what I implemented)* – compares items using their own content/attributes (early YouTube style).  
- **Collaborative filtering** – uses behavior from similar users (e.g., “people like you also liked…”).  
- **Hybrid** – combines both (modern large‑scale systems like new YouTube).

I chose **content‑based** because I wanted a fully offline, cold‑start‑friendly system that doesn’t need user histories.

---

## Process I Followed

1. **Download data** --> TMDB 5000 (movies + credits).  
2. **Preprocessing**  
   - Merge the two datasets on `title`.  
   - Keep the **important columns**; ignore budget/homepage/tagline.  
   - Check **nulls** (drop if insignificant; I fill text columns to avoid crashes).  
   - Check **duplicates** (dataset typically has none).  
   - Convert stringified lists (e.g., `genres`, `keywords`, `cast`, `crew`) into Python lists/dicts using **`ast.literal_eval`** via helpers.  
   - For **cast**: keep **top 3** names.  
   - For **crew**: keep **director** (`job == "Director"`).  
   - Split `overview` into tokens, then **normalize** everything:
     - lowercase,  
     - remove non‑alphanumerics (also collapses name spaces --> “tomcruise”),  
     - **stem** via **NLTK**’s **`PorterStemmer`** to map inflections (e.g., `love`, `loved`, `loving` --> `love`).  
   - Concatenate `overview + genres + keywords + cast(top3) + director` into a single paragraph called **`tags`**.
3. **Vectorization**  
   - Use **Bag of Words** with scikit‑learn’s **`CountVectorizer`**:  
     - remove **stop words** (e.g., “in”, “at”, “the”),  
     - keep **top 5000** most frequent tokens (feature cap),  
     - output a **NumPy array** of vectors.
4. **Similarity**  
   - Compute **cosine similarity** between movie vectors (works better than Euclidean in high dimensions).  
   - For a given title, pick the **top‑5** most similar movies.  
   - Use **`enumerate`** when sorting so I preserve original indices when ordering by similarity values (not by index).
5. **Web App**  
   - Flask backend (`app.py`) with routes for:
     - `/` – homepage form,  
     - `/api/suggest` – lightweight title suggestions,  
     - `/recommend` – returns the top‑5 recommendations.  
   - Front‑end: `index.html` (Bootstrap), `styles.css` (dark “Netflix‑ish” look), `app.js` (datalist suggestions).
6. **Deploy**  
   - I can run it locally via Flask or containerize it for a lightweight deployment.

---

## My Decision‑Making Trail

- **Why content‑based (not collaborative)?**  
  I don’t have user behavior logs (ratings/clicks). Content‑based works entirely from item metadata and handles the “cold start” problem for users.

- **Why `CountVectorizer` (not TF‑IDF or Word2Vec)?**  
  I wanted a simple, transparent baseline. BoW with `CountVectorizer` is fast, easy to tune (stopwords, vocabulary size), and works well for short, topical text like overviews/genres/keywords. TF‑IDF is a great follow‑up; Word2Vec/embeddings add nuance but require more tuning and compute.

- **Why cosine similarity (not Euclidean)?**  
  In high‑dimensional sparse spaces, vector **direction** matters more than magnitude; cosine captures that, while Euclidean can be misleading.

- **Why stemming + stopwords?**  
  Stemming reduces sparsity by grouping word forms (e.g., `dance/dancing/danced` into `danc`), and removing stopwords prevents frequent but uninformative words from dominating the top‑5000 features.

- **What about digits?**  
  Many raw numbers don’t help similarity (e.g., “12”, “34”); by normalizing tokens and focusing on meaningful words, I avoid noisy features. Years/decades can be useful, but the overview/keywords/genres already carry strong signals.

---

##  Core Implementation (Python)

The heart of my engine lives in **`mlp.py`**. Here’s how it’s structured:

### Top‑Level Helpers

- **`_safe_literal(value)`** – safely parse JSON‑like strings to Python (via `ast.literal_eval`); otherwise return `[]`.  
- **`_names_from_list_of_dicts(s)`** – pull only the `'name'` fields from a list of dicts (genres/keywords).  
- **`_top_cast(s, k=3)`** – return the first `k` cast names (I use `k=3`).  
- **`_director(crew_str)`** – return `[director_name]` if `job == "Director"`.  
- **`_normalize_token(token)`** – lowercase --> strip non‑alphanumerics --> **Porter stem**.  
- **`_build_tags(row)`** – create one normalized paragraph from `overview + genres + keywords + cast(top3) + director`.

### The Main Class: `Recommender`

- **`__init__(movies_csv, credits_csv, verbose=True)`**  
  - Merge movies + credits on `title`.  
  - Fill text nulls; (duplicates are typically none).  
  - Build **`tags`**, then vectorize with `CountVectorizer(max_features=5000, stop_words="english")`.  
  - Compute a **cosine similarity matrix** across all movies.  
  - Cache quick lookups: `all_titles`, `title_to_index`, `similarity`.  
  - Print simple terminal messages and the total setup time if `verbose` is `True`.

- **`recommend_titles(title, top_k=5)`**  
  - Guard unknown titles (return `[]` with a friendly note if `verbose`).  
  - Fetch that movie’s similarity row, **sort by value** (keep indices via `enumerate`), and return the **next `top_k` titles**.  
  - Print the recs and per‑call timing if `verbose`.

- **`get_recommender()`** *(module‑level factory)*  
  - A `@lru_cache(maxsize=1)` wrapper that builds a single cached `Recommender` for the app.

---

## Web App (using Flask)

- **`app.py`**  
  - Build the recommender **once at startup** (warm cache).  
  - `GET /` – render the page with the list of all titles.  
  - `GET /api/suggest?q=...` – **contains‑match** suggestions (fast datalist fill).  
  - `POST /recommend` – resolve an **exact (case‑insensitive)** title and return the top‑5 similar movies.  
  - I added tiny `print()` timings to show request latency and a simple `VERBOSE` switch to silence logs.

- **`templates/index.html`**  
  - Bootstrap layout + Font Awesome icons.  
  - Simple, accessible form with datalist suggestions and a polished recommendations list.

- **`static/app.js`**  
  - Debounced fetch to `/api/suggest` while typing; renders fresh `<option>` entries.  
  - Press **Enter** to submit without any forced auto‑correction.

- **`static/styles.css`**  
  - A minimal, Netflix‑inspired dark theme, brand accent color, soft shadows, and focus rings for accessibility.

> Example UI (screenshot of my running app):  
>
> ![Working UI](sandbox:/mnt/data/16f4166b-c0e1-4ca2-8efc-9fa42d814bc8.png)

---

## How To Run It Locally

```bash
# 1) create venv + install deps
python -m venv .venv
.\.venv\Scripts\Activate.ps1 
pip install -r requirements.txt

# 2) make sure datasets are in folder dataset/
#    ├─ dataset/tmdb_5000_movies.csv
#    └─ dataset/tmdb_5000_credits.csv

# 3) run the server
python app.py or flask run
# open http://0.0.0.0:5000
```

---

## What I’d Improve Next

- Try **TF‑IDF** and compare with BoW (often stronger for short text).  
- Add **synonym/phrase handling** (e.g., light lemmatization or phrase models).  
- Clean noisy digits/year tokens more intentionally.  
- Add **poster images** and movie metadata in the UI.  
- Provide **fuzzy title match** (`difflib.get_close_matches`) with an “Did you mean…?” flow.  
- Containerize + deploy to a minimal cloud instance.