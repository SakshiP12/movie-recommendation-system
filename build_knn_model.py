import ast
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ==============================
#  Helpers to parse JSON-like text
# ==============================

def safe_literal_eval(x):
    """Safely parse the stringified list/dict from the CSV."""
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError, TypeError):
        return []


def extract_names(list_obj, key="name", max_items=None):
    """
    From a list of dicts, extract 'name' fields.
    Example: [{'id': 28, 'name': 'Action'}, ...] -> ['Action', ...]
    """
    if not isinstance(list_obj, list):
        return []

    names = [d.get(key, "") for d in list_obj if isinstance(d, dict)]
    names = [n.replace(" ", "") for n in names if n]  # remove spaces
    if max_items is not None:
        names = names[:max_items]
    return names


def get_director(crew_list):
    """From crew list, return director name (no spaces)."""
    if not isinstance(crew_list, list):
        return ""
    for member in crew_list:
        if isinstance(member, dict) and member.get("job") == "Director":
            name = member.get("name", "")
            return name.replace(" ", "")
    return ""


# ==============================
# 1. Load and merge data
# ==============================

print("Loading CSV files...")
movies_df = pd.read_csv("tmdb_5000_movies.csv")
credits_df = pd.read_csv("tmdb_5000_credits.csv")

# Rename IDs so we can merge cleanly
movies_df = movies_df.rename(columns={"id": "movie_id"})
credits_df = credits_df.rename(columns={"movie_id": "movie_id"})

# Keep only what we need from credits
credits_df = credits_df[["movie_id", "cast", "crew"]]

# Merge on movie_id
movies = movies_df.merge(credits_df, on="movie_id", how="left")

# Some datasets sometimes use original_title; we standardize to 'title'
title_col = "title"
if "title" not in movies.columns and "original_title" in movies.columns:
    movies = movies.rename(columns={"original_title": "title"})
    title_col = "title"

# Keep needed columns
needed_cols = [
    "movie_id",
    title_col,
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew",
]
movies = movies[needed_cols].copy()
movies = movies.rename(columns={title_col: "title"})

# Fill NaNs
movies["overview"] = movies["overview"].fillna("")
movies["genres"] = movies["genres"].fillna("[]")
movies["keywords"] = movies["keywords"].fillna("[]")
movies["cast"] = movies["cast"].fillna("[]")
movies["crew"] = movies["crew"].fillna("[]")

print("Rows in movies:", len(movies))


# ==============================
# 2. Build weighted 'soup'
# ==============================

def create_weighted_soup(row):
    # parse stringified JSON
    genres_list = extract_names(safe_literal_eval(row["genres"]), max_items=None)
    keywords_list = extract_names(safe_literal_eval(row["keywords"]), max_items=None)
    cast_list = extract_names(safe_literal_eval(row["cast"]), max_items=3)  # top 3 actors
    crew_list = safe_literal_eval(row["crew"])

    director_name = get_director(crew_list)
    overview = str(row["overview"])

    # join them into words
    genres_str = " ".join(genres_list)
    keywords_str = " ".join(keywords_list)
    cast_str = " ".join(cast_list)

    # ===== WEIGHTING =====
    # genres * 3, keywords * 2, cast * 2, director * 3, overview * 1
    soup = (
        (genres_str + " ") * 3 +
        (keywords_str + " ") * 2 +
        (cast_str + " ") * 2 +
        (director_name + " ") * 3 +
        overview
    )

    return soup


print("Creating weighted text soup...")
movies["soup"] = movies.apply(create_weighted_soup, axis=1)


# ==============================
# 3. TF-IDF vectorization
# ==============================

print("Vectorizing with TF-IDF...")
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=20000  # can adjust; more features = heavier but richer
)

feature_matrix = tfidf.fit_transform(movies["soup"])
print("Feature matrix shape:", feature_matrix.shape)


# ==============================
# 4. Fit KNN
# ==============================

print("Fitting KNN model...")
knn = NearestNeighbors(
    n_neighbors=6,      # 1 (itself) + 5 recommendations
    metric="cosine",
    algorithm="brute"
)
knn.fit(feature_matrix)

# ==============================
# 5. Save for app.py
# ==============================

# app.py expects: movie_id, title, description
movies_small = movies[["movie_id", "title", "overview"]].copy()
movies_small = movies_small.rename(columns={"overview": "description"})

print("Saving pickles...")
with open("movie_list_knn.pkl", "wb") as f:
    pickle.dump(movies_small, f)

with open("movie_features.pkl", "wb") as f:
    pickle.dump(feature_matrix, f)

with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Done! Saved: movie_list_knn.pkl, movie_features.pkl, knn_model.pkl, tfidf_vectorizer.pkl")
