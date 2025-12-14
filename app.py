import pickle
import streamlit as st
import requests
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ---------- TMDB API KEY ----------
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# ---------- Page config ----------
st.set_page_config(page_title="Movie Recommender System", layout="wide")

# ---------- Load data + model ----------
movies = pickle.load(open("movie_list_knn.pkl", "rb"))
feature_matrix = pickle.load(open("movie_features.pkl", "rb"))
knn: NearestNeighbors = pickle.load(open("knn_model.pkl", "rb"))

movies = pd.DataFrame(movies).reset_index(drop=True)
# Columns: movie_id | title | description

# ---------- Session state ----------
if "recs" not in st.session_state:
    st.session_state.recs = []
if "base_movie" not in st.session_state:
    st.session_state.base_movie = None

# ---------- Helper functions ----------
def extract_keywords(text):
    if pd.isna(text):
        return []
    words = text.lower().split()
    stopwords = {
        "the", "and", "of", "in", "to", "a", "is", "with", "for",
        "on", "an", "as", "by", "from", "at", "this", "that",
        "becomes", "become", "becoming", "made", "make", "makes",
        "get", "gets", "got", "one", "two", "three"
    }
    return list(set([w.strip(".,") for w in words if w not in stopwords]))[:15]


def similarity_label(similarity):
    if similarity >= 20:
        return "High"
    elif similarity >= 12:
        return "Medium"
    else:
        return "Low"


@st.cache_data
def fetch_movie_details(movie_id):
    base_poster = "https://via.placeholder.com/500x750?text=No+Image"
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url).json()
    except:
        return base_poster, 0.0, "N/A"

    poster = (
        "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
        if data.get("poster_path")
        else base_poster
    )

    rating = data.get("vote_average", 0.0)
    year = data.get("release_date", "N/A")[:4]

    return poster, rating, year


# ---------- Recommendation logic ----------
def recommend(movie_title):
    match = movies[movies["title"].str.lower() == movie_title.lower()]
    if match.empty:
        return []

    idx = match.index[0]

    base_movie = movies.iloc[idx]
    base_keywords = set(extract_keywords(base_movie.description))

    distances, indices = knn.kneighbors(
        feature_matrix[idx].reshape(1, -1),
        n_neighbors=25
    )

    recommendations = []

    for dist, i in zip(distances[0], indices[0]):
        if i == idx:
            continue

        row = movies.iloc[i]
        rec_keywords = set(extract_keywords(row.description))
        common_keywords = list(base_keywords & rec_keywords)[:3]

        similarity = round((1 - dist) * 100, 2)
        poster, rating, year = fetch_movie_details(row.movie_id)

        recommendations.append(
            {
                "title": row.title,
                "poster": poster,
                "rating": rating,
                "year": year,
                "similarity_label": similarity_label(similarity),
                "why": common_keywords,
            }
        )

        if len(recommendations) == 10:
            break

    return recommendations


# ---------- CSS ----------
st.markdown(
    """
    <style>
    body { background-color: #000; }
    .movie-card {
        background-color: #111;
        padding: 12px;
        border-radius: 12px;
        text-align: center;
        height: 420px;
    }
    .poster {
        height: 230px;
        object-fit: cover;
        border-radius: 10px;
    }
    .title {
        color: white;
        font-size: 14px;
        font-weight: bold;
        margin-top: 6px;
    }
    .meta {
        color: #bbb;
        font-size: 12px;
    }
    .why {
        color: #aaa;
        font-size: 11px;
        margin-top: 6px;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- UI ----------
st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values,
)

if st.button("Get Recommendations"):
    st.session_state.recs = recommend(selected_movie)
    st.session_state.base_movie = selected_movie

# ---------- Display ----------
if st.session_state.recs:
    st.subheader(f"Top 10 Recommendations for: {st.session_state.base_movie}")

    cols = st.columns(5)
    for i, movie in enumerate(st.session_state.recs):
        with cols[i % 5]:
            why_text = (
                "🧠 Similar themes: " + ", ".join(movie["why"]).title()
                if movie["why"]
                else "🧠 Similar content"
            )

            st.markdown(
                f"""
                <div class="movie-card">
                    <img src="{movie['poster']}" class="poster">
                    <div class="title">{movie['title']}</div>
                    <div class="meta">
                        ⭐ {movie['rating']} | 📅 {movie['year']} <br>
                        🔗 Similarity: {movie['similarity_label']}
                    </div>
                    <div class="why">{why_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
