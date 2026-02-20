import time
import pickle
import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import psycopg2
import bcrypt

# ================= DATABASE CONNECTION =================
def get_connection():
    return psycopg2.connect(
        host="movie-db",
        database="movie_app",
        user="username",
        password="******"
    )

# ================= AUTH FUNCTIONS =================
def create_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, hashed_password)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Signup error: {e}")
        return False


def authenticate(username, password):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT password FROM users WHERE username=%s",
            (username,)
        )
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result:
            return bcrypt.checkpw(password.encode(), result[0].encode())
        return False
    except Exception as e:
        st.error(f"Login error: {e}")
        return False


def log_activity(username, movie_title):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_activity (username, movie_title) VALUES (%s, %s)",
            (username, movie_title)
        )
        conn.commit()
        cur.close()
        conn.close()
    except:
        pass


# ================= CONFIG =================
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    movies = pickle.load(open("movie_list_knn.pkl", "rb"))
    feature_matrix = pickle.load(open("movie_features.pkl", "rb"))
    knn: NearestNeighbors = pickle.load(open("knn_model.pkl", "rb"))
    movies = pd.DataFrame(movies).reset_index(drop=True)
    return movies, feature_matrix, knn

movies, feature_matrix, knn = load_models()

# ================= SESSION STATE =================
if "recs" not in st.session_state:
    st.session_state.recs = []

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

if "login_time" not in st.session_state:
    st.session_state.login_time = None

if "remember_me" not in st.session_state:
    st.session_state.remember_me = False

# ================= AUTO LOGOUT =================
if st.session_state.logged_in and st.session_state.login_time:
    now = time.time()
    timeout = 60 * 60
    if st.session_state.remember_me:
        timeout = 60 * 60 * 24

    if now - st.session_state.login_time > timeout:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.login_time = None
        st.session_state.remember_me = False
        st.warning("Session expired. Please login again.")
        st.rerun()

# ================= LOGIN / SIGNUP =================
if not st.session_state.logged_in:
    st.title("🔐 Authentication")

    auth_mode = st.radio("Choose Action", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    remember = st.checkbox("🔑 Remember me")

    if auth_mode == "Signup":
        if st.button("Create Account"):
            if create_user(username, password):
                st.success("Account created! Please login.")
    else:
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.login_time = time.time()
                st.session_state.remember_me = remember
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    st.stop()

# ================= SIDEBAR =================
st.sidebar.header("📌 Navigation")
page = st.sidebar.radio("Go to", ["🎬 Recommender", "👤 Profile"])

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.login_time = None
    st.session_state.remember_me = False
    st.rerun()

# ================= PROFILE PAGE =================
if page == "👤 Profile":
    st.title("👤 User Profile")

    st.write("**Username:**", st.session_state.username)
    st.write(
        "**Logged in at:**",
        time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(st.session_state.login_time)
        )
    )

    if st.session_state.remember_me:
        st.success("🔑 Remember Me is ON (24 hour session)")
    else:
        st.info("🔒 Normal session (1 hour timeout)")

    st.subheader("🧾 Your Activity History")

    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT movie_title, searched_at
        FROM user_activity
        WHERE username = %s
        ORDER BY searched_at DESC
        """,
        conn,
        params=(st.session_state.username,)
    )
    conn.close()

    if df.empty:
        st.info("No activity yet.")
    else:
        st.dataframe(df, use_container_width=True)

        if st.button("🗑 Clear History"):
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM user_activity WHERE username = %s",
                (st.session_state.username,)
            )
            conn.commit()
            cur.close()
            conn.close()
            st.success("History cleared!")
            st.rerun()

    st.stop()

# ================= FILTERS =================
st.sidebar.header("🎛 Filters")

min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 6.0, 0.5)
min_year = st.sidebar.slider("Release Year After", 1980, 2025, 2000)

sort_option = st.sidebar.selectbox(
    "Sort By",
    ["Similarity (Default)", "Rating (High to Low)", "Rating (Low to High)"]
)

# ================= TMDB HELPERS =================
@st.cache_data
def fetch_movie_details(movie_id):
    base = "https://via.placeholder.com/500x750?text=No+Image"
    try:
        data = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}",
            timeout=5
        ).json()
    except:
        return base, 0.0, "N/A"

    poster = (
        "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
        if data.get("poster_path") else base
    )
    return poster, data.get("vote_average", 0.0), data.get("release_date", "N/A")[:4]

# ================= RECOMMENDER =================
def similarity_label(sim):
    if sim >= 70:
        return "High"
    elif sim >= 40:
        return "Medium"
    return "Low"


def recommend(movie_title):
    match = movies[movies["title"].str.lower() == movie_title.lower()]
    if match.empty:
        return []

    idx = match.index[0]
    distances, indices = knn.kneighbors(
        feature_matrix[idx].reshape(1, -1), n_neighbors=20
    )

    recs = []

    for dist, i in zip(distances[0], indices[0]):
        if i == idx:
            continue

        row = movies.iloc[i]
        poster, rating, year = fetch_movie_details(row.movie_id)

        if rating < min_rating or (year != "N/A" and int(year) < min_year):
            continue

        recs.append({
            "title": row.title,
            "poster": poster,
            "rating": rating,
            "year": year,
            "similarity": similarity_label((1 - dist) * 100)
        })

        if len(recs) == 10:
            break

    if sort_option == "Rating (High to Low)":
        recs.sort(key=lambda x: x["rating"], reverse=True)
    elif sort_option == "Rating (Low to High)":
        recs.sort(key=lambda x: x["rating"])

    return recs

# ================= UI =================
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Select a movie", movies["title"].values)

if st.button("Get Recommendations"):
    log_activity(st.session_state.username, selected_movie)
    st.session_state.recs = recommend(selected_movie)

# ================= DISPLAY =================
if st.session_state.recs:
    html = """
    <style>
    .grid { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; }
    .card {
        background:#111; padding:8px; border-radius:12px;
        text-align:center; color:white; height:420px;
        transition:transform .3s, box-shadow .3s;
    }
    .card:hover {
        transform:scale(1.06);
        box-shadow:0 12px 30px rgba(0,0,0,.7);
    }
    img {
        height:220px; width:100%;
        object-fit:cover; border-radius:10px;
    }
    .meta { font-size:12px; color:#bbb; }
    </style>
    <div class="grid">
    """

    for m in st.session_state.recs:
        html += f"""
        <div class="card">
            <img src="{m['poster']}">
            <b>{m['title']}</b>
            <div class="meta">
                ⭐ {m['rating']} | 📅 {m['year']} <br>
                🔗 Similarity: {m['similarity']}
            </div>
        </div>
        """

    html += "</div>"
    components.html(html, height=900, scrolling=True)
