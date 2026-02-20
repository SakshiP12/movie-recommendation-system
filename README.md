# 🎬 Movie Recommendation System

An end-to-end Movie Recommendation System built using **KNN, TF-IDF, Streamlit, PostgreSQL, Docker, and AWS EC2**.

This project provides personalized movie recommendations based on content similarity and includes full authentication with user activity tracking.

---

## 🚀 Features

- 🔍 Content-based Movie Recommendation using KNN
- 🧠 TF-IDF Vectorization for feature extraction
- 🔐 User Authentication (Login / Signup)
- 🗂 User Search History Tracking
- 🐳 Dockerized Application
- ☁️ Deployed on AWS EC2
- 🗄 PostgreSQL Database Integration

---

## 🏗 Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- PostgreSQL
- Docker
- AWS EC2

---

## 📂 Project Structure

```
movie-recommendation-system-main/
│
├── app.py
├── build_knn_model.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
│
├── movie_list_knn.pkl
├── movie_features.pkl
├── knn_model.pkl
├── tfidf_vectorizer.pkl
└── movie_list.pkl
```

---

## ⚙️ How It Works

1. Movie metadata is processed.
2. Important text features are combined into a single "soup".
3. TF-IDF converts text into numerical vectors.
4. KNN finds similar movies using cosine similarity.
5. Recommendations are displayed via Streamlit UI.
6. User login and activity are stored in PostgreSQL.

---

## 🧠 Model Building

To rebuild the model locally:

```bash
python build_knn_model.py
```

This generates:

- movie_list_knn.pkl
- movie_features.pkl
- knn_model.pkl
- tfidf_vectorizer.pkl

---

## 🖥 Run Locally (Without Docker)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

---

## 🐳 Run Using Docker

### 1️⃣ Build Docker Image

```bash
docker build -t movie-app .
```

### 2️⃣ Run PostgreSQL Container

```bash
docker run -d \
  --name movie-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=**** \
  -e POSTGRES_DB=movie_app \
  -p 5432:5432 \
  postgres
```

### 3️⃣ Run App Container

```bash
docker run -d \
  --name movie-app-container \
  --link movie-db \
  -p 8501:8501 \
  movie-app
```

---

## 🗄 Database Schema

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password TEXT NOT NULL
);

CREATE TABLE user_activity (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100),
    movie_title VARCHAR(255),
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## ☁️ AWS EC2 Deployment Steps

1. Launch Ubuntu EC2 instance
2. Install Docker
3. Clone repository
4. Build Docker image
5. Run PostgreSQL container
6. Run Streamlit container
7. Open port 8501 in Security Group

---

## 📊 Recommendation Logic

The system uses:

- Content-based filtering
- Cosine similarity
- K-Nearest Neighbors algorithm

Movies are recommended based on similarity in:

- Genres
- Cast
- Keywords
- Overview
- Director

---

