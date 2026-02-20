# 🎬 Movie Recommendation System

An advanced content-based movie recommendation system built using Machine Learning and deployed with Streamlit.  
The system recommends similar movies based on genres, keywords, cast, director, and movie overview using TF-IDF vectorization and K-Nearest Neighbors (KNN) with cosine similarity.

---

## 🚀 Features

- Content-based recommendation engine  
- Weighted feature engineering (Genres, Director, Cast, Keywords)  
- TF-IDF text vectorization  
- Cosine similarity-based KNN model  
- Interactive web application using Streamlit  
- Docker-ready deployment  
- Cloud-ready (AWS EC2 compatible)

---

## 🛠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Docker  

---

## 📂 Project Structure

movie-recommendation-system/
│
├── app.py
├── build_knn_model.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore

---

## ⚙️ How the System Works

### 1️⃣ Data Preprocessing
- Load TMDB dataset (tmdb_5000_movies.csv, tmdb_5000_credits.csv)
- Parse JSON-like columns (genres, keywords, cast, crew)
- Extract:
  - Genres
  - Top 3 cast members
  - Director
  - Keywords
  - Overview

---

### 2️⃣ Feature Engineering (Weighted Text “Soup”)

To improve recommendation quality, features are weighted:

- Genres → 3x  
- Director → 3x  
- Keywords → 2x  
- Cast → 2x  
- Overview → 1x  

This improves similarity accuracy compared to a simple text-based system.

---

### 3️⃣ TF-IDF Vectorization

- Converts text into numerical feature vectors  
- Removes stop words  
- Uses up to 20,000 features for richer representation  

---

### 4️⃣ KNN Model (Cosine Similarity)

- Uses NearestNeighbors from scikit-learn  
- Metric: Cosine similarity  
- Recommends top similar movies  

---

## ▶️ How to Run Locally

### Step 1: Clone Repository

git clone https://github.com/SakshiP12/movie-recommendation-system.git  
cd movie-recommendation-system  

### Step 2: Create Virtual Environment

python -m venv venv  
venv\Scripts\activate   (Windows)

### Step 3: Install Dependencies

pip install -r requirements.txt  

### Step 4: Train Model

Make sure dataset files are present in the folder:
- tmdb_5000_movies.csv  
- tmdb_5000_credits.csv  

Then run:

python build_knn_model.py  

This generates:
- knn_model.pkl  
- movie_features.pkl  
- movie_list_knn.pkl  
- tfidf_vectorizer.pkl  

### Step 5: Run Streamlit App

streamlit run app.py  

Open in browser:
http://localhost:8501  

---

## 🐳 Run Using Docker

### Build Image

docker build -t movie-app .  

### Run Container

docker run -p 8501:8501 movie-app  

---

## ☁️ AWS Deployment (Optional)

The application can be deployed on:

- AWS EC2  
- Docker container  
- Cloud-based Linux server  

Basic Steps:
1. Launch EC2 instance  
2. Install Docker  
3. Clone repository  
4. Build Docker image  
5. Run container  

---

## 📈 Future Improvements

- Add collaborative filtering  
- Add hybrid recommendation system  
- Add user authentication  
- Improve UI design  
- Add CI/CD pipeline  

---

## 🎯 Interview Highlights

This project demonstrates:

- Feature engineering techniques  
- NLP preprocessing  
- TF-IDF vectorization  
- Similarity-based recommendation  
- Machine Learning model deployment  
- Docker containerization  
- Cloud-ready architecture  

---

## 👩‍💻 Author

Sakshi Patil  
Final Year AI & DS Student  

---

✔ Clean  
✔ Deployable  
✔ Interview Ready  