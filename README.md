# 🎬 Movie Recommendation System

An end-to-end content-based Movie Recommendation System built using Machine Learning and deployed using Docker and AWS EC2. The application provides personalized movie recommendations based on similarity analysis and includes secure user authentication.

---

## 📌 Features

- Content-based movie recommendations using TF-IDF
- K-Nearest Neighbors (KNN) similarity model
- Cosine similarity distance metric
- User authentication system (Signup / Login)
- PostgreSQL database integration
- Dockerized deployment
- AWS EC2 hosting
- Interactive UI built using Streamlit
- User search history tracking

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- PostgreSQL
- Docker
- AWS EC2

---

## 🧠 Machine Learning Approach

1. Data preprocessing and cleaning of movie metadata.
2. Creation of a weighted "soup" of movie features (genres, keywords, cast, director, overview).
3. TF-IDF vectorization to convert text data into numerical format.
4. KNN model trained using cosine similarity.
5. Top similar movies returned based on distance ranking.

---

## ⚙️ Project Workflow

1. Load TMDB dataset.
2. Preprocess and engineer features.
3. Train KNN similarity model.
4. Save trained model using Pickle.
5. Build Streamlit web interface.
6. Integrate PostgreSQL for authentication.
7. Containerize using Docker.
8. Deploy on AWS EC2.

---

## 🐳 Run Locally Using Docker

Build Docker image:

```bash
docker build -t movie-app .
```

Run container:

```bash
docker run -d -p 8501:8501 movie-app
```

Open in browser:

```
http://localhost:8501
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

## ☁ Deployment Architecture

User → AWS EC2 → Docker Container (Streamlit App)  
            ↓  
      Docker Network  
            ↓  
     PostgreSQL Container  

---

## 📂 Project Structure

```
movie-recommendation-system/
│
├── app.py
├── build_knn_model.py
├── Dockerfile
├── requirements.txt
├── README.md
├── movie_list_knn.pkl
├── movie_features.pkl
├── knn_model.pkl
└── tmdb_5000_movies.csv
```

---

## 🎯 Key Highlights

- End-to-end ML deployment project
- Full-stack integration (ML + Backend + Deployment)
- Real-world Docker networking
- Secure password hashing using bcrypt
- Production-ready architecture

---

