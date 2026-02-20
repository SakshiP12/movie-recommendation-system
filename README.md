# 🎬 Movie Recommendation System

An end-to-end, explainable content-based movie recommendation system built using machine learning and MLOps tools.

## 🚀 Features
- Content-based movie recommendations using **TF-IDF + KNN**
- Explainable recommendations with **“Why this movie?”** logic
- Interactive web application built using **Streamlit**
- Experiment tracking and model versioning using **MLflow**
- Data and model versioning using **DVC**
- Dockerized for reproducible and portable deployment
- Cloud-ready architecture (AWS compatible)

## 🛠 Tech Stack
- Python
- Streamlit
- Scikit-learn
- MLflow
- DVC
- Docker
- AWS (EC2, S3)

## 📁 Project Workflow
1. Data preprocessing and feature extraction using TF-IDF  
2. Model training using KNN for similarity-based recommendations  
3. Experiment tracking using MLflow  
4. Data and model versioning using DVC with cloud storage  
5. Interactive UI development using Streamlit  
6. Containerization using Docker  
7. Deployment on cloud infrastructure  

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
dvc pull
streamlit run app.py
