import mlflow
import mlflow.sklearn
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load feature matrix
feature_matrix = pickle.load(open("movie_features.pkl", "rb"))

# Train KNN model
knn = NearestNeighbors(n_neighbors=10, metric="cosine")
knn.fit(feature_matrix)

# Dummy metric for tracking
distances, _ = knn.kneighbors(feature_matrix[:10])
avg_similarity = np.mean(1 - distances)

# MLflow logging
with mlflow.start_run():
    mlflow.log_param("model_type", "KNN")
    mlflow.log_param("n_neighbors", 10)
    mlflow.log_param("metric", "cosine")
    mlflow.log_metric("avg_similarity", avg_similarity)

    mlflow.sklearn.log_model(knn, "knn_model")
    mlflow.log_artifact("movie_list_knn.pkl")
    mlflow.log_artifact("movie_features.pkl")
