import mlflow
import mlflow.sklearn
from src.data.load_data import load_ratings, load_movies
from src.features.build_features import create_user_item_matrix
from src.models.train_model import train_knn

def main():
    # Créer ou récupérer l'expérience MLflow
    mlflow.set_experiment("movie-recommendation-knn")

    with mlflow.start_run(run_name="knn-v1"):
        print("Chargement des données...")
        ratings = load_ratings()
        movies = load_movies()

        print("Création de la matrice utilisateur-item...")
        matrix = create_user_item_matrix(ratings)

        print("Entraînement du modèle KNN...")
        model = train_knn(matrix, n_neighbors=30)

        # Métriques utiles
        n_users, n_items = matrix.shape
        sparsity = 1 - (len(ratings) / (n_users * n_items))

        # Log dans MLflow
        mlflow.log_param("n_neighbors", 30)
        mlflow.log_param("n_users", n_users)
        mlflow.log_param("n_items", n_items)
        mlflow.log_metric("sparsity", sparsity)

        # Sauvegarde du modèle
        mlflow.sklearn.log_model(model, "knn_model")

        # Sauvegarde des données/movies pour référence
        mlflow.log_artifact("data/raw/movies.csv", artifact_path="data")

        print("Entraînement terminé !")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()