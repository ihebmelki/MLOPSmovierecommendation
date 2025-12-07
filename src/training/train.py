import mlflow
import mlflow.sklearn
import pickle
import os
import numpy as np
from sklearn.metrics import mean_squared_error

from src.data.load_data import load_ratings
from src.features.build_features import create_user_item_matrix
from src.models.train_svd import train_svd


def evaluate_svd(model_dict, matrix):
    """
    Compute RMSE on reconstruction of the user-item matrix.
    """
    user_factors = model_dict["user_factors"]
    movie_factors = model_dict["movie_factors"]

    reconstructed = user_factors @ movie_factors.T
    true = matrix.values

    # Only compare non-zero entries (movies rated)
    mask = true > 0
    rmse = np.sqrt(mean_squared_error(true[mask], reconstructed[mask]))
    return rmse


def main():
    mlflow.set_experiment("movie-recommendation-svd")

    print("Loading data...")
    ratings = load_ratings()
    matrix = create_user_item_matrix(ratings)

    # -------------------------------
    # 1️⃣ Hyperparameters to test
    # -------------------------------
    param_grid = [20, 50, 80, 120]  # try multiple embedding sizes

    best_rmse = float("inf")
    best_model = None
    best_params = None

    print("Starting hyperparameter search...")

    for n_comp in param_grid:
        with mlflow.start_run(run_name=f"SVD_{n_comp}"):

            print(f"Training SVD with n_components={n_comp}...")
            model_dict = train_svd(matrix, n_components=n_comp)

            rmse = evaluate_svd(model_dict, matrix)

            # Log params & metrics
            mlflow.log_param("n_components", n_comp)
            mlflow.log_metric("rmse", rmse)

            print(f"RMSE for n_components={n_comp}: {rmse}")

            # Select best model
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_dict
                best_params = n_comp

    # -------------------------------
    # 2️⃣ Save best model
    # -------------------------------
    print(f"Best model: n_components={best_params} with RMSE={best_rmse}")

    os.makedirs("models", exist_ok=True)
    model_path = "models/svd_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"Best model saved to {model_path}")


if __name__ == "__main__":
    main()
