import pandas as pd

def create_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Crée la matrice utilisateur-film (sparse → remplie avec 0)"""
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0
    )