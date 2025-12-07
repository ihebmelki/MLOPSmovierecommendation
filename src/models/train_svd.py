import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

def train_svd(matrix, n_components=50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(matrix.values)
    movie_factors = svd.components_.T

    model_dict = {
        "user_ids": matrix.index.tolist(),
        "movie_ids": matrix.columns.tolist(),
        "user_factors": user_factors,
        "movie_factors": movie_factors,
        "n_components": n_components
    }

    return model_dict
