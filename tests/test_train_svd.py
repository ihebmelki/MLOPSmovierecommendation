import numpy as np
import pandas as pd
from src.models.train_svd import train_svd

def test_svd_training():
    matrix = pd.DataFrame(
        np.random.rand(5, 4),
        index=[1,2,3,4,5],
        columns=[10,11,12,13]
    )

    model = train_svd(matrix, n_components=2)

    assert len(model["user_factors"]) == 5
    assert len(model["movie_factors"]) == 4
    assert model["n_components"] == 2
