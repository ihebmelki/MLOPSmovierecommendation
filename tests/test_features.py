import pandas as pd
from src.features.build_features import create_user_item_matrix

def test_user_item_matrix():
    sample = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [10, 11, 10],
        "rating": [4, 5, 3]
    })

    matrix = create_user_item_matrix(sample)

    assert matrix.shape == (2, 2)   # 2 users × 2 movies
    assert matrix.loc[1, 10] == 4
    assert matrix.loc[2, 10] == 3
