from sklearn.neighbors import NearestNeighbors

def train_knn(matrix, n_neighbors: int = 30):
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute"
    )
    model.fit(matrix.values)
    return model