import pandas as pd
from pathlib import Path

def load_ratings():
    path = Path("data/processed/ratings_processed.csv")
    return pd.read_csv(path)

def load_movies():
    path = Path("data/processed/movies_processed.csv")
    return pd.read_csv(path)