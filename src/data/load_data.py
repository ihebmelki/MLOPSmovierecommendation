import pandas as pd
from pathlib import Path

def load_ratings():
    path = Path("data/raw/ratings.csv")
    return pd.read_csv(path)

def load_movies():
    path = Path("data/raw/movies.csv")
    return pd.read_csv(path)