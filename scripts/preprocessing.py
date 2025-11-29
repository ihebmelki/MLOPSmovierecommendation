# =========================================
# preprocessing.py
# =========================================
"""
Preprocessing and feature engineering for MovieLens dataset.
This script:
- Handles missing values
- Converts column types
- Computes movie and user statistics
- One-hot encodes genres
- Aggregates genome tag scores
- Aggregates user tags
- Saves processed datasets to data/processed/ (overwriting if they exist)
- Adds processed datasets to DVC automatically
"""

import os
import subprocess
import pandas as pd
import numpy as np

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.getcwd()  # assumes script is run from project root
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')  # save processed data here

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# -----------------------------
# Load raw data
# -----------------------------
movies = pd.read_csv(os.path.join(RAW_DATA_PATH, 'movies.csv'))
ratings = pd.read_csv(os.path.join(RAW_DATA_PATH, 'ratings.csv'))
tags = pd.read_csv(os.path.join(RAW_DATA_PATH, 'tags.csv'))
links = pd.read_csv(os.path.join(RAW_DATA_PATH, 'links.csv'))

# -----------------------------
# Handle missing values
# -----------------------------
ratings = ratings.dropna(subset=['userId','movieId','rating'])
tags = tags.dropna(subset=['userId','movieId','tag'])
movies = movies.dropna(subset=['movieId','title'])
links = links.dropna(subset=['movieId'])  # Keep links as-is
links['imdbId'] = links['imdbId'].fillna('unknown')
links['tmdbId'] = links['tmdbId'].fillna(-1)

# -----------------------------
# Convert data types
# -----------------------------
movies['movieId'] = movies['movieId'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)
ratings['userId'] = ratings['userId'].astype(int)
tags['movieId'] = tags['movieId'].astype(int)
tags['userId'] = tags['userId'].astype(int)
links['movieId'] = links['movieId'].astype(int)

ratings['rating'] = ratings['rating'].astype(float)

# -----------------------------
# Timestamp → datetime
# -----------------------------
if np.issubdtype(ratings['timestamp'].dtype, np.number):
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
else:
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], errors='coerce')

if np.issubdtype(tags['timestamp'].dtype, np.number):
    tags['timestamp'] = pd.to_datetime(tags['timestamp'], unit='s')
else:
    tags['timestamp'] = pd.to_datetime(tags['timestamp'], errors='coerce')

# -----------------------------
# Text → string
# -----------------------------
movies['title'] = movies['title'].astype(str)
movies['genres'] = movies['genres'].astype(str)
tags['tag'] = tags['tag'].astype(str)

# -----------------------------
# Feature Engineering
# -----------------------------
# Movie statistics
movie_stats = ratings.groupby('movieId').agg(
    movie_avg_rating=('rating', 'mean'),
    movie_rating_count=('rating', 'count')
).reset_index()
movies = movies.merge(movie_stats, on='movieId', how='left')

# User statistics
user_stats = ratings.groupby('userId').agg(
    user_avg_rating=('rating', 'mean'),
    user_rating_count=('rating', 'count')
).reset_index()

# One-hot encode genres
movies_exploded = movies.copy()
movies_exploded['genres'] = movies_exploded['genres'].str.split('|')
movies_exploded = movies_exploded.explode('genres')
genre_dummies = pd.get_dummies(movies_exploded['genres'], prefix='genre')
movies_genres = pd.concat([movies_exploded[['movieId']], genre_dummies], axis=1)
movies_genres = movies_genres.groupby('movieId').max().reset_index()
movies = movies.merge(movies_genres, on='movieId', how='left')

# Aggregate user tags
tags_agg = tags.groupby(['userId','movieId'])['tag'].apply(lambda x: ' '.join(x)).reset_index()
ratings_with_tags = ratings.merge(tags_agg, on=['userId','movieId'], how='left')

# -----------------------------
# Save processed datasets (overwrite)
# -----------------------------
output_files = {
    'movies_processed.csv': movies,
    'ratings_processed.csv': ratings,
    'user_stats.csv': user_stats,
    'ratings_with_tags.csv': ratings_with_tags,
    'links_processed.csv': links
}

for filename, df in output_files.items():
    file_path = os.path.join(PROCESSED_DATA_PATH, filename)
    df.to_csv(file_path, index=False)
    print(f"✅ Saved {filename} (overwritten if existed)")

# -----------------------------
# Add processed files to DVC
# -----------------------------
for filename in output_files.keys():
    file_path = os.path.join(PROCESSED_DATA_PATH, filename)
    subprocess.run(['dvc', 'add', file_path])
    print(f"📦 Added {filename} to DVC")

print("\n🎉 Preprocessing, feature engineering, and DVC tracking completed successfully!")
