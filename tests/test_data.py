from src.data.load_data import load_ratings, load_movies

def test_load_ratings():
    ratings = load_ratings()
    assert not ratings.empty
    assert "userId" in ratings.columns
    assert "movieId" in ratings.columns
    assert "rating" in ratings.columns

def test_load_movies():
    movies = load_movies()
    assert not movies.empty
    assert "movieId" in movies.columns
    assert "title" in movies.columns
