"""Train both models from the raw TMDB CSVs and save artifacts into ./models.

Run once (or whenever the data changes):  python train.py
"""
import ast
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RANDOM_STATE = 42


def parse(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []


def names(text, limit=None):
    items = [d["name"] for d in parse(text)]
    return items[:limit] if limit else items


def director(text):
    return next((d["name"] for d in parse(text) if d.get("job") == "Director"), "")


def squash(items):
    return [str(i).replace(" ", "").lower() for i in items if i]


def load():
    movies = pd.read_csv(DATA_DIR / "tmdb_5000_movies.csv")
    credits = pd.read_csv(DATA_DIR / "tmdb_5000_credits.csv")
    df = movies.merge(credits.drop(columns="title"), left_on="id", right_on="movie_id").drop(columns="movie_id")
    df["genre_list"] = df["genres"].apply(names)
    df["keyword_list"] = df["keywords"].apply(names)
    df["cast_list"] = df["cast"].apply(lambda x: names(x, 3))
    df["director"] = df["crew"].apply(director)
    df["company_list"] = df["production_companies"].apply(names)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year
    df["month"] = df["release_date"].dt.month
    df["overview"] = df["overview"].fillna("")
    return df


def train_recommender(df):
    tags = (
        df["overview"].str.lower() + " "
        + df["genre_list"].apply(lambda x: " ".join(squash(x) * 2)) + " "
        + df["keyword_list"].apply(lambda x: " ".join(squash(x))) + " "
        + df["cast_list"].apply(lambda x: " ".join(squash(x))) + " "
        + df["director"].apply(lambda x: " ".join(squash([x]) * 2))
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=20000, min_df=2)
    matrix = tfidf.fit_transform(tags)  # L2-normalised rows -> dot product == cosine similarity

    meta = pd.DataFrame({
        "id": df["id"],
        "title": df["title"],
        "year": df["year"].fillna(0).astype(int),
        "genres": df["genre_list"],
        "director": df["director"],
        "cast": df["cast_list"],
        "overview": df["overview"],
        "vote_average": df["vote_average"].round(1),
        "vote_count": df["vote_count"],
    })
    joblib.dump({"matrix": matrix, "meta": meta}, MODEL_DIR / "recommender.joblib", compress=3)
    print(f"Recommender saved: {matrix.shape[0]} movies x {matrix.shape[1]} terms")


def build_features(frame, all_genres):
    """Feature frame used for BOTH training and live prediction (keeps them in sync)."""
    out = pd.DataFrame(index=frame.index)
    out["log_budget"] = np.log1p(frame["budget"])
    out["runtime"] = frame["runtime"]
    out["year"] = frame["year"]
    out["month"] = frame["month"]
    out["is_english"] = frame["is_english"]
    out["n_companies"] = frame["n_companies"]
    out["n_genres"] = frame["genre_list"].apply(len)
    out["is_summer_or_xmas"] = frame["month"].isin([5, 6, 7, 11, 12]).astype(int)
    for g in all_genres:
        out[f"genre_{g}"] = frame["genre_list"].apply(lambda gs, g=g: int(g in gs))
    return out


def train_revenue_model(df):
    all_genres = sorted({g for gs in df["genre_list"] for g in gs})
    reg = df[(df["budget"] > 10_000) & (df["revenue"] > 10_000) & (df["runtime"] > 0) & df["year"].notna()].copy()
    reg["is_english"] = (reg["original_language"] == "en").astype(int)
    reg["n_companies"] = reg["company_list"].apply(len)

    X = build_features(reg, all_genres)
    y = np.log1p(reg["revenue"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    r2 = r2_score(y_te, model.predict(X_te))
    print(f"Revenue model (RandomForest) test R2 on log revenue: {r2:.3f}")

    # Refit on all rows for the production model
    model.fit(X, y)
    joblib.dump({"model": model, "features": list(X.columns), "genres": all_genres, "test_r2": float(r2)},
                MODEL_DIR / "revenue_model.joblib", compress=3)
    print("Revenue model saved")


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    data = load()
    train_recommender(data)
    train_revenue_model(data)
