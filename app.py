"""Flask app serving the recommender and revenue predictor."""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from train import build_features  # shared feature logic = no train/serve skew

MODEL_DIR = Path(__file__).parent / "models"

app = Flask(__name__)

# ---------- load artifacts once at startup ----------
_rec = joblib.load(MODEL_DIR / "recommender.joblib")
MATRIX = _rec["matrix"].tocsr()
META: pd.DataFrame = _rec["meta"].reset_index(drop=True)
TITLE_LOWER = META["title"].str.lower()
TITLE_INDEX = pd.Series(META.index, index=TITLE_LOWER).groupby(level=0).first()  # first match wins on duplicate titles

_rev = joblib.load(MODEL_DIR / "revenue_model.joblib")
REV_MODEL, REV_FEATURES, ALL_GENRES = _rev["model"], _rev["features"], _rev["genres"]
REV_R2 = _rev["test_r2"]


def movie_card(i, similarity=None):
    row = META.iloc[i]
    card = {
        "id": int(row["id"]),
        "title": row["title"],
        "year": int(row["year"]) or None,
        "genres": list(row["genres"]),
        "director": row["director"],
        "cast": list(row["cast"]),
        "overview": row["overview"],
        "rating": float(row["vote_average"]),
        "votes": int(row["vote_count"]),
    }
    if similarity is not None:
        card["similarity"] = round(float(similarity), 3)
    return card


def search_titles(q, limit=8):
    q = q.strip().lower()
    if not q:
        return []
    starts = TITLE_LOWER.str.startswith(q)
    contains = TITLE_LOWER.str.contains(q, regex=False) & ~starts
    idx = list(META.index[starts])[:limit] + list(META.index[contains])[: max(0, limit - starts.sum())]
    return idx[:limit]


# ---------- pages ----------
@app.get("/")
def index():
    return render_template("index.html", genres=ALL_GENRES)


@app.get("/health")
def health():
    return jsonify(status="ok", movies=len(META))


# ---------- API ----------
@app.get("/api/search")
def api_search():
    idx = search_titles(request.args.get("q", ""))
    return jsonify([{"title": META.at[i, "title"], "year": int(META.at[i, "year"]) or None} for i in idx])


@app.get("/api/recommend")
def api_recommend():
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify(error="Query parameter 'title' is required."), 400
    try:
        n = max(1, min(int(request.args.get("n", 10)), 30))
    except ValueError:
        return jsonify(error="'n' must be an integer."), 400
    genre = request.args.get("genre", "").strip()

    if title.lower() not in TITLE_INDEX.index:
        suggestions = [META.at[i, "title"] for i in search_titles(title, 5)]
        return jsonify(error=f"Movie '{title}' not found.", suggestions=suggestions), 404

    idx = int(TITLE_INDEX[title.lower()])
    sims = (MATRIX @ MATRIX[idx].T).toarray().ravel()
    sims[idx] = -1  # exclude the movie itself
    order = np.argsort(-sims)
    if genre:
        order = [i for i in order if genre in META.at[i, "genres"]]
    order = list(order)[:n]

    return jsonify(query=movie_card(idx), recommendations=[movie_card(i, sims[i]) for i in order])


@app.post("/api/predict-revenue")
def api_predict_revenue():
    data = request.get_json(silent=True) or {}
    try:
        budget = float(data["budget"])
        runtime = float(data["runtime"])
        year = int(data["year"])
        month = int(data["month"])
        genres = [g for g in data.get("genres", []) if g in ALL_GENRES]
        n_companies = int(data.get("n_companies", 2))
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Required numeric fields: budget, runtime, year, month. Optional: genres[], language, n_companies."), 400
    if not (10_000 <= budget <= 1e9 and 40 <= runtime <= 300 and 1 <= month <= 12 and 1900 <= year <= 2100):
        return jsonify(error="Values out of range (budget 10k-1B, runtime 40-300, month 1-12, year 1900-2100)."), 400
    if not genres:
        return jsonify(error="Select at least one valid genre."), 400

    row = pd.DataFrame([{
        "budget": budget, "runtime": runtime, "year": year, "month": month,
        "is_english": int(data.get("language", "en") == "en"),
        "n_companies": n_companies, "genre_list": genres,
    }])
    X = build_features(row, ALL_GENRES)[REV_FEATURES]

    # per-tree spread gives a rough uncertainty band
    per_tree = np.array([t.predict(X.values)[0] for t in REV_MODEL.estimators_])
    point = float(np.expm1(per_tree.mean()))
    low, high = (float(np.expm1(np.percentile(per_tree, p))) for p in (10, 90))
    return jsonify(predicted_revenue=round(point), low=round(low), high=round(high),
                   model_r2_log=round(REV_R2, 3))


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found."), 404


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
