# CineRecommender

Content-based movie recommendations plus box-office revenue prediction.

## Structure
```
movie-recommender/
├── data/                  tmdb_5000_movies.csv, tmdb_5000_credits.csv
├── models/                recommender.joblib, revenue_model.joblib (created by train.py)
├── templates/index.html   UI
├── static/                style.css, app.js
├── tests/test_app.py      API tests
├── train.py               trains both models from the CSVs
├── app.py                 Flask app + JSON API
├── tmdb_movies_analysis.ipynb   exploration and model comparison
├── requirements.txt  Procfile  Dockerfile
```

## Run locally
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py          # ~30 s; writes models/*.joblib
python app.py            # http://127.0.0.1:5000
pytest -q tests          # optional
```

## API
| Method | Endpoint | Notes |
|---|---|---|
| GET | `/health` | liveness check |
| GET | `/api/search?q=avat` | title autocomplete (max 8) |
| GET | `/api/recommend?title=Avatar&n=10&genre=Action` | `n` 1-30, `genre` optional; 404 + suggestions if title unknown |
| POST | `/api/predict-revenue` | JSON: `budget, runtime, year, month, genres[]`, optional `language, n_companies` |

Example:
```bash
curl "http://127.0.0.1:5000/api/recommend?title=Toy%20Story&n=3"
curl -X POST http://127.0.0.1:5000/api/predict-revenue -H "Content-Type: application/json" \
  -d '{"budget":150000000,"runtime":130,"year":2025,"month":6,"genres":["Action","Science Fiction"]}'
```

## Deploy
- **Gunicorn:** `gunicorn app:app --workers 2 --bind 0.0.0.0:8000`
- **Docker:** `docker build -t movie-rec . && docker run -p 8000:8000 movie-rec`
- **Render/Railway/Heroku:** the `Procfile` works as is; commit `models/` (about 10 MB) or run `python train.py` in the build step.

## Design notes
- `app.py` imports `build_features` from `train.py`, so training and serving share one feature pipeline.
- Similarity is computed per request from the sparse TF-IDF matrix (one row x matrix), so no 4803x4803 matrix is held in memory.
- Revenue R² is about 0.48 on log revenue, so treat predictions as rough estimates. The range shown is the 10th-90th percentile across the forest's trees.
