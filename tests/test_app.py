import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["movies"] > 4000


def test_index(client):
    assert client.get("/").status_code == 200


def test_search(client):
    r = client.get("/api/search?q=avat")
    assert r.status_code == 200 and any("Avatar" in m["title"] for m in r.json)


def test_recommend_ok(client):
    r = client.get("/api/recommend?title=The Dark Knight Rises&n=5")
    assert r.status_code == 200
    recs = r.json["recommendations"]
    assert len(recs) == 5
    assert all(x["title"] != "The Dark Knight Rises" for x in recs)
    assert recs[0]["similarity"] >= recs[-1]["similarity"]


def test_recommend_case_insensitive_and_genre_filter(client):
    r = client.get("/api/recommend?title=toy story&n=5&genre=Animation")
    assert r.status_code == 200
    assert all("Animation" in x["genres"] for x in r.json["recommendations"])


def test_recommend_not_found(client):
    r = client.get("/api/recommend?title=zzzznotamovie")
    assert r.status_code == 404 and "error" in r.json


def test_recommend_missing_title(client):
    assert client.get("/api/recommend").status_code == 400


def test_recommend_bad_n(client):
    assert client.get("/api/recommend?title=Avatar&n=abc").status_code == 400


def test_predict_ok(client):
    r = client.post("/api/predict-revenue", json={"budget": 100000000, "runtime": 120, "year": 2020,
                                                  "month": 7, "genres": ["Action", "Adventure"]})
    assert r.status_code == 200
    assert r.json["low"] <= r.json["predicted_revenue"] <= r.json["high"] * 1.5


def test_predict_validation(client):
    assert client.post("/api/predict-revenue", json={"budget": "abc"}).status_code == 400
    assert client.post("/api/predict-revenue", json={"budget": 1e8, "runtime": 120, "year": 2020,
                                                     "month": 13, "genres": ["Action"]}).status_code == 400
    assert client.post("/api/predict-revenue", json={"budget": 1e8, "runtime": 120, "year": 2020,
                                                     "month": 6, "genres": []}).status_code == 400
