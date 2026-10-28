"""Web app: login, blind rating flow, admin-only analytics and exports."""
from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from benchmark.rating.models import Assignment, Idea, Rating
from benchmark.rating.web.main import create_app

from .conftest import PASSWORD


@pytest.fixture()
def client(cfg, demo):
    return TestClient(create_app(cfg))


def login(client, username):
    r = client.post("/login", data={"username": username, "password": PASSWORD}, follow_redirects=False)
    assert r.status_code == 302


def test_login_required_and_bad_password(client):
    assert client.get("/rate", follow_redirects=False).status_code == 302
    r = client.post("/login", data={"username": "r1", "password": "wrong"})
    assert r.status_code == 401


def test_rating_screen_is_blind(client, db, demo):
    from benchmark.rating.ratings import record_ratings

    # another rater has already scored every idea, and the dataset contains AI ideas
    other = demo["users"]["r2"]
    for idea in db.query(Idea):
        record_ratings(db, idea_id=idea.id, rater_id=other.id, rubric_sha256=demo["dataset"].rubric_sha256,
                       scores={"novelty": 5, "usefulness": 1})
    login(client, "r1")
    page = client.get("/rate?dataset=demo").text
    assert "The idea" in page and "This idea is novel." in page
    for leak in ("demo-model", "source", "human", ">ai<", "mean", "SD", "r2"):
        assert leak not in page, f"rating screen leaks {leak!r}"


def test_submit_rating_and_flag_then_next(client, db, demo):
    login(client, "r1")
    client.get("/rate?dataset=demo")
    a = db.query(Assignment).filter_by(rater_id=demo["users"]["r1"].id, status="pending").one()
    r = client.post("/rate/submit", data={"assignment_id": a.id, "score_novelty": "4",
                                          "score_usefulness": "2"}, follow_redirects=False)
    assert r.status_code == 303
    db.expire_all()
    assert db.query(Rating).filter_by(idea_id=a.idea_id, rater_id=demo["users"]["r1"].id).count() == 2
    rated = db.get(Rating, db.query(Rating.id).filter_by(idea_id=a.idea_id).first()[0])
    assert rated.session_id and rated.duration_ms is not None
    client.get("/rate?dataset=demo")
    b = db.query(Assignment).filter_by(rater_id=demo["users"]["r1"].id, status="pending").one()
    assert b.idea_id != a.idea_id
    client.post("/rate/submit", data={"assignment_id": b.id, "flag": "off_task"})
    db.expire_all()
    assert db.get(Assignment, b.id).status == "done"


def test_incomplete_scores_are_rejected(client, db, demo):
    login(client, "r1")
    client.get("/rate?dataset=demo")
    a = db.query(Assignment).filter_by(rater_id=demo["users"]["r1"].id, status="pending").one()
    r = client.post("/rate/submit", data={"assignment_id": a.id, "score_novelty": "4"}, follow_redirects=False)
    assert "msg=" in r.headers["location"]
    db.expire_all()
    assert db.get(Assignment, a.id).status == "pending"


def test_analytics_and_exports_are_admin_only(client):
    login(client, "r1")
    assert client.get("/admin/analytics", follow_redirects=False).status_code == 302
    assert client.get("/api/export/ratings.csv?dataset=demo").status_code == 403
    client.get("/logout")
    login(client, "admin")
    assert client.get("/admin/analytics?dataset=demo").status_code == 200
    for page in ("/admin", "/admin/datasets", "/admin/users", "/admin/runs", "/admin/exports"):
        assert client.get(page).status_code == 200, page


def test_export_api_basic_auth(cfg, demo):
    c = TestClient(create_app(cfg))
    assert c.get("/api/export/ideas.csv?dataset=demo").status_code == 401
    token = base64.b64encode(f"admin:{PASSWORD}".encode()).decode()
    r = c.get("/api/export/ideas.csv?dataset=demo", headers={"Authorization": f"Basic {token}"})
    assert r.status_code == 200 and r.text.startswith("dataset,idea_id,idea_text")
