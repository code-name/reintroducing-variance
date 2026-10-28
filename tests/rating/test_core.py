"""Core plumbing: rubric hashing, imports, sampler coverage, ratings, aggregates, exports."""
from __future__ import annotations

import csv
import io

import pytest

from benchmark.rating import importers, sampler
from benchmark.rating.aggregate import idea_aggregates, most_contested, summarise
from benchmark.rating.exports import RATING_COLUMNS, rater_code, render, write_release
from benchmark.rating.models import AuditLog, Idea, Rating
from benchmark.rating.ratings import RatingError, record_ratings
from benchmark.rating.rubrics import RubricError, parse_rubric, register_rubric

from .conftest import DEMO, RUBRIC


def test_rubric_hash_ignores_formatting_and_tracks_content(db):
    text = RUBRIC.read_text()
    a = register_rubric(db, text)
    b = register_rubric(db, "# a comment\n" + text)          # same content, new formatting
    assert a.sha256 == b.sha256
    _, _, _, sha_changed = parse_rubric(text.replace("This idea is novel.", "This idea is original."))
    assert sha_changed != a.sha256


def test_rubric_validation_rejects_bad_scale():
    bad = RUBRIC.read_text().replace("scale_max: 5", "scale_max: 0")
    with pytest.raises(RubricError):
        parse_rubric(bad)


def test_imports_are_idempotent(db, demo):
    ds = demo["dataset"]
    assert importers.import_ideas(db, ds, DEMO / "ideas.csv") == 0
    assert db.query(Idea).filter_by(dataset_id=ds.id).count() == 6


def test_historical_ratings_become_external_raters(db, demo):
    n = importers.import_ratings(db, demo["dataset"], DEMO / "ratings_long.csv")
    assert n == 8
    from benchmark.rating.models import User

    ext = db.query(User).filter_by(role="external").all()
    assert len(ext) == 2 and all(u.password_hash is None and not u.active for u in ext)
    assert {r.condition for r in db.query(Rating)} == {"baseline", "treatment"}


def test_sampler_reaches_k_and_never_repeats_a_rater(db, demo):
    ds, users = demo["dataset"], demo["users"]
    seen: dict[str, list[str]] = {}
    for name in ("r1", "r2", "r3", "r4"):
        while (a := sampler.next_assignment(db, ds, users[name].id)) is not None:
            seen.setdefault(a.idea_id, []).append(name)
            record_ratings(db, idea_id=a.idea_id, rater_id=users[name].id, rubric_sha256=ds.rubric_sha256,
                           scores={"novelty": 3, "usefulness": 3})
            sampler.complete(db, a)
    assert all(len(v) == len(set(v)) for v in seen.values())      # no rater twice on one idea
    assert all(len(v) == 3 for v in seen.values())                 # exactly k = 3 per idea
    assert len(seen) == 6


def test_sampler_resumes_pending_assignment(db, demo):
    ds, u = demo["dataset"], demo["users"]["r1"]
    a1 = sampler.next_assignment(db, ds, u.id)
    a2 = sampler.next_assignment(db, ds, u.id)
    assert a1.id == a2.id and a1.method == "coverage_v1"


def test_rating_validation_and_audit(db, demo):
    ds, u = demo["dataset"], demo["users"]["r1"]
    idea = db.query(Idea).first()
    with pytest.raises(RatingError):
        record_ratings(db, idea_id=idea.id, rater_id=u.id, rubric_sha256=ds.rubric_sha256,
                       scores={"novelty": 9, "usefulness": 3})
    db.rollback()
    with pytest.raises(RatingError):
        record_ratings(db, idea_id=idea.id, rater_id=u.id, rubric_sha256=ds.rubric_sha256, scores={"novelty": 3})
    db.rollback()
    record_ratings(db, idea_id=idea.id, rater_id=u.id, rubric_sha256=ds.rubric_sha256, scores={"novelty": 2, "usefulness": 4})
    record_ratings(db, idea_id=idea.id, rater_id=u.id, rubric_sha256=ds.rubric_sha256, scores={"novelty": 3, "usefulness": 4})
    assert db.query(Rating).filter_by(rater_id=u.id).count() == 2      # updated, not duplicated
    upd = db.query(AuditLog).filter_by(action="update").one()
    assert upd.payload == {"dimension": "novelty", "from": 2.0, "to": 3.0}


def test_aggregates_report_spread_and_contested(db, demo):
    importers.import_ratings(db, demo["dataset"], DEMO / "ratings_long.csv")
    rows = idea_aggregates(db, demo["dataset"].id, demo["dataset"].rubric_sha256)
    d002 = next(r for r in rows if r["idea_key"] == "d002" and r["dimension"] == "novelty")
    assert d002["n"] == 2 and d002["mean"] == 4.5 and round(d002["sd"], 4) == 0.7071
    base_only = idea_aggregates(db, demo["dataset"].id, demo["dataset"].rubric_sha256, exclude_conditions={"treatment"})
    assert all(r["n"] == 1 for r in base_only)
    assert most_contested(rows, "novelty", min_n=2)[0]["sd"] >= most_contested(rows, "novelty", min_n=2)[-1]["sd"]
    assert summarise([]) == {"n": 0, "mean": None, "sd": None, "se": None}


def test_exports_match_load_schema_and_pseudonymise(db, demo):
    importers.import_ratings(db, demo["dataset"], DEMO / "ratings_long.csv")
    text = render(db, demo["dataset"], "ratings")
    rows = list(csv.DictReader(io.StringIO(text)))
    from benchmark.load import REQUIRED_RATING_COLUMNS

    assert set(REQUIRED_RATING_COLUMNS) <= set(rows[0]) and list(rows[0]) == RATING_COLUMNS
    assert all(r["rater_id"].startswith("R") and "ext:" not in r["rater_id"] for r in rows)
    assert rater_code("x") == rater_code("x")


def test_release_withholds_heldout_scores(db, demo, tmp_path):
    ds, u = demo["dataset"], demo["users"]["r1"]
    for idea in db.query(Idea).filter_by(dataset_id=ds.id):
        record_ratings(db, idea_id=idea.id, rater_id=u.id, rubric_sha256=ds.rubric_sha256, scores={"novelty": 3, "usefulness": 3})
    manifest = write_release(db, ds, tmp_path / "rel")
    released = list(csv.DictReader(open(tmp_path / "rel" / "ratings_long.csv")))
    assert {r["split"] for r in released} == {"calibration"}
    heldout = list(csv.DictReader(open(tmp_path / "rel" / "heldout_ideas.csv")))
    assert {h["idea_id"] for h in heldout} == {"d003", "d006"}
    assert manifest["kind"] == "release" and all(len(f["sha256"]) == 64 for f in manifest["files"])


def test_judge_score_import_hashes_config(db, demo, tmp_path):
    f = tmp_path / "scores.csv"
    f.write_text("idea_id,dimension,score\nd001,novelty,2\nd001,usefulness,\n")
    run = importers.import_judge_scores(db, demo["dataset"], f, name="t", backend="openai",
                                        model_name="m", model_version="v", prompt_name="pointwise_v1.0")
    run2 = importers.import_judge_scores(db, demo["dataset"], f, name="t2", backend="openai",
                                         model_name="m", model_version="v", prompt_name="pointwise_v1.0")
    assert run.config_sha256 == run2.config_sha256 and len(run.config_sha256) == 64
    assert "d001" in render(db, demo["dataset"], "judge_scores")
