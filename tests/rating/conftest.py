"""Fixtures: a throwaway config, database and app per test (no shared state, no network)."""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.rating.auth import hash_password
from benchmark.rating.config import AppConfig
from benchmark.rating.models import Dataset, User, init_db
from benchmark.rating.rubrics import register_rubric_file
from benchmark.rating import importers

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "demo"
RUBRIC = REPO / "configs" / "rubrics" / "novelty_usefulness_v1.0.yaml"
PASSWORD = "test-password"


@pytest.fixture()
def cfg(tmp_path) -> AppConfig:
    return AppConfig(db_path=tmp_path / "rating.db", data_dir=tmp_path, rubrics_dir=REPO / "configs" / "rubrics")


@pytest.fixture()
def db(cfg):
    session = init_db(cfg.db_path)()
    yield session
    session.close()


@pytest.fixture()
def demo(db):
    """Demo dataset with tasks and ideas loaded, target 3 raters per idea, and four users."""
    rub = register_rubric_file(db, RUBRIC)
    ds = Dataset(name="demo", rubric_sha256=rub.sha256, target_raters=3)
    db.add(ds)
    db.commit()
    importers.import_tasks(db, ds, DEMO / "tasks.csv")
    importers.import_ideas(db, ds, DEMO / "ideas.csv")
    users = {}
    for name, role in [("admin", "admin"), ("r1", "rater"), ("r2", "rater"), ("r3", "rater"), ("r4", "rater")]:
        u = User(username=name, display_name=name, role=role, password_hash=hash_password(PASSWORD))
        db.add(u)
        users[name] = u
    db.commit()
    return {"dataset": ds, "rubric": rub, "users": users}
