"""Exports in the benchmark's long format (the README's "Run it on your own ideas" schema).

Two kinds:

``write_internal``  everything, including held-out scores, for the analysis pipeline. Written
                    under ``data/processed/`` (gitignored by default).
``write_release``   what may be published: all ideas and tasks, calibration ratings only, the
                    list of held-out idea ids without scores, and a manifest with file hashes.

Rater identities are pseudonymised (``rater_code``) in every export; usernames never leave the
database. Ported from NSDA ``core/exports.py`` (payload + manifest + filename rules), reshaped to
CSV because the analysis side reads CSV.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from .models import Dataset, Idea, JudgeRun, JudgeScore, Rating, Rubric, Task, User
from .rubrics import rubric_model

EXPORT_SCHEMA = "orgsci-ratings-v1"

IDEA_COLUMNS = ["dataset", "idea_id", "idea_text", "task_id", "source", "model_name",
                "model_version", "generation_condition", "split"]
TASK_COLUMNS = ["dataset", "task_id", "task_prompt_verbatim"]
RATING_COLUMNS = ["dataset", "idea_id", "rater_id", "dimension", "score", "scale_min", "scale_max",
                  "rubric", "rubric_sha256", "condition", "rater_lab", "rater_expertise",
                  "rater_source", "split", "created_at"]
JUDGE_COLUMNS = ["dataset", "run_id", "run_name", "config_sha256", "backend", "model_name",
                 "model_version", "prompt_name", "perturbation_id", "temperature", "idea_id",
                 "dimension", "score", "sample_index", "cache_key"]


def rater_code(user_id: str) -> str:
    """Stable pseudonym for a rater: R + first 10 hex chars of sha256(user id)."""
    return "R" + hashlib.sha256(user_id.encode()).hexdigest()[:10]


def _csv(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def idea_rows(db: Session, dataset: Dataset) -> list[dict]:
    tasks = {t.id: t.task_key for t in db.query(Task).filter_by(dataset_id=dataset.id)}
    return [
        {"dataset": dataset.name, "idea_id": i.idea_key, "idea_text": i.text, "task_id": tasks[i.task_id],
         "source": i.source, "model_name": i.model_name or "", "model_version": i.model_version or "",
         "generation_condition": i.generation_condition or "", "split": i.split}
        for i in db.query(Idea).filter_by(dataset_id=dataset.id).order_by(Idea.idea_key)
    ]


def task_rows(db: Session, dataset: Dataset) -> list[dict]:
    return [
        {"dataset": dataset.name, "task_id": t.task_key, "task_prompt_verbatim": t.prompt_verbatim}
        for t in db.query(Task).filter_by(dataset_id=dataset.id).order_by(Task.task_key)
    ]


def rating_rows(db: Session, dataset: Dataset, splits: set[str] | None = None) -> list[dict]:
    rub_row = db.get(Rubric, dataset.rubric_sha256)
    rub = rubric_model(rub_row)
    users = {u.id: u for u in db.query(User)}
    q = (
        db.query(Rating, Idea)
        .join(Idea, Idea.id == Rating.idea_id)
        .filter(Idea.dataset_id == dataset.id)
        .order_by(Idea.idea_key, Rating.dimension, Rating.created_at)
    )
    out = []
    for rating, idea in q:
        if splits is not None and idea.split not in splits:
            continue
        u = users.get(rating.rater_id)
        out.append({
            "dataset": dataset.name, "idea_id": idea.idea_key, "rater_id": rater_code(rating.rater_id),
            "dimension": rating.dimension, "score": rating.score, "scale_min": rub.scale_min,
            "scale_max": rub.scale_max, "rubric": rub.name, "rubric_sha256": rating.rubric_sha256,
            "condition": rating.condition or "", "rater_lab": (u.lab or "") if u else "",
            "rater_expertise": u.expertise if u else "", "rater_source": rating.source,
            "split": idea.split,
            "created_at": rating.created_at.isoformat() if rating.created_at else "",
        })
    return out


def judge_rows(db: Session, dataset: Dataset, splits: set[str] | None = None) -> list[dict]:
    ideas = {i.id: i for i in db.query(Idea).filter_by(dataset_id=dataset.id)}
    out = []
    for run in db.query(JudgeRun).filter_by(dataset_id=dataset.id).order_by(JudgeRun.created_at):
        for s in db.query(JudgeScore).filter_by(run_id=run.id).order_by(JudgeScore.idea_id, JudgeScore.dimension):
            idea = ideas[s.idea_id]
            if splits is not None and idea.split not in splits:
                continue
            out.append({
                "dataset": dataset.name, "run_id": run.id, "run_name": run.name,
                "config_sha256": run.config_sha256, "backend": run.backend, "model_name": run.model_name,
                "model_version": run.model_version or "", "prompt_name": run.prompt_name,
                "perturbation_id": run.perturbation_id, "temperature": run.temperature,
                "idea_id": idea.idea_key, "dimension": s.dimension,
                "score": "" if s.score is None else s.score, "sample_index": s.sample_index,
                "cache_key": s.cache_key or "",
            })
    return out


def render(db: Session, dataset: Dataset, what: str, splits: set[str] | None = None) -> str:
    """CSV text for one table: ideas | tasks | ratings | judge_scores."""
    if what == "ideas":
        return _csv(idea_rows(db, dataset), IDEA_COLUMNS)
    if what == "tasks":
        return _csv(task_rows(db, dataset), TASK_COLUMNS)
    if what == "ratings":
        return _csv(rating_rows(db, dataset, splits), RATING_COLUMNS)
    if what == "judge_scores":
        return _csv(judge_rows(db, dataset, splits), JUDGE_COLUMNS)
    raise ValueError(f"unknown export table {what!r}")


def _write(out_dir: Path, name: str, text: str) -> dict:
    path = out_dir / name
    path.write_text(text, encoding="utf-8")
    return {"file": name, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "rows": max(text.count("\n") - 1, 0)}


def write_internal(db: Session, dataset: Dataset, out_dir: Path) -> dict:
    """All tables, all splits, for the analysis pipeline. Returns the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [
        _write(out_dir, "ideas.csv", render(db, dataset, "ideas")),
        _write(out_dir, "tasks.csv", render(db, dataset, "tasks")),
        _write(out_dir, "ratings_long.csv", render(db, dataset, "ratings")),
        _write(out_dir, "judge_scores.csv", render(db, dataset, "judge_scores")),
    ]
    return _manifest(db, dataset, out_dir, files, kind="internal")


def write_release(db: Session, dataset: Dataset, out_dir: Path) -> dict:
    """The publishable package: calibration ratings only; held-out ideas listed without scores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    heldout = [r for r in idea_rows(db, dataset) if r["split"] == "heldout"]
    files = [
        _write(out_dir, "ideas.csv", render(db, dataset, "ideas")),
        _write(out_dir, "tasks.csv", render(db, dataset, "tasks")),
        _write(out_dir, "ratings_long.csv", render(db, dataset, "ratings", splits={"calibration"})),
        _write(out_dir, "heldout_ideas.csv", _csv(heldout, ["dataset", "idea_id", "task_id"])),
    ]
    return _manifest(db, dataset, out_dir, files, kind="release")


def _manifest(db: Session, dataset: Dataset, out_dir: Path, files: list[dict], kind: str) -> dict:
    rub = db.get(Rubric, dataset.rubric_sha256)
    manifest = {
        "schema": EXPORT_SCHEMA, "kind": kind, "dataset": dataset.name,
        "rubric": f"{rub.rubric_id}_v{rub.version}", "rubric_sha256": rub.sha256,
        "target_raters_per_idea": dataset.target_raters,
        "exported_at": datetime.now(timezone.utc).isoformat(), "files": files,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
