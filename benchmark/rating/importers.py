"""Import tasks, ideas, historical ratings and judge scores from CSV.

Column names follow the "Bring your own ratings" format in the README, so a lab's existing
export loads without reshaping. Every import is idempotent on its natural keys and records its
source file in ``source`` columns and in the audit log.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from sqlalchemy.orm import Session

from .models import SPLITS, AuditLog, Dataset, Idea, JudgeRun, JudgeScore, Rating, Task, User, utcnow
from .rubrics import config_hash, rubric_model


class ImportErrorWithRow(ValueError):
    pass


def _read_csv(path: Path, required: tuple[str, ...]) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ImportErrorWithRow(f"{Path(path).name}: missing columns {missing}")
        return [{k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in reader]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_tasks(db: Session, dataset: Dataset, path: Path) -> int:
    """``task_id, task_prompt_verbatim``. Returns the number of new tasks."""
    rows = _read_csv(path, ("task_id", "task_prompt_verbatim"))
    added = 0
    for r in rows:
        exists = db.query(Task).filter_by(dataset_id=dataset.id, task_key=r["task_id"]).one_or_none()
        if exists:
            if exists.prompt_verbatim != r["task_prompt_verbatim"]:
                raise ImportErrorWithRow(f"task {r['task_id']}: prompt differs from the stored one")
            continue
        db.add(Task(dataset_id=dataset.id, task_key=r["task_id"], prompt_verbatim=r["task_prompt_verbatim"]))
        added += 1
    db.commit()
    return added


def import_ideas(db: Session, dataset: Dataset, path: Path) -> int:
    """``idea_id, idea_text, task_id, source`` plus optional ``model_name, model_version,
    generation_condition, split``. Tasks must be imported first."""
    rows = _read_csv(path, ("idea_id", "idea_text", "task_id", "source"))
    tasks = {t.task_key: t for t in db.query(Task).filter_by(dataset_id=dataset.id)}
    added = 0
    for i, r in enumerate(rows, start=2):
        if r["task_id"] not in tasks:
            raise ImportErrorWithRow(f"row {i}: unknown task_id {r['task_id']!r} (import tasks first)")
        if r["source"] not in ("human", "ai"):
            raise ImportErrorWithRow(f"row {i}: source must be 'human' or 'ai'")
        split = r.get("split") or "unassigned"
        if split not in SPLITS:
            raise ImportErrorWithRow(f"row {i}: split must be one of {SPLITS}")
        if db.query(Idea).filter_by(dataset_id=dataset.id, idea_key=r["idea_id"]).one_or_none():
            continue
        db.add(Idea(
            dataset_id=dataset.id, idea_key=r["idea_id"], task_id=tasks[r["task_id"]].id,
            text=r["idea_text"], text_sha256=text_sha256(r["idea_text"]), source=r["source"],
            model_name=r.get("model_name") or None, model_version=r.get("model_version") or None,
            generation_condition=r.get("generation_condition") or None, split=split,
            meta={"import_source": Path(path).name},
        ))
        added += 1
    db.commit()
    return added


def _external_rater(db: Session, dataset: Dataset, rater_key: str, row: dict, source: str) -> User:
    username = f"ext:{dataset.name}:{rater_key}"
    user = db.query(User).filter_by(username=username).one_or_none()
    if user is None:
        user = User(username=username, display_name=rater_key, role="external", password_hash=None,
                    active=False, lab=row.get("rater_lab") or None,
                    expertise=row.get("rater_expertise") or "unknown",
                    domain=row.get("rater_domain") or None, source=source)
        db.add(user)
        db.flush()
    return user


def import_ratings(db: Session, dataset: Dataset, path: Path) -> int:
    """Historical rater-level ratings: ``idea_id, rater_id, dimension, score`` plus optional
    ``condition, rater_lab, rater_expertise, rater_domain``. Raters become ``external`` users.
    Scores are validated against the dataset's rubric scale and dimensions."""
    rows = _read_csv(path, ("idea_id", "rater_id", "dimension", "score"))
    from .models import Rubric

    rub = rubric_model(db.get(Rubric, dataset.rubric_sha256))
    dims = {d.id for d in rub.dimensions}
    ideas = {i.idea_key: i for i in db.query(Idea).filter_by(dataset_id=dataset.id)}
    source = f"import:{Path(path).name}"
    added = 0
    for i, r in enumerate(rows, start=2):
        idea = ideas.get(r["idea_id"])
        if idea is None:
            raise ImportErrorWithRow(f"row {i}: unknown idea_id {r['idea_id']!r}")
        if r["dimension"] not in dims:
            raise ImportErrorWithRow(f"row {i}: dimension {r['dimension']!r} not in rubric {rub.name}")
        score = float(r["score"])
        if not rub.scale_min <= score <= rub.scale_max:
            raise ImportErrorWithRow(f"row {i}: score {score} outside {rub.scale_min}-{rub.scale_max}")
        rater = _external_rater(db, dataset, r["rater_id"], r, source)
        exists = db.query(Rating).filter_by(idea_id=idea.id, rater_id=rater.id, dimension=r["dimension"],
                                            rubric_sha256=dataset.rubric_sha256).one_or_none()
        if exists:
            continue
        db.add(Rating(idea_id=idea.id, rater_id=rater.id, dimension=r["dimension"], score=score,
                      rubric_sha256=dataset.rubric_sha256, condition=r.get("condition") or None,
                      source=source))
        added += 1
    db.add(AuditLog(entity="dataset", entity_id=dataset.id, action="import_ratings",
                    payload={"file": Path(path).name, "added": added}))
    db.commit()
    return added


def import_judge_scores(db: Session, dataset: Dataset, path: Path, *, name: str, backend: str,
                        model_name: str, model_version: str | None, prompt_name: str,
                        perturbation_id: str = "P0_baseline", temperature: float = 0.0,
                        extra_config: dict | None = None) -> JudgeRun:
    """Judge scores ``idea_id, dimension, score`` plus optional ``sample_index, cache_key,
    raw_text``. Creates one JudgeRun whose config hash identifies what produced the scores."""
    rows = _read_csv(path, ("idea_id", "dimension", "score"))
    config = {"backend": backend, "model_name": model_name, "model_version": model_version,
              "prompt_name": prompt_name, "perturbation_id": perturbation_id,
              "temperature": temperature, **(extra_config or {})}
    run = JudgeRun(dataset_id=dataset.id, name=name, backend=backend, model_name=model_name,
                   model_version=model_version, prompt_name=prompt_name,
                   perturbation_id=perturbation_id, temperature=temperature, config=config,
                   config_sha256=config_hash(config), source=f"import:{Path(path).name}",
                   produced_at=utcnow())
    db.add(run)
    db.flush()
    ideas = {i.idea_key: i for i in db.query(Idea).filter_by(dataset_id=dataset.id)}
    for i, r in enumerate(rows, start=2):
        idea = ideas.get(r["idea_id"])
        if idea is None:
            raise ImportErrorWithRow(f"row {i}: unknown idea_id {r['idea_id']!r}")
        score = r.get("score")
        db.add(JudgeScore(run_id=run.id, idea_id=idea.id, dimension=r["dimension"],
                          score=float(score) if score not in (None, "", "NA") else None,
                          sample_index=int(r.get("sample_index") or 0),
                          cache_key=r.get("cache_key") or None, raw_text=r.get("raw_text") or None))
    db.commit()
    return run
