"""Versioned rubrics: validated, canonicalised and identified by sha256.

The hashing rule is ported from NSDA ``core/runner_configs.py``: identity is the sha256 of the
canonical YAML (sorted keys), so reformatting a file does not create a new rubric and any change
of wording, scale or anchors does. Every rating stores the rubric hash it was made under.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator
from sqlalchemy.orm import Session

from .models import Rubric


class RubricError(ValueError):
    """User-safe error for an invalid rubric."""


class Dimension(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label: str
    statement: str = Field(min_length=1)
    description: str | None = None


class RubricModel(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    version: str = Field(min_length=1, max_length=16)
    status: str = "draft"
    instructions: str = Field(min_length=1)
    scale_min: int
    scale_max: int
    anchors: dict[int, str] = Field(default_factory=dict)
    dimensions: list[Dimension] = Field(min_length=1)
    changelog: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "RubricModel":
        if self.scale_max <= self.scale_min:
            raise ValueError("scale_max must be greater than scale_min")
        for k in self.anchors:
            if not self.scale_min <= k <= self.scale_max:
                raise ValueError(f"anchor {k} is outside the scale")
        ids = [d.id for d in self.dimensions]
        if len(ids) != len(set(ids)):
            raise ValueError("dimension ids must be unique")
        return self

    @property
    def name(self) -> str:
        return f"{self.id}_v{self.version}"


def canonicalise(config: dict) -> str:
    """Deterministic YAML serialisation used for hashing (NSDA rule)."""
    return yaml.safe_dump(config, sort_keys=True, default_flow_style=False, allow_unicode=True)


def config_hash(config: dict) -> str:
    return hashlib.sha256(canonicalise(config).encode("utf-8")).hexdigest()


def parse_rubric(yaml_text: str) -> tuple[RubricModel, dict, str, str]:
    """Parse and validate rubric YAML. Returns (model, config, canonical_yaml, sha256)."""
    try:
        config = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise RubricError(f"Invalid YAML: {str(exc)[:200]}") from exc
    if not isinstance(config, dict):
        raise RubricError("A rubric must be a YAML mapping.")
    try:
        model = RubricModel(**config)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise RubricError(f"Invalid rubric: {problems[:300]}") from exc
    canonical = canonicalise(config)
    return model, config, canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def register_rubric(db: Session, yaml_text: str, source: str | None = None) -> Rubric:
    """Store a rubric; re-registering identical content is a no-op returning the existing row."""
    model, config, canonical, sha = parse_rubric(yaml_text)
    existing = db.get(Rubric, sha)
    if existing is not None:
        return existing
    row = Rubric(sha256=sha, rubric_id=model.id, version=model.version, yaml_text=canonical,
                 config=config, source=source)
    db.add(row)
    db.commit()
    return row


def register_rubric_file(db: Session, path: Path) -> Rubric:
    return register_rubric(db, Path(path).read_text(encoding="utf-8"), source=f"file:{Path(path).name}")


def find_rubric(db: Session, name_or_sha: str) -> Rubric | None:
    """Look up by full sha256, or by ``<id>_v<version>`` (latest registration wins)."""
    row = db.get(Rubric, name_or_sha)
    if row is not None:
        return row
    if "_v" in name_or_sha:
        rid, _, ver = name_or_sha.rpartition("_v")
        return (
            db.query(Rubric)
            .filter(Rubric.rubric_id == rid, Rubric.version == ver)
            .order_by(Rubric.created_at.desc())
            .first()
        )
    return None


def rubric_model(row: Rubric) -> RubricModel:
    return RubricModel(**row.config)
