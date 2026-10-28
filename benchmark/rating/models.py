"""SQLAlchemy 2.0 models (SQLite first, PostgreSQL portable).

Adapted from NSDA ``db/models.py``. Mapping from the OCR app:

    NSDA OCR app            this app
    -------------------     -----------------------------------------------------------
    User                    User, plus rater metadata (lab, expertise, domain, training)
    Dataset / Document      Dataset (a set of ideas rated under one rubric)
    Page                    Idea (with its Task, source, split)
    AnnotationUnit          (none: the unit is the idea x dimension)
    Annotation              Rating (one row per idea x rater x dimension x rubric)
    Annotation reject/flag  IdeaFlag (invalid / off_task / duplicate / unclear)
    Assignment              Assignment (coverage sampler log, plus served_at for timing)
    RunnerConfig (sha256)   Rubric (sha256 of canonical YAML) and JudgeRun.config_sha256
    ModelRun / PageOutput   JudgeRun / JudgeScore
    Disagreement            not ported: truth is an aggregate, not a resolution
    AuditLog                AuditLog (every rating create/update/flag)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


ROLES = ("admin", "rater", "external")
EXPERTISE = ("novice", "trained_novice", "expert", "manager", "unknown")
SPLITS = ("calibration", "heldout", "unassigned")
FLAGS = ("invalid", "off_task", "duplicate", "unclear")


class User(Base):
    """A rater or admin. ``external`` users are raters imported from historical data: they have
    no password and cannot log in, but their ratings are first-class rows."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    username: Mapped[str] = mapped_column(String(128), unique=True)
    display_name: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(16), default="rater")
    password_hash: Mapped[str | None] = mapped_column(String(256), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    lab: Mapped[str | None] = mapped_column(String(128), nullable=True)
    expertise: Mapped[str] = mapped_column(String(32), default="unknown")
    domain: Mapped[str | None] = mapped_column(String(128), nullable=True)
    training: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(256), default="app")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Rubric(Base):
    """A versioned rubric. Primary key is the sha256 of the canonical YAML (NSDA runner-config
    rule): the same content always maps to the same row; any change is a new rubric."""

    __tablename__ = "rubrics"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    rubric_id: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(16))
    yaml_text: Mapped[str] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(JSON)
    source: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="open")  # open|locked
    rubric_sha256: Mapped[str] = mapped_column(ForeignKey("rubrics.sha256"))
    target_raters: Mapped[int] = mapped_column(Integer, default=10)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (UniqueConstraint("dataset_id", "task_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    task_key: Mapped[str] = mapped_column(String(128))
    prompt_verbatim: Mapped[str] = mapped_column(Text)


class Idea(Base):
    __tablename__ = "ideas"
    __table_args__ = (UniqueConstraint("dataset_id", "idea_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    idea_key: Mapped[str] = mapped_column(String(128))
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"))
    text: Mapped[str] = mapped_column(Text)
    text_sha256: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(16))  # human|ai  (never shown to raters)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    generation_condition: Mapped[str | None] = mapped_column(String(128), nullable=True)
    split: Mapped[str] = mapped_column(String(16), default="unassigned")
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Rating(Base):
    """One score: idea x rater x dimension under one rubric. Revisions update the row and are
    recorded in the audit log with the previous value."""

    __tablename__ = "ratings"
    __table_args__ = (UniqueConstraint("idea_id", "rater_id", "dimension", "rubric_sha256"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    idea_id: Mapped[str] = mapped_column(ForeignKey("ideas.id"))
    rater_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    dimension: Mapped[str] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Float)
    rubric_sha256: Mapped[str] = mapped_column(ForeignKey("rubrics.sha256"))
    condition: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(256), default="app")
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class IdeaFlag(Base):
    """A rater's categorical call that an idea should not be scored as-is. This is the only
    place where a single answer is meaningful, so it is the only adjudicable record."""

    __tablename__ = "idea_flags"
    __table_args__ = (UniqueConstraint("idea_id", "rater_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    idea_id: Mapped[str] = mapped_column(ForeignKey("ideas.id"))
    rater_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    flag: Mapped[str] = mapped_column(String(16))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Assignment(Base):
    """Every idea handed to a rater, with the method and priority that chose it (NSDA rule:
    ordering is auditable and replayable)."""

    __tablename__ = "assignments"
    __table_args__ = (UniqueConstraint("idea_id", "rater_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    idea_id: Mapped[str] = mapped_column(ForeignKey("ideas.id"))
    rater_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    method: Mapped[str] = mapped_column(String(32), default="coverage_v1")
    priority_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(8), default="pending")  # pending|done|skipped
    served_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class JudgeRun(Base):
    """One run of an AI judge over a dataset. ``config_sha256`` is the hash of the canonical
    run configuration (backend, model version, prompt, perturbation, temperature)."""

    __tablename__ = "judge_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    name: Mapped[str] = mapped_column(String(128))
    backend: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_name: Mapped[str] = mapped_column(String(64))
    perturbation_id: Mapped[str] = mapped_column(String(64), default="P0_baseline")
    temperature: Mapped[float] = mapped_column(Float, default=0.0)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    config_sha256: Mapped[str] = mapped_column(String(64))
    source: Mapped[str | None] = mapped_column(String(256), nullable=True)
    produced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class JudgeScore(Base):
    __tablename__ = "judge_scores"
    __table_args__ = (UniqueConstraint("run_id", "idea_id", "dimension", "sample_index"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("judge_runs.id"))
    idea_id: Mapped[str] = mapped_column(ForeignKey("ideas.id"))
    dimension: Mapped[str] = mapped_column(String(64))
    score: Mapped[float | None] = mapped_column(Float, nullable=True)  # None = parse failure
    sample_index: Mapped[int] = mapped_column(Integer, default=0)
    cache_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    entity: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


def make_engine(db_path):
    """SQLite engine with WAL and foreign keys on (NSDA rule)."""
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def make_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(db_path) -> sessionmaker:
    """Create the database file and tables if needed; return a session factory."""
    from pathlib import Path

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
    return make_session_factory(engine)
