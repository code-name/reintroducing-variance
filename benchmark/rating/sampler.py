"""Coverage sampler: hands each rater the next idea to rate.

Adapted from NSDA Corp (see NOTICE). Policy (``coverage_v1``):

1. A rater with a pending assignment in the dataset gets that idea again (resume).
2. Otherwise the candidates are ideas in the dataset that this rater has never been assigned and
   has never rated, and whose coverage is below its target: the idea's own ``target_raters``
   (stored in ``Idea.meta``) if it has one, otherwise the dataset's ``target_raters``. A
   per-idea target lets a subset of the reference set be rated by more people (for example 40)
   so that disagreement can be measured well. Coverage is the number of distinct raters who have
   rated the idea or hold a pending assignment for it.
3. Least-covered first. Ties are broken by a hash of (rater, idea), so different raters see
   different orders (no shared position effects) while each rater's order is reproducible.
4. Every assignment is logged with method and priority (coverage at serve time), so the order
   is auditable and replayable.

What differs from NSDA: the target is k raters per idea (default 10), with no stop on agreement
and no tie-break, because disagreement between raters is the quantity being measured.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from sqlalchemy.orm import Session

from .models import Assignment, Dataset, Idea, Rating, utcnow

METHOD = "coverage_v1"


def _tiebreak(rater_id: str, idea_id: str) -> str:
    return hashlib.sha256(f"{rater_id}:{idea_id}".encode()).hexdigest()


def target(idea: Idea, dataset: Dataset) -> int:
    """The number of raters an idea should get: its own target if set, else the dataset's."""
    t = (idea.meta or {}).get("target_raters")
    return int(t) if t else dataset.target_raters


def coverage(db: Session, dataset_id: str) -> Counter:
    """idea_id -> number of distinct raters who rated it or hold a pending assignment."""
    raters: dict[str, set[str]] = {}
    for idea_id, rater_id in (
        db.query(Rating.idea_id, Rating.rater_id).join(Idea, Idea.id == Rating.idea_id)
        .filter(Idea.dataset_id == dataset_id).distinct()
    ):
        raters.setdefault(idea_id, set()).add(rater_id)
    for idea_id, rater_id in db.query(Assignment.idea_id, Assignment.rater_id).filter(
        Assignment.dataset_id == dataset_id, Assignment.status == "pending"
    ):
        raters.setdefault(idea_id, set()).add(rater_id)
    return Counter({k: len(v) for k, v in raters.items()})


def next_assignment(db: Session, dataset: Dataset, rater_id: str) -> Assignment | None:
    """Return the rater's current or next assignment, or None when nothing is left for them."""
    pending = (
        db.query(Assignment)
        .filter_by(dataset_id=dataset.id, rater_id=rater_id, status="pending")
        .order_by(Assignment.served_at)
        .first()
    )
    if pending is not None:
        return pending
    if dataset.status != "open":
        return None
    seen = {a for (a,) in db.query(Assignment.idea_id).filter_by(rater_id=rater_id, dataset_id=dataset.id)}
    seen |= {
        i for (i,) in db.query(Rating.idea_id).join(Idea, Idea.id == Rating.idea_id)
        .filter(Idea.dataset_id == dataset.id, Rating.rater_id == rater_id)
    }
    cov = coverage(db, dataset.id)
    candidates = [
        idea.id for idea in db.query(Idea).filter_by(dataset_id=dataset.id)
        if idea.id not in seen and cov.get(idea.id, 0) < target(idea, dataset)
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: (cov.get(i, 0), _tiebreak(rater_id, i)))
    assignment = Assignment(idea_id=best, rater_id=rater_id, dataset_id=dataset.id, method=METHOD,
                            priority_score=float(cov.get(best, 0)), status="pending",
                            served_at=utcnow())
    db.add(assignment)
    db.commit()
    return assignment


def complete(db: Session, assignment: Assignment, status: str = "done") -> None:
    assignment.status = status
    assignment.completed_at = utcnow()
    db.commit()
