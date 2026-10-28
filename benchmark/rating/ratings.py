"""Record ratings and flags, with an audit trail (NSDA rule: every action is its own record)."""
from __future__ import annotations

from sqlalchemy.orm import Session

from .models import FLAGS, AuditLog, IdeaFlag, Rating, Rubric
from .rubrics import rubric_model


class RatingError(ValueError):
    pass


def record_ratings(db: Session, *, idea_id: str, rater_id: str, rubric_sha256: str,
                   scores: dict[str, float], condition: str | None = None,
                   duration_ms: int | None = None, session_id: str | None = None,
                   source: str = "app") -> list[Rating]:
    """Create or update one rating per dimension. Validates dimensions and scale against the
    rubric. Updates keep the row id and log the previous score."""
    rub = rubric_model(db.get(Rubric, rubric_sha256))
    dims = {d.id for d in rub.dimensions}
    unknown = set(scores) - dims
    if unknown:
        raise RatingError(f"unknown dimension(s) {sorted(unknown)} for rubric {rub.name}")
    missing = dims - set(scores)
    if missing:
        raise RatingError(f"missing score(s) for {sorted(missing)}")
    rows = []
    for dim, score in scores.items():
        score = float(score)
        if not rub.scale_min <= score <= rub.scale_max:
            raise RatingError(f"{dim}: score {score} outside {rub.scale_min}-{rub.scale_max}")
        row = db.query(Rating).filter_by(idea_id=idea_id, rater_id=rater_id, dimension=dim,
                                         rubric_sha256=rubric_sha256).one_or_none()
        if row is None:
            row = Rating(idea_id=idea_id, rater_id=rater_id, dimension=dim, score=score,
                         rubric_sha256=rubric_sha256, condition=condition, duration_ms=duration_ms,
                         session_id=session_id, source=source)
            db.add(row)
            db.flush()
            db.add(AuditLog(user_id=rater_id, entity="rating", entity_id=row.id, action="create",
                            payload={"dimension": dim, "score": score}))
        elif row.score != score:
            db.add(AuditLog(user_id=rater_id, entity="rating", entity_id=row.id, action="update",
                            payload={"dimension": dim, "from": row.score, "to": score}))
            row.score = score
        rows.append(row)
    db.commit()
    return rows


def record_flag(db: Session, *, idea_id: str, rater_id: str, flag: str, note: str | None = None) -> IdeaFlag:
    if flag not in FLAGS:
        raise RatingError(f"flag must be one of {FLAGS}")
    row = db.query(IdeaFlag).filter_by(idea_id=idea_id, rater_id=rater_id).one_or_none()
    if row is None:
        row = IdeaFlag(idea_id=idea_id, rater_id=rater_id, flag=flag, note=note)
        db.add(row)
    else:
        row.flag, row.note = flag, note
    db.flush()
    db.add(AuditLog(user_id=rater_id, entity="idea_flag", entity_id=row.id, action="flag",
                    payload={"flag": flag}))
    db.commit()
    return row
