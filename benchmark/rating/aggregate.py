"""Per-idea aggregates: the reference is the distribution of ratings, not one answer.

For each idea and dimension: n raters, mean, standard deviation (the idea's contestedness) and
standard error of the mean. As raters are added the standard error shrinks roughly with the
square root of n, while the standard deviation settles at the idea's true level of
disagreement. Both are reported. Pure Python so the web app needs no pandas.

The full statistics (ICC ceiling, leniency-adjusted rater models, the rater report and the
judge comparisons) live in ``benchmark.agree``.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from sqlalchemy.orm import Session

from .models import Idea, Rating


def summarise(scores: list[float]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "se": None}
    mean = statistics.fmean(scores)
    sd = statistics.stdev(scores) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    return {"n": n, "mean": mean, "sd": sd, "se": se}


def idea_aggregates(db: Session, dataset_id: str, rubric_sha256: str,
                    exclude_conditions: set[str] | None = None) -> list[dict]:
    """One row per (idea, dimension) with n, mean, sd, se. ``exclude_conditions`` drops ratings
    made under those conditions (e.g. to build a reference from baseline raters only)."""
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    q = (
        db.query(Rating.idea_id, Rating.dimension, Rating.score, Rating.condition)
        .join(Idea, Idea.id == Rating.idea_id)
        .filter(Idea.dataset_id == dataset_id, Rating.rubric_sha256 == rubric_sha256)
    )
    for idea_id, dim, score, condition in q:
        if exclude_conditions and condition in exclude_conditions:
            continue
        by_key[(idea_id, dim)].append(score)
    ideas = {i.id: i for i in db.query(Idea).filter_by(dataset_id=dataset_id)}
    out = []
    for (idea_id, dim), scores in by_key.items():
        idea = ideas[idea_id]
        out.append({"idea_id": idea_id, "idea_key": idea.idea_key, "text": idea.text,
                    "split": idea.split, "dimension": dim, **summarise(scores)})
    out.sort(key=lambda r: (r["idea_key"], r["dimension"]))
    return out


def most_contested(rows: list[dict], dimension: str, top: int = 20, min_n: int = 3) -> list[dict]:
    """Ideas with the largest rater spread on one dimension (n >= min_n)."""
    pool = [r for r in rows if r["dimension"] == dimension and r["sd"] is not None and r["n"] >= min_n]
    return sorted(pool, key=lambda r: -r["sd"])[:top]


def coverage_summary(db: Session, dataset) -> dict:
    """Counts for dashboards: ideas, ratings, raters, ideas at target, by split."""
    from .sampler import coverage

    ideas = db.query(Idea).filter_by(dataset_id=dataset.id).all()
    cov = coverage(db, dataset.id)
    n_ratings = (
        db.query(Rating).join(Idea, Idea.id == Rating.idea_id).filter(Idea.dataset_id == dataset.id).count()
    )
    raters = {
        r for (r,) in db.query(Rating.rater_id).join(Idea, Idea.id == Rating.idea_id)
        .filter(Idea.dataset_id == dataset.id).distinct()
    }
    at_target = sum(1 for i in ideas if cov.get(i.id, 0) >= dataset.target_raters)
    by_split: dict[str, int] = defaultdict(int)
    for i in ideas:
        by_split[i.split] += 1
    return {"ideas": len(ideas), "ratings": n_ratings, "raters": len(raters),
            "at_target": at_target, "target": dataset.target_raters, "by_split": dict(by_split)}
