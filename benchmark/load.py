"""Read and check a data folder.

A data folder holds the tables in the README's "Run it on your own ideas" format:

    ideas.csv         idea_id, idea_text, task_id (optional: source, split; target_raters for ideas
                      that need more raters than the dataset default, such as a subset rated by 40
                      people; advanced, 0 or 1, the organization's decision in a field test)
    tasks.csv         task_id, task_prompt_verbatim
    ratings_long.csv  idea_id, rater_id, dimension, score (optional: scale_min, scale_max, condition)
    judge_scores.csv  AI judge scores written by ``benchmark.judge`` (optional)

``examples/alternative_uses/`` is a complete example.

    python -m benchmark.load my_study/      # check a folder and print row counts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import REPO_DIR

EXAMPLE_DIR = REPO_DIR / "examples" / "alternative_uses"

REQUIRED_IDEA_COLUMNS = ("idea_id", "idea_text", "task_id")
REQUIRED_RATING_COLUMNS = ("idea_id", "rater_id", "dimension", "score", "scale_min", "scale_max")
REQUIRED_TASK_COLUMNS = ("task_id", "task_prompt_verbatim")


def read_folder(data_dir: Path | str = EXAMPLE_DIR) -> dict[str, pd.DataFrame]:
    """Every table present in a data folder, keyed by file name without ``.csv``."""
    d = Path(data_dir)
    tables = {p.stem: pd.read_csv(p) for p in sorted(d.glob("*.csv"))}
    missing = [t for t in ("ideas", "tasks") if t not in tables]
    if missing:
        raise FileNotFoundError(f"{d}: missing {', '.join(t + '.csv' for t in missing)}")
    return tables


def validate(tables: dict[str, pd.DataFrame]) -> None:
    """Schema and integrity checks; raises ValueError on the first problem."""
    ideas, tasks = tables["ideas"], tables["tasks"]
    ratings = tables.get("ratings_long", pd.DataFrame(columns=REQUIRED_RATING_COLUMNS))
    for cols, df, name in ((REQUIRED_IDEA_COLUMNS, ideas, "ideas"), (REQUIRED_RATING_COLUMNS, ratings, "ratings_long"),
                           (REQUIRED_TASK_COLUMNS, tasks, "tasks")):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing {missing}")
    if ideas.idea_id.duplicated().any():
        raise ValueError("ideas: duplicate idea_id")
    if not set(ideas.task_id) <= set(tasks.task_id):
        raise ValueError("ideas: task_id not in tasks")
    if len(ratings) and not set(ratings.idea_id) <= set(ideas.idea_id):
        raise ValueError("ratings: idea_id not in ideas")
    if len(ratings) and ((ratings.score < ratings.scale_min) | (ratings.score > ratings.scale_max)).any():
        raise ValueError("ratings: score outside scale")
    if len(ratings) and ratings.duplicated(["idea_id", "rater_id", "dimension"]).any():
        raise ValueError("ratings: duplicate (idea, rater, dimension)")
    if "advanced" in ideas and not set(pd.to_numeric(ideas.advanced, errors="coerce").dropna()) <= {0, 1}:
        raise ValueError("ideas: advanced must be 0 or 1")
    if "target_raters" in ideas:
        t = pd.to_numeric(ideas.target_raters, errors="coerce").dropna()
        if (t < 1).any() or (t != t.round()).any():
            raise ValueError("ideas: target_raters must be a positive whole number")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m benchmark.load")
    p.add_argument("data_dir", nargs="?", default=str(EXAMPLE_DIR))
    a = p.parse_args(argv)
    tables = read_folder(a.data_dir)
    validate(tables)
    for name, df in tables.items():
        print(f"{name}: {len(df):,} rows")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
