"""Field test: an AI judge scores an organization's ideas in shadow, next to its usual evaluators.

The organization screens submitted ideas as it always does; the AI judge's scores never reach the
evaluators before they decide. Afterwards this module answers two questions:

1. How many of the ideas the organization advanced would the AI judge have dropped? The AI screen
   advances the same number of ideas as the organization did, ranked by the judge's score. Ties
   at the cut-off are shared out evenly, so every idea gets a probability of being advanced by the
   AI screen and the count of dropped ideas is an expected count (``ai_would_drop``). Ideas below
   the cut-off are dropped for certain (``certainly_dropped``).
2. Are those the ideas the evaluators disagreed about? Among the ideas the organization
   advanced, the evaluators' per-idea spread (SD) is compared between ideas the AI screen would
   drop and ideas it would keep: mean spread in each group (weighted by the probability of being
   dropped) and the Spearman between that probability and the spread. How reliably the
   evaluators measured disagreement is reported alongside, from the diagnostic.

It also writes the sheet for the debrief: after the evaluators have decided, each idea with the
organization's decision, the evaluators' mean and spread, each AI judge's score and whether the
AI screen would have advanced it. The sheet is only written when every idea has a decision.

Inputs: a data folder whose ``ideas.csv`` has an ``advanced`` column (1 if the organization
advanced the idea, 0 if not), ``ratings_long.csv`` with the evaluators' own scores, and judge
scores from ``benchmark.judge`` (the baseline prompt is used).

    python -m benchmark.shadow --data-dir my_screen/                     # judge scores from results/judge_runs/my_screen/
    python -m benchmark.shadow --data-dir my_screen/ --scores s.csv --dimension novelty
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import agree


def _advanced(ideas: pd.DataFrame) -> pd.Series:
    if "advanced" not in ideas:
        raise ValueError("ideas.csv needs an 'advanced' column: 1 if the organization advanced the idea, 0 if not")
    adv = ideas.set_index("idea_id").advanced
    if adv.isna().any():
        raise ValueError("every idea needs a decision in 'advanced' before the shadow screen is read out")
    vals = set(pd.to_numeric(adv, errors="coerce").dropna().astype(int))
    if not vals <= {0, 1}:
        raise ValueError("'advanced' must be 0 or 1")
    return pd.to_numeric(adv).astype(int)


def screen_score(judge_scores: pd.DataFrame, dimensions: list[str] | None = None,
                 baseline: str = "P0_baseline") -> pd.DataFrame:
    """Per judge and idea: the judge's screening score, the mean of its baseline scores over the
    chosen dimensions (all dimensions if None). Adds the panel of all judges as judge 'all judges'."""
    js = judge_scores.dropna(subset=["score"])
    if "perturbation_id" in js:
        js = js[js.perturbation_id == baseline]
    if "sample_index" in js:
        js = js[js.sample_index == 0]
    if dimensions:
        js = js[js.dimension.isin(dimensions)]
    s = js.groupby(["judge", "idea_id"]).score.mean().rename("ai_score").reset_index()
    panel = s.groupby("idea_id").ai_score.mean().rename("ai_score").reset_index().assign(judge="all judges")
    return pd.concat([s, panel], ignore_index=True)


def ai_advance_probability(scores: pd.Series, k: int) -> pd.Series:
    """Probability that each idea is among the top ``k`` by score, with ties at the cut-off
    shared out evenly."""
    if k <= 0:
        return pd.Series(0.0, index=scores.index)
    if k >= len(scores):
        return pd.Series(1.0, index=scores.index)
    cut = scores.sort_values(ascending=False).iloc[k - 1]
    above = scores > cut
    tied = scores == cut
    slots = k - int(above.sum())
    p = pd.Series(0.0, index=scores.index)
    p[above] = 1.0
    p[tied] = slots / int(tied.sum())
    return p


def shadow_screen(ideas: pd.DataFrame, ratings: pd.DataFrame, judge_scores: pd.DataFrame,
                  dimensions: list[str] | None = None, reference_conditions: set[str] | None = None,
                  contested_quantile: float = 0.75) -> dict[str, pd.DataFrame]:
    """Both questions, per judge and for the panel of all judges, plus the debrief sheet."""
    adv = _advanced(ideas)
    k = int(adv.sum())
    pool = ratings if reference_conditions is None or "condition" not in ratings else ratings[ratings.condition.isin(reference_conditions)]
    if dimensions:
        pool = pool[pool.dimension.isin(dimensions)]
    ev = pool.groupby(["idea_id", "dimension"]).score.agg(["mean", "std", "count"]).reset_index()
    ev_idea = ev.groupby("idea_id").agg(evaluator_mean=("mean", "mean"), evaluator_sd=("std", "mean"),
                                        evaluators=("count", "max"))
    sc = screen_score(judge_scores, dimensions)
    sc = sc[sc.idea_id.isin(adv.index)]
    rel = agree.disagreement_reliability(pool, None) if len(pool) else pd.DataFrame()
    rel_by = agree._reliability_by_dimension(rel)
    summary, sheet = [], ideas.set_index("idea_id")[["idea_text"]].join(adv.rename("organization_advanced")).join(ev_idea)
    cut = sheet.evaluator_sd.quantile(contested_quantile) if sheet.evaluator_sd.notna().any() else np.nan
    sheet["evaluators_contested"] = sheet.evaluator_sd >= cut
    for judge, g in sc.groupby("judge", sort=False):
        s = g.set_index("idea_id").ai_score.reindex(adv.index)
        if s.isna().any():
            continue  # the judge did not score every idea
        p = ai_advance_probability(s, k)
        drop = (1 - p)[adv == 1]
        sd = sheet.evaluator_sd[adv == 1]
        row = {"judge": judge, "ideas": len(adv), "organization_advanced": k, "ai_would_drop": float(drop.sum()),
               "certainly_dropped": int((drop == 1).sum()), "share_dropped": float(drop.sum() / k) if k else np.nan}
        ok = sd.notna()
        if ok.sum() > 2:
            w_drop, w_keep = drop[ok], 1 - drop[ok]
            row["evaluator_sd_of_dropped"] = float((sd[ok] * w_drop).sum() / w_drop.sum()) if w_drop.sum() > 0 else np.nan
            row["evaluator_sd_of_kept"] = float((sd[ok] * w_keep).sum() / w_keep.sum()) if w_keep.sum() > 0 else np.nan
            if drop[ok].std() > 0 and sd[ok].std() > 0:
                rho = stats.spearmanr(drop[ok], sd[ok])
                row["spearman_drop_vs_evaluator_sd"], row["p"] = float(rho.statistic), float(rho.pvalue)
        summary.append(row)
        sheet[f"ai_score: {judge}"] = s
        sheet[f"ai_would_advance: {judge}"] = p.round(3)
    out = pd.DataFrame(summary)
    if len(rel_by):
        out = out.assign(evaluator_disagreement_reliability=float(rel_by.human_disagreement_reliability.mean()))
    return {"shadow_summary": out, "debrief_sheet": sheet.reset_index(), "evaluator_disagreement_reliability": rel}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from . import RESULTS_DIR
    from .load import read_folder

    p = argparse.ArgumentParser(prog="python -m benchmark.shadow",
                                description="Field test: what an AI judge would have dropped from an organization's screen.")
    p.add_argument("--data-dir", required=True, help="folder with ideas.csv (with 'advanced'), tasks.csv, ratings_long.csv")
    p.add_argument("--scores", default=None, help="judge_scores.csv (default: results/judge_runs/<folder>/judge_scores.csv)")
    p.add_argument("--dimension", action="append", default=None, help="screen on this dimension (repeatable; default all)")
    p.add_argument("--reference-condition", default=None, help="evaluators in this condition only")
    p.add_argument("--out", default=None, help="output folder (default: results/shadow/<folder>/)")
    a = p.parse_args(argv)
    d = Path(a.data_dir)
    t = read_folder(d)
    scores_path = Path(a.scores) if a.scores else RESULTS_DIR / "judge_runs" / d.name / "judge_scores.csv"
    res = shadow_screen(t["ideas"], t.get("ratings_long", pd.DataFrame(columns=["idea_id", "rater_id", "dimension", "score"])),
                        pd.read_csv(scores_path), a.dimension, {a.reference_condition} if a.reference_condition else None)
    out = Path(a.out) if a.out else RESULTS_DIR / "shadow" / d.name
    out.mkdir(parents=True, exist_ok=True)
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {out / (name + '.csv')}")
    print(res["shadow_summary"].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
