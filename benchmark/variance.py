"""Reintroducing variance into AI judgment: panels of AI judges that disagree where people disagree.

This is an early prototype of the pipeline. The aim is not more
disagreement among AI judges. A panel that disagrees about everything adds noise. The aim is
disagreement in the places where human raters disagree, so that a screen built on the panel can
see which ideas are contested.

The pipeline has five steps.

1. Calibrate against people. Per-idea human disagreement (the SD across raters) is the target.
   ``disagreement_reliability`` says how well a given number of raters pins it down (split-half
   Spearman of per-idea SD), and so how high any panel's correlation with it can go.
2. Build panels from different sources of disagreement. Each source turns judge scores into a
   per-idea spread:
     model_families   SD across different judges at the baseline prompt (no new calls)
     temperature      SD across one judge's repeated samples, averaged over judges (baseline)
     prompt_variants  SD across the stability grid within one judge (baseline)
     perspectives     SD across judges asked to rate as a customer, an engineer or an investor
     rater_twins      SD across judges each shown one human rater's own past ratings
     expected_spread  the SD of the rating distribution a judge forecasts for ten raters
   The first three use runs that already exist. ``plan_perspective_jobs``,
   ``plan_rater_twin_jobs`` and ``run_expected_spread`` make the calls for the last three through
   the same cached harness as every other judge call.
3. Search. ``search_panels`` scores every panel (every subset of judges within a source).
   ``cross_validated_search`` picks the best panel on training ideas and scores it on ideas it
   never saw, which is the number to report: choosing the best of many panels on the same ideas
   inflates the correlation.
4. Accept or reject on held-out ideas. ``score_spread`` reports the Spearman of panel spread with
   human spread, the same correlation corrected for the unreliability of human spread, the area
   under the curve for picking out the most contested ideas, and the panel's own ICC(A,1), which
   should sit near the human raters' own ICC and not near 1.
5. Deploy. ``panel_output`` returns, for every idea, a mean, a spread and a contested flag.

Nothing here is definitive; it is a demonstration and a starting point for other researchers.

    python -m benchmark.variance                                   # the example folder
    python -m benchmark.variance --ratings r.csv --scores s.csv    # your own data
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import agree

PERSPECTIVES = {
    "customer": "Rate the idea as a potential customer or user would see it.",
    "engineer": "Rate the idea as an engineer who would have to build or carry it out would see it.",
    "investor": "Rate the idea as an investor deciding whether to fund it would see it.",
}


# --- Step 1: the target and how well people measure it ------------------------------------

def human_disagreement(ratings: pd.DataFrame, conditions: set[str] | None = None) -> pd.DataFrame:
    """Per idea and dimension: n raters, mean and SD (the disagreement a panel should reproduce)."""
    return agree.reference(ratings, conditions)[["idea_id", "dimension", "n", "mean", "sd"]]


def disagreement_reliability(ratings: pd.DataFrame, conditions: set[str] | None = None,
                             ks: tuple[int, ...] = (5, 10, 15, 20), n_splits: int = 500, seed: int = 1) -> pd.DataFrame:
    """How well k raters pin down which ideas are contested: the mean Spearman, over random
    splits, between per-idea SD in two disjoint panels of k raters. Also the full pool's
    reliability by Spearman-Brown from the largest half split, and the ceiling it puts on any
    panel's correlation with human SD (its square root)."""
    pool = ratings if conditions is None or "condition" not in ratings else ratings[ratings.condition.isin(conditions)]
    rng = np.random.default_rng(seed)
    rows = []
    for dim, g in pool.groupby("dimension"):
        wide = g.pivot_table(index="rater_id", columns="idea_id", values="score")
        raters = wide.index.to_numpy()
        n = len(raters)
        for k in sorted(set(ks) | {n // 2}):
            if k < 2 or 2 * k > n:
                continue
            rs = []
            for _ in range(n_splits):
                perm = rng.permutation(raters)
                a, b = wide.loc[perm[:k]].std(), wide.loc[perm[k:2 * k]].std()
                ok = a.notna() & b.notna()
                if ok.sum() > 2:
                    rs.append(stats.spearmanr(a[ok], b[ok]).statistic)
            rows.append({"dimension": dim, "raters_per_panel": k, "split_half_spearman": float(np.nanmean(rs)),
                         "n_raters": n, "n_ideas": wide.shape[1]})
    out = pd.DataFrame(rows)
    if len(out):
        half = out.loc[out.groupby("dimension").raters_per_panel.idxmax()].set_index("dimension").split_half_spearman
        full = (2 * half / (1 + half)).clip(lower=0)
        out["full_pool_reliability"] = out.dimension.map(full)
        out["ceiling_for_panel_spearman"] = np.sqrt(out.full_pool_reliability)
    return out


# --- Step 2: sources of disagreement --------------------------------------------------------

def _first(js: pd.DataFrame) -> pd.DataFrame:
    return js[js.sample_index == 0] if "sample_index" in js else js


def source_scores(judge_scores: pd.DataFrame, source: str, baseline: str = "P0_baseline",
                  sampled: str = "P6_temperature_sampled") -> pd.DataFrame:
    """Long table (member, idea_id, dimension, score) for one source. A member is one voice on
    the panel: a judge, a judge's sample, a judge under one prompt variant, a perspective or a
    rater twin."""
    js = judge_scores.dropna(subset=["score"])
    if "perturbation_id" not in js:
        js = js.assign(perturbation_id=baseline)
    if source == "model_families":
        s = _first(js[js.perturbation_id == baseline])
        return s.groupby(["judge", "idea_id", "dimension"]).score.mean().reset_index().rename(columns={"judge": "member"})
    if source == "temperature":
        s = js[js.perturbation_id == sampled]
        return s.assign(member=s.judge + ":s" + s.sample_index.astype(str))[["member", "judge", "idea_id", "dimension", "score"]]
    if source == "prompt_variants":
        s = _first(js[~js.perturbation_id.isin([sampled]) & ~js.perturbation_id.str.startswith("V_")])
        s = s.groupby(["judge", "perturbation_id", "idea_id", "dimension"]).score.mean().reset_index()
        return s.assign(member=s.judge + ":" + s.perturbation_id)[["member", "judge", "idea_id", "dimension", "score"]]
    if source in ("perspectives", "rater_twins"):
        pid = {"perspectives": "V_perspective", "rater_twins": "V_rater_twin"}[source]
        s = _first(js[js.perturbation_id == pid])
        return s.assign(member=s.judge + ":" + s.variant.astype(str))[["member", "judge", "idea_id", "dimension", "score"]]
    raise ValueError(f"unknown source {source}")


def spread(members: pd.DataFrame) -> pd.Series:
    """Per idea and dimension: the SD across panel members."""
    return members.groupby(["idea_id", "dimension"]).score.std().rename("panel_sd")


def source_spread(judge_scores: pd.DataFrame, source: str) -> pd.Series:
    """Per-idea spread for a source. Within-judge sources (temperature, prompt variants) are
    averaged over judges so that differences between models do not leak in."""
    m = source_scores(judge_scores, source)
    if len(m) == 0:
        return pd.Series(dtype=float, name="panel_sd")
    if source in ("temperature", "prompt_variants"):
        return m.groupby(["judge", "idea_id", "dimension"]).score.std().groupby(["idea_id", "dimension"]).mean().rename("panel_sd")
    return spread(m)


# --- Step 4: scoring a panel against people -------------------------------------------------

def score_spread(panel_sd: pd.Series, human: pd.DataFrame, reliability: pd.DataFrame | None = None,
                 contested_quantile: float = 0.75, members: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per dimension: Spearman of panel SD with human SD (and p), the Spearman corrected for the
    unreliability of human SD, the AUC for picking out the top quarter of contested ideas, the
    mean panel and human SDs, and (given the members' scores) the panel's ICC(A,1)."""
    m = panel_sd.reset_index().merge(human, on=["idea_id", "dimension"]).dropna(subset=["panel_sd", "sd"])
    rows = []
    for dim, g in m.groupby("dimension"):
        if len(g) < 4 or g.panel_sd.std() == 0:
            rows.append({"dimension": dim, "n_ideas": len(g), "spearman": np.nan, "p": np.nan})
            continue
        rho = stats.spearmanr(g.panel_sd, g.sd)
        top = g.sd >= g.sd.quantile(contested_quantile)
        auc = np.nan
        if 0 < top.sum() < len(g):
            auc = float(stats.mannwhitneyu(g.panel_sd[top], g.panel_sd[~top]).statistic / (top.sum() * (~top).sum()))
        r = {"dimension": dim, "n_ideas": len(g), "spearman": float(rho.statistic), "p": float(rho.pvalue),
             "auc_contested": auc, "mean_panel_sd": float(g.panel_sd.mean()), "mean_human_sd": float(g.sd.mean())}
        if reliability is not None and len(reliability):
            rel = reliability[reliability.dimension == dim].full_pool_reliability
            if len(rel) and rel.iloc[0] > 0:
                r["spearman_disattenuated"] = float(min(1.0, rho.statistic / np.sqrt(rel.iloc[0])))
        if members is not None:
            sub = members[members.dimension == dim]
            if sub.member.nunique() > 1:
                r["panel_icc_a1"] = agree._icc_a1(sub, "member")
        rows.append(r)
    return pd.DataFrame(rows)


# --- Step 3: search -------------------------------------------------------------------------

def search_panels(judge_scores: pd.DataFrame, ratings: pd.DataFrame, conditions: set[str] | None = None,
                  sources: tuple[str, ...] = ("model_families",), min_size: int = 2,
                  holdout: list[str] | None = None, dimension: str = "novelty") -> pd.DataFrame:
    """Score every panel: every subset (of at least ``min_size``) of the judges within each
    source. With ``holdout`` idea ids, panels are ranked on the other ideas and reported on the
    held-out ones, so the winner is not chosen on the ideas it is judged on. With few ideas (the
    example has 15) the ranking is illustrative only."""
    human = human_disagreement(ratings, conditions)
    rows = []
    for src in sources:
        m = source_scores(judge_scores, src)
        if len(m) == 0:
            continue
        if "judge" not in m:
            m = m.assign(judge=m.member)
        judges = sorted(m.judge.unique())
        for size in range(min_size, len(judges) + 1):
            for combo in itertools.combinations(judges, size):
                sub = m[m.judge.isin(combo) & (m.dimension == dimension)]
                sd = spread(sub) if src not in ("temperature", "prompt_variants") else \
                    sub.groupby(["judge", "idea_id", "dimension"]).score.std().groupby(["idea_id", "dimension"]).mean().rename("panel_sd")
                for part, ids in (("calibration", None), ("held_out", holdout)):
                    if part == "held_out" and not holdout:
                        continue
                    s = sd
                    if holdout:
                        idx = s.index.get_level_values("idea_id")
                        s = s[idx.isin(holdout)] if part == "held_out" else s[~idx.isin(holdout)]
                    sc = score_spread(s, human)
                    if len(sc):
                        rows.append({"source": src, "panel": "+".join(combo), "size": size, "part": part,
                                     **sc.iloc[0].drop("dimension").to_dict()})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["part", "spearman"], ascending=[True, False], ignore_index=True)
    return out


def cross_validated_search(judge_scores: pd.DataFrame, ratings: pd.DataFrame, conditions: set[str] | None = None,
                           source: str = "model_families", dimension: str = "novelty", test_share: float = 1 / 3,
                           n_splits: int = 200, seed: int = 1) -> pd.DataFrame:
    """Honest version of the search: on each random split of ideas, pick the panel with the best
    Spearman on the training ideas and score it on the test ideas it never saw. Returns one row
    per split (chosen panel, its training and test Spearman, and the full panel's test Spearman
    for comparison). Picking the best of many panels on the same ideas it is scored on inflates
    the result; this does not."""
    human = human_disagreement(ratings, conditions)
    m = source_scores(judge_scores, source)
    m = m[m.dimension == dimension]
    if "judge" not in m:
        m = m.assign(judge=m.member)
    judges = sorted(m.judge.unique())
    panels = [c for k in range(2, len(judges) + 1) for c in itertools.combinations(judges, k)]
    sds = {c: spread(m[m.judge.isin(c)]).reset_index().merge(human, on=["idea_id", "dimension"]).set_index("idea_id")
           for c in panels}
    ideas = np.array(sorted(human[human.dimension == dimension].idea_id))
    n_test = max(4, int(round(len(ideas) * test_share)))
    rng = np.random.default_rng(seed)

    def rho(df):
        return stats.spearmanr(df.panel_sd, df.sd).statistic if df.panel_sd.std() > 0 else np.nan

    rows = []
    for k in range(n_splits):
        test = rng.choice(ideas, n_test, replace=False)
        train = np.setdiff1d(ideas, test)
        train_rho = {c: rho(sds[c].loc[sds[c].index.isin(train)]) for c in panels}
        best = max(panels, key=lambda c: -np.inf if np.isnan(train_rho[c]) else train_rho[c])
        rows.append({"split": k, "chosen_panel": "+".join(best), "train_spearman": train_rho[best],
                     "test_spearman": rho(sds[best].loc[sds[best].index.isin(test)]),
                     "all_judges_test_spearman": rho(sds[tuple(judges)].loc[sds[tuple(judges)].index.isin(test)])})
    return pd.DataFrame(rows)


# --- Step 5: what a screen gets --------------------------------------------------------------

def panel_output(members: pd.DataFrame, contested_quantile: float = 0.75) -> pd.DataFrame:
    """Per idea and dimension: the panel's mean, its spread, and a contested flag (spread in the
    top quarter). A screen built on this can route contested ideas to people instead of
    discarding them."""
    g = members.groupby(["idea_id", "dimension"]).score.agg(panel_mean="mean", panel_sd="std").reset_index()
    cut = g.groupby("dimension").panel_sd.transform(lambda s: s.quantile(contested_quantile))
    g["contested"] = g.panel_sd >= cut
    return g


# --- Calls for the new sources (through the cached judge harness) ---------------------------

def plan_perspective_jobs(ideas: pd.DataFrame, tasks: pd.DataFrame, rubric,
                          perspectives: dict[str, str] | None = None) -> list[dict]:
    """One job per idea and perspective: the shared rubric with one added sentence asking the
    judge to rate from that perspective."""
    from .perturb import rubric_values

    perspectives = perspectives or PERSPECTIVES
    task_text = tasks.set_index("task_id").task_prompt_verbatim
    jobs = []
    base = rubric_values(rubric)
    for name, line in perspectives.items():
        vals = {**base, "instructions": base["instructions"] + "\n" + line}
        for row in ideas.itertuples():
            jobs.append({"perturbation_id": "V_perspective", "variant": name, "seed": None, "sample_index": 0,
                         "temperature": 0.0, "kind": "single", "idea_ids": [row.idea_id],
                         "values": {**vals, "task_prompt": task_text[row.task_id], "idea_text": row.idea_text,
                                    "few_shot_block": ""}})
    return jobs


def plan_rater_twin_jobs(ideas: pd.DataFrame, tasks: pd.DataFrame, rubric, ratings: pd.DataFrame,
                         train_ideas: list[str], raters: list[str] | None = None) -> list[dict]:
    """One job per (rater, idea not in ``train_ideas``): the judge sees that rater's own scores on
    the training ideas as worked examples and rates the new idea as that rater would. Rater data
    stays in the local cache; no rater identity is sent beyond a pseudonymous id."""
    from .perturb import rubric_values

    dims = [d.id for d in rubric.dimensions]
    task_text = tasks.set_index("task_id").task_prompt_verbatim
    text = ideas.set_index("idea_id").idea_text
    vals = rubric_values(rubric)
    raters = raters or sorted(ratings.rater_id.unique())
    test = ideas[~ideas.idea_id.isin(train_ideas)]
    jobs = []
    for r in raters:
        mine = ratings[(ratings.rater_id == r) & ratings.idea_id.isin(train_ideas)]
        wide = mine.pivot_table(index="idea_id", columns="dimension", values="score")
        if wide.empty:
            continue
        lines = ["Here are the scores one particular rater gave to other ideas. Rate the new idea as this same rater would:"]
        for i in wide.index:
            sc = ", ".join(f'"{d}": {int(wide.loc[i, d])}' for d in dims if d in wide and pd.notna(wide.loc[i, d]))
            lines.append(f'- "{text[i]}" -> {{{sc}}}')
        block = "\n".join(lines) + "\n"
        for row in test.itertuples():
            jobs.append({"perturbation_id": "V_rater_twin", "variant": str(r), "seed": None, "sample_index": 0,
                         "temperature": 0.0, "kind": "single", "idea_ids": [row.idea_id],
                         "values": {**vals, "task_prompt": task_text[row.task_id], "idea_text": row.idea_text,
                                    "few_shot_block": block}})
    return jobs


def parse_expected_spread(text: str, dims: list[str], scale: tuple[int, int]) -> dict[str, tuple[float, float]] | None:
    """{"novelty": {"1": 2, ..., "5": 1}, ...} -> {dim: (mean, sd)}; None if unusable."""
    import json
    import re

    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out = {}
    points = np.arange(scale[0], scale[1] + 1)
    for d in dims:
        counts = obj.get(d)
        if not isinstance(counts, dict):
            return None
        c = np.array([float(counts.get(str(p), counts.get(p, 0)) or 0) for p in points])
        if c.sum() <= 0:
            return None
        w = c / c.sum()
        mean = float((w * points).sum())
        out[d] = (mean, float(np.sqrt((w * (points - mean) ** 2).sum())))
    return out


def run_expected_spread(ideas: pd.DataFrame, tasks: pd.DataFrame, rubric, judge: dict, backend, cache=None) -> pd.DataFrame:
    """Ask a judge for the rating distribution of ten raters, per idea. Returns judge, idea_id,
    dimension, forecast_mean, forecast_sd. Cached like every judge call."""
    from .judge import DiskCache, cache_key, load_prompt, render_prompt
    from .perturb import rubric_values

    spec = load_prompt("expected_spread_v1.0")
    dims = [d.id for d in rubric.dimensions]
    scale = (rubric.scale_min, rubric.scale_max)
    pts = ", ".join(f'"{p}": <count>' for p in range(scale[0], scale[1] + 1))
    base = {**rubric_values(rubric), "json_format": "{" + ", ".join(f'"{d}": {{{pts}}}' for d in dims) + "}"}
    task_text = tasks.set_index("task_id").task_prompt_verbatim
    cache = cache or DiskCache()
    rows = []
    for row in ideas.itertuples():
        vals = {**base, "task_prompt": task_text[row.task_id], "idea_text": row.idea_text}
        system, user = render_prompt(spec, **{k: v for k, v in vals.items() if k in spec.variables})
        key = cache_key(getattr(backend, "name", "openrouter"), judge["model"], spec.name, "V_expected_spread", system, user, 0.0, 0)
        rec = cache.get(spec.name, key)
        if rec is None:
            rec = {"judge": judge["label"], "model": judge["model"], "prompt": spec.name, "idea_ids": [row.idea_id],
                   "system": system, "user": user, **backend.complete(system, user, temperature=0.0, max_tokens=800)}
            cache.put(spec.name, key, rec)
        parsed = parse_expected_spread(rec["text"], dims, scale)
        for d in dims:
            mean, sd = parsed[d] if parsed else (np.nan, np.nan)
            rows.append({"judge": judge["label"], "idea_id": row.idea_id, "dimension": d,
                         "forecast_mean": mean, "forecast_sd": sd})
    return pd.DataFrame(rows)


# --- One call on existing runs ----------------------------------------------------------------

def evaluate(ratings: pd.DataFrame, judge_scores: pd.DataFrame, conditions: set[str] | None = None,
             expected: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Every source the scores support, scored against human disagreement, plus the reliability
    of human disagreement and the panel search over model families."""
    human = human_disagreement(ratings, conditions)
    rel = disagreement_reliability(ratings, conditions)
    rows = []
    for src in ("model_families", "temperature", "prompt_variants", "perspectives", "rater_twins"):
        try:
            m = source_scores(judge_scores, src)
        except (KeyError, AttributeError):
            continue
        if len(m) == 0:
            continue
        sc = score_spread(source_spread(judge_scores, src), human, rel,
                          members=m if src in ("model_families", "perspectives", "rater_twins") else None)
        rows += [{"source": src, **r} for r in sc.to_dict("records")]
    if expected is not None and len(expected):
        sd = expected.groupby(["idea_id", "dimension"]).forecast_sd.mean().rename("panel_sd")
        rows += [{"source": "expected_spread", **r} for r in score_spread(sd, human, rel).to_dict("records")]
    cv = cross_validated_search(judge_scores, ratings, conditions)
    summary = pd.DataFrame([{"source": "model_families", "dimension": "novelty", "splits": len(cv),
                             "mean_train_spearman_of_chosen_panel": cv.train_spearman.mean(),
                             "mean_test_spearman_of_chosen_panel": cv.test_spearman.mean(),
                             "mean_test_spearman_all_judges": cv.all_judges_test_spearman.mean(),
                             "most_chosen_panel": cv.chosen_panel.mode().iloc[0]}]) if len(cv) else pd.DataFrame()
    return {"variance_sources": pd.DataFrame(rows), "disagreement_reliability": rel,
            "panel_search": search_panels(judge_scores, ratings, conditions),
            "panel_search_cross_validated": summary}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from . import RESULTS_DIR

    p = argparse.ArgumentParser(prog="python -m benchmark.variance")
    p.add_argument("--ratings", default=None, help="ratings_long.csv (default: the example)")
    p.add_argument("--scores", default=None, help="judge_scores.csv (default: the example)")
    p.add_argument("--reference-condition", default=None, help="raters in this condition form the reference")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    from .load import EXAMPLE_DIR

    ratings = pd.read_csv(a.ratings or EXAMPLE_DIR / "ratings_long.csv")
    conds = {a.reference_condition} if a.reference_condition else None
    scores = pd.read_csv(a.scores or EXAMPLE_DIR / "judge_scores.csv")
    out = Path(a.out) if a.out else RESULTS_DIR / "variance"
    out.mkdir(parents=True, exist_ok=True)
    res = evaluate(ratings, scores, conds)
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {out / (name + '.csv')}")
    print(res["variance_sources"].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
