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
   The first three use the stability runs that already exist. ``run_sources`` makes the calls
   for the last three through the same cached harness as every other judge call
   (``python -m benchmark.variance run``).
3. Search. ``candidate_spreads`` lists every way the runs can produce a per-idea spread: every
   panel of two or more different judges, each judge's own spread within a source, and each
   judge's forecast of how people would split. A single judge can pass: it may rate consistently
   and still flag the right ideas as contested, for example by predicting the spread of scores.
4. Judge by contested ideas the pipeline has not seen. ``cross_validated_search`` chooses the
   best candidate on some ideas and tests it on the others, and reports how well the chosen
   candidate picks out the contested ideas (the top quarter of human disagreement) among ideas it
   was not chosen on: the area under the curve, with the Spearman of its spread with human spread
   alongside. Choosing the best of many candidates on the same ideas it is scored on would inflate
   both. ``score_spread`` reports the same measures for a fixed source, the Spearman corrected
   for the unreliability of human disagreement, and the panel's own ICC(A,1), which should sit
   near the human raters' ICC and not near 1.
5. Deploy. ``panel_output`` returns, for every idea, a mean, a spread and a contested flag.

Nothing here is definitive; it is a demonstration and a starting point for other researchers.

    python -m benchmark.variance                                   # evaluate the example folder
    python -m benchmark.variance --ratings r.csv --scores s.csv    # evaluate your own data
    python -m benchmark.variance run --data-dir my_study/ --judges qwen3.8-27b,deepseek-v4-pro
                                   [--sources perspectives,rater_twins,expected_spread] [--backend fake]
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


# The reliability of human disagreement lives in the diagnostic, so that every diagnostic reports
# it; the pipeline uses the same function.
disagreement_reliability = agree.disagreement_reliability


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
            rel = reliability[(reliability.dimension == dim) & reliability.raters_per_panel.isna()].full_pool_reliability
            if len(rel) and rel.iloc[0] > 0:
                r["spearman_disattenuated"] = float(np.clip(rho.statistic / np.sqrt(rel.iloc[0]), -1.0, 1.0))
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


def candidate_spreads(judge_scores: pd.DataFrame, expected: pd.DataFrame | None = None,
                      dimension: str = "novelty", min_panel: int = 2) -> dict[str, pd.Series]:
    """Every per-idea spread the runs support, keyed by a readable name. The spreads use no human
    data, so they can be computed on all ideas before any split. Panels of different judges need
    at least ``min_panel`` judges; within-judge sources and forecasts are offered judge by judge
    and averaged over all judges."""
    out: dict[str, pd.Series] = {}

    def keep(name, sd):
        sd = sd.xs(dimension, level="dimension") if isinstance(sd.index, pd.MultiIndex) else sd
        if len(sd.dropna()) > 3 and sd.std() > 0:
            out[name] = sd.rename("panel_sd")

    fam = source_scores(judge_scores, "model_families")
    fam = fam[fam.dimension == dimension]
    judges = sorted(fam.member.unique())
    for k in range(min_panel, len(judges) + 1):
        for combo in itertools.combinations(judges, k):
            keep("model families: " + " + ".join(combo), spread(fam[fam.member.isin(combo)]))
    for src in ("temperature", "prompt_variants", "perspectives", "rater_twins"):
        m = source_scores(judge_scores, src)
        m = m[m.dimension == dimension]
        if not len(m):
            continue
        per = m.groupby(["judge", "idea_id"]).score.std()
        for j in sorted(m.judge.unique()):
            keep(f"{src}: {j}", per.xs(j, level="judge"))
        if m.judge.nunique() > 1:
            keep(f"{src}: mean over judges", per.groupby("idea_id").mean())
    if expected is not None and len(expected):
        e = expected[expected.dimension == dimension]
        per = e.set_index(["judge", "idea_id"]).forecast_sd
        for j in sorted(e.judge.unique()):
            keep(f"expected spread: {j}", per.xs(j, level="judge"))
        if e.judge.nunique() > 1:
            keep("expected spread: mean over judges", per.groupby("idea_id").mean())
    return out


def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    if positive.sum() == 0 or positive.sum() == len(positive):
        return np.nan
    return float(stats.mannwhitneyu(scores[positive], scores[~positive]).statistic / (positive.sum() * (~positive).sum()))


def cross_validated_search(judge_scores: pd.DataFrame, ratings: pd.DataFrame, conditions: set[str] | None = None,
                           expected: pd.DataFrame | None = None, dimension: str = "novelty",
                           test_share: float = 1 / 3, n_splits: int = 200, seed: int = 1,
                           contested_quantile: float = 0.75) -> pd.DataFrame:
    """The pipeline's test: how well it picks out the contested ideas among ideas it has not seen.

    On each random split of ideas, the candidate (``candidate_spreads``) that best picks out the
    contested ideas among the training ideas (AUC; Spearman breaks ties) is chosen, then tested on
    the other ideas. Contested ideas are the top quarter of human disagreement (per-idea SD over
    raters). One row per split: the chosen candidate, its training and test AUC and Spearman, and
    the same test measures for the panel of all judges as a fixed comparison."""
    human = human_disagreement(ratings, conditions)
    h = human[human.dimension == dimension].set_index("idea_id").sd.dropna()
    cands = {n: sd.reindex(h.index) for n, sd in candidate_spreads(judge_scores, expected, dimension).items()}
    cands = {n: sd for n, sd in cands.items() if sd.notna().sum() > 3}
    if not cands or len(h) < 6:
        return pd.DataFrame()
    fam = [n for n in cands if n.startswith("model families:")]
    full = max(fam, key=lambda n: n.count("+")) if fam else None
    ideas = h.index.to_numpy()
    n_test = max(3, int(round(len(ideas) * test_share)))
    cut = h.quantile(contested_quantile)
    rng = np.random.default_rng(seed)

    def measure(sd, ids):
        x, y = sd.loc[ids], h.loc[ids]
        ok = x.notna().to_numpy()
        x, y = x.to_numpy()[ok], y.to_numpy()[ok]
        if len(x) < 3 or np.std(x) == 0:
            return np.nan, np.nan
        return _auc(x, y >= cut), float(stats.spearmanr(x, y).statistic)

    rows = []
    for k in range(n_splits):
        test = rng.choice(ideas, n_test, replace=False)
        train = np.setdiff1d(ideas, test)
        scored = {n: measure(sd, train) for n, sd in cands.items()}
        best = max(scored, key=lambda n: tuple(-np.inf if np.isnan(v) else v for v in scored[n]))
        t_auc, t_rho = measure(cands[best], test)
        f_auc, f_rho = measure(cands[full], test) if full else (np.nan, np.nan)
        rows.append({"split": k, "chosen": best, "train_auc": scored[best][0], "train_spearman": scored[best][1],
                     "test_auc": t_auc, "test_spearman": t_rho, "all_judges_test_auc": f_auc,
                     "all_judges_test_spearman": f_rho})
    return pd.DataFrame(rows)


def summarise_search(cv: pd.DataFrame, dimension: str = "novelty") -> pd.DataFrame:
    """One row: the pipeline's result on unseen ideas (mean over splits)."""
    if not len(cv):
        return pd.DataFrame()
    return pd.DataFrame([{"dimension": dimension, "splits": len(cv),
                          "test_auc_contested": cv.test_auc.mean(), "test_spearman": cv.test_spearman.mean(),
                          "train_auc_contested": cv.train_auc.mean(),
                          "all_judges_test_auc_contested": cv.all_judges_test_auc.mean(),
                          "all_judges_test_spearman": cv.all_judges_test_spearman.mean(),
                          "most_often_chosen": cv.chosen.mode().iloc[0],
                          "share_chosen_most_often": float((cv.chosen == cv.chosen.mode().iloc[0]).mean())}])


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


def run_sources(data_dir, judges: list[dict], *, rubric_path, sources=("perspectives", "rater_twins", "expected_spread"),
                n_twins: int = 10, seed: int = 1, out_dir=None, backend_factory=None, cache=None,
                budget_usd: float = 50.0) -> dict[str, pd.DataFrame]:
    """Make the calls for the sources that need them, through the cached judge harness.

    perspectives     every idea, every judge, each perspective in ``PERSPECTIVES``
    rater_twins      ``n_twins`` human raters (chosen at random with ``seed``). Each twin sees that
                     rater's own scores on the training ideas (the ideas whose ``split`` is
                     ``calibration``; without a split column, a random half) and rates the others.
    expected_spread  every idea, every judge: the judge forecasts how ten raters would split.

    Writes ``judge_scores.csv`` (perspectives and twins, in the harness format) and
    ``expected_spread.csv`` to ``out_dir`` (default ``results/variance_runs/<folder name>/``)."""
    from . import RESULTS_DIR
    from .judge import DiskCache, OpenRouterBackend, load_rubric, run
    from .load import read_folder

    data_dir = Path(data_dir)
    t = read_folder(data_dir)
    ideas, tasks = t["ideas"], t["tasks"]
    rubric, _ = load_rubric(Path(rubric_path))
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR / "variance_runs" / data_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = cache or DiskCache()
    jobs = []
    if "perspectives" in sources:
        jobs += plan_perspective_jobs(ideas, tasks, rubric)
    if "rater_twins" in sources:
        ratings = t.get("ratings_long")
        if ratings is None or not len(ratings):
            raise ValueError("rater_twins needs ratings_long.csv in the data folder")
        rng = np.random.default_rng(seed)
        if "split" in ideas and (ideas.split == "calibration").any():
            train = list(ideas.idea_id[ideas.split == "calibration"])
        else:
            train = list(rng.choice(ideas.idea_id.to_numpy(), len(ideas) // 2, replace=False))
        pool = sorted(ratings.rater_id.astype(str).unique())
        twins = list(rng.choice(pool, min(n_twins, len(pool)), replace=False))
        jobs += plan_rater_twin_jobs(ideas, tasks, rubric, ratings.assign(rater_id=ratings.rater_id.astype(str)), train, twins)
    res = {}
    if jobs:
        res["judge_scores"] = run(str(data_dir), judges, rubric_path=Path(rubric_path), jobs=jobs, out_dir=out_dir,
                                  backend_factory=backend_factory, cache=cache, budget_usd=budget_usd)
    if "expected_spread" in sources:
        frames = []
        for j in judges:
            backend = backend_factory(j) if backend_factory else OpenRouterBackend(j["model"], reasoning_effort=j.get("reasoning_effort"))
            frames.append(run_expected_spread(ideas, tasks, rubric, j, backend, cache))
        res["expected_spread"] = pd.concat(frames, ignore_index=True)
        res["expected_spread"].to_csv(out_dir / "expected_spread.csv", index=False)
    return res


# --- One call on existing runs ----------------------------------------------------------------

def evaluate(ratings: pd.DataFrame, judge_scores: pd.DataFrame, conditions: set[str] | None = None,
             expected: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Every source the scores support, scored against human disagreement; the reliability of
    human disagreement; the in-sample panel search; and the pipeline's test on unseen ideas
    (``pipeline_test``: how well the chosen candidate picks out the contested ideas)."""
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
    dims = sorted(human.dimension.unique())
    summary = pd.concat([summarise_search(cross_validated_search(judge_scores, ratings, conditions, expected, d), d)
                         for d in dims], ignore_index=True)
    return {"variance_sources": pd.DataFrame(rows), "disagreement_reliability": rel,
            "panel_search": search_panels(judge_scores, ratings, conditions),
            "pipeline_test": summary}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    from . import REPO_DIR, RESULTS_DIR
    from .load import EXAMPLE_DIR

    p = argparse.ArgumentParser(prog="python -m benchmark.variance",
                                description="The variance pipeline: evaluate sources of disagreement, or run the sources that need new calls.")
    p.add_argument("command", nargs="?", choices=["evaluate", "run"], default="evaluate")
    p.add_argument("--data-dir", default=None, help="data folder (default: the example). evaluate reads its ratings, its judge "
                   "scores (from the folder or results/judge_runs/<folder>/) and the runs made by `run` for it")
    p.add_argument("--ratings", default=None, help="ratings_long.csv (default: the data folder's)")
    p.add_argument("--scores", default=None, help="judge_scores.csv (default: the data folder's, or results/judge_runs/<folder>/)")
    p.add_argument("--variance-runs", default=None, help="folder written by `run` (default: results/variance_runs/<folder>/ if present)")
    p.add_argument("--reference-condition", default=None, help="raters in this condition form the reference")
    p.add_argument("--judges", default=None, help="run: comma-separated labels from configs/judges.yaml")
    p.add_argument("--sources", default="perspectives,rater_twins,expected_spread", help="run: which sources to call")
    p.add_argument("--twins", type=int, default=10, help="run: number of human raters to build twins of")
    p.add_argument("--rubric", default=None, help="run: rubric YAML (default: the one in configs/judges.yaml)")
    p.add_argument("--backend", choices=["openrouter", "fake"], default="openrouter")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    data_dir = Path(a.data_dir) if a.data_dir else EXAMPLE_DIR
    if a.command == "run":
        from .judge import DiskCache, FakeBackend, load_env, load_judges

        load_env()
        cfg = load_judges()
        judges = [j for j in cfg["judges"] if not a.judges or j["label"] in a.judges.split(",")]
        fake = a.backend == "fake"
        res = run_sources(data_dir, judges, rubric_path=Path(a.rubric) if a.rubric else REPO_DIR / cfg["rubric"],
                          sources=tuple(a.sources.split(",")), n_twins=a.twins, out_dir=a.out,
                          backend_factory=(lambda j: FakeBackend(j["label"])) if fake else None,
                          cache=DiskCache(RESULTS_DIR / "cache_fake") if fake else None,
                          budget_usd=float(os.environ.get("JUDGE_BUDGET_USD", cfg.get("budget_usd", 50))))
        for name, df in res.items():
            print(f"{name}: {len(df):,} rows")
        return 0
    ratings = pd.read_csv(a.ratings or data_dir / "ratings_long.csv")
    conds = {a.reference_condition} if a.reference_condition else None
    scores_path = Path(a.scores) if a.scores else (data_dir / "judge_scores.csv" if (data_dir / "judge_scores.csv").exists()
                                                    else RESULTS_DIR / "judge_runs" / data_dir.name / "judge_scores.csv")
    scores = pd.read_csv(scores_path)
    # The runs made by `run` belong to a data folder: read them only for the folder named (or the
    # example when nothing is named), never alongside ratings or scores given as separate files.
    if a.variance_runs:
        runs = Path(a.variance_runs)
    elif a.data_dir or not (a.ratings or a.scores):
        runs = RESULTS_DIR / "variance_runs" / data_dir.name
    else:
        runs = Path("/nonexistent")
    expected = None
    if (runs / "judge_scores.csv").exists():
        scores = pd.concat([scores, pd.read_csv(runs / "judge_scores.csv")], ignore_index=True)
    if (runs / "expected_spread.csv").exists():
        expected = pd.read_csv(runs / "expected_spread.csv")
    out = Path(a.out) if a.out else RESULTS_DIR / "variance" / data_dir.name
    out.mkdir(parents=True, exist_ok=True)
    res = evaluate(ratings, scores, conds, expected)
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {out / (name + '.csv')}")
    print(res["variance_sources"].round(2).to_string(index=False))
    if len(res["pipeline_test"]):
        print("\nThe pipeline on ideas it has not seen (picking out the top quarter of contested ideas):")
        print(res["pipeline_test"].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
