"""Agreement, rater-quality and stability statistics. Humans and AI judges go through the same tests.

Inputs are long tables in the README format: ``ratings`` with idea_id, rater_id, dimension, score
(and optional condition). An AI judge enters as one more rater (``judge_as_rater``).

Reference
    ``reference``: per idea and dimension, the mean, spread and standard error of a chosen pool of
    raters (for example baseline raters only). The reference is an aggregate, never one answer.

Human ceiling and how many raters are enough
    ``human_ceiling``: ICC(A,1) (one rater) and ICC(A,k) (mean of k raters), two-way random,
    absolute agreement, with 95% CIs, per dimension and condition.
    ``raters_needed``: Spearman-Brown, the number of raters whose mean reaches a target reliability.

Contested ideas
    ``idea_summary``: n, mean, SD (contestedness) and SE per idea and dimension.
    ``contested_detection``: does a judge's variability across repeated samples track human
    disagreement (Spearman between judge SD and human SD over ideas)?

Rater report (the same for a person and a model)
    ``rater_report``: per rater and dimension, leniency (mean difference from a leave-one-out
    reference), noise (SD of the residual after removing leniency), and agreement (Spearman with
    the reference). The novelty-minus-usefulness leniency gap is reported because a bias can move
    one dimension and not the other.

Stability
    ``drift``: per perturbation, mean absolute change from the baseline prompt and the change in
    agreement with the reference.

The AI judge diagnostic (the instrument's fourth module)
    ``agreement_ladder``: ICC(A,1) for person with person, AI judge with AI judge, and each AI
    judge with itself across repeated samples (Figure 1, panel A).
    ``judge_diagnostics``: one row per judge and dimension on the four measures: agreement,
    leniency, stability and variance.
    ``diagnose``: everything above for any rated dataset and any set of judge scores, so a lab can
    run the diagnostic on its own ideas, rubric and raters (``python -m benchmark.agree --diagnose``).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

DIMENSIONS = ("novelty", "usefulness")


# --- Reference and per-idea summaries ------------------------------------------------------

def reference(ratings: pd.DataFrame, conditions: set[str] | None = None,
              exclude_rater: str | None = None) -> pd.DataFrame:
    """Per idea x dimension: n, mean, sd, se of the chosen rater pool."""
    pool = ratings
    if conditions is not None and "condition" in pool:
        pool = pool[pool.condition.isin(conditions)]
    if exclude_rater is not None:
        pool = pool[pool.rater_id != exclude_rater]
    g = pool.groupby(["idea_id", "dimension"]).score
    out = g.agg(n="count", mean="mean", sd="std").reset_index()
    out["se"] = out.sd / np.sqrt(out.n)
    return out


def idea_summary(ratings: pd.DataFrame, conditions: set[str] | None = None) -> pd.DataFrame:
    """Reference with ideas ranked by contestedness (SD) within each dimension."""
    ref = reference(ratings, conditions)
    ref["contested_rank"] = ref.groupby("dimension").sd.rank(ascending=False, method="min").astype(int)
    return ref.sort_values(["dimension", "contested_rank"], ignore_index=True)


# --- Reliability ---------------------------------------------------------------------------

def _icc(sub: pd.DataFrame) -> dict:
    import pingouin as pg

    try:
        t = pg.intraclass_corr(data=sub, targets="idea_id", raters="rater_id", ratings="score", nan_policy="omit")
    except (AssertionError, ValueError):  # too few ideas or raters to estimate
        nan = float("nan")
        return {"icc_a1": nan, "icc_a1_ci": (nan, nan), "icc_ak": nan, "icc_ak_ci": (nan, nan)}
    t = t.set_index("Type")
    return {"icc_a1": float(t.loc["ICC(A,1)", "ICC"]), "icc_a1_ci": tuple(t.loc["ICC(A,1)", "CI95"]),
            "icc_ak": float(t.loc["ICC(A,k)", "ICC"]), "icc_ak_ci": tuple(t.loc["ICC(A,k)", "CI95"])}


def human_ceiling(ratings: pd.DataFrame, by_condition: bool = True) -> pd.DataFrame:
    """ICC(A,1) and ICC(A,k) per dimension (and condition), with rater and idea counts."""
    rows = []
    groups = ratings.groupby(["dimension", "condition"]) if by_condition and "condition" in ratings else ratings.groupby(["dimension"])
    for key, sub in groups:
        key = key if isinstance(key, tuple) else (key,)
        r = _icc(sub)
        rows.append({"dimension": key[0], "condition": key[1] if len(key) > 1 else "all",
                     "n_raters": sub.rater_id.nunique(), "n_ideas": sub.idea_id.nunique(), **r,
                     "raters_needed_0.7": raters_needed(r["icc_a1"], 0.7),
                     "raters_needed_0.8": raters_needed(r["icc_a1"], 0.8)})
    return pd.DataFrame(rows)


def raters_needed(icc_single: float, target: float = 0.8) -> int | None:
    """Spearman-Brown: smallest k with k*r / (1 + (k-1)*r) >= target."""
    if not icc_single > 0:  # also catches NaN
        return None
    return math.ceil(target * (1 - icc_single) / (icc_single * (1 - target)) - 1e-9)


def spearman_brown(icc_single: float, k: int) -> float:
    return k * icc_single / (1 + (k - 1) * icc_single)


# --- Rater report --------------------------------------------------------------------------

def rater_report(ratings: pd.DataFrame, reference_conditions: set[str] | None = None) -> pd.DataFrame:
    """Leniency, noise and agreement for every rater (human or judge) against a leave-one-out
    reference built from ``reference_conditions`` (all raters if None)."""
    pool = ratings if reference_conditions is None else ratings[ratings.condition.isin(reference_conditions)]
    sums = pool.groupby(["idea_id", "dimension"]).score.agg(["sum", "count"])
    rows = []
    for (rater, dim), sub in ratings.groupby(["rater_id", "dimension"]):
        s = sums.reindex(pd.MultiIndex.from_arrays([sub.idea_id, sub.dimension]))
        in_pool = rater in set(pool.rater_id)
        tot, cnt = s["sum"].to_numpy(float), s["count"].to_numpy(float)
        if in_pool:
            tot, cnt = tot - sub.score.to_numpy(float), cnt - 1
        ref = np.divide(tot, cnt, out=np.full_like(tot, np.nan), where=cnt > 0)
        diff = sub.score.to_numpy(float) - ref
        ok = ~np.isnan(diff)
        lean = float(np.mean(diff[ok])) if ok.any() else np.nan
        noise = float(np.std(diff[ok] - lean, ddof=1)) if ok.sum() > 1 else np.nan
        rho = float(stats.spearmanr(sub.score.to_numpy(float)[ok], ref[ok]).statistic) if ok.sum() > 2 and np.std(sub.score.to_numpy(float)[ok]) > 0 else np.nan
        rows.append({"rater_id": rater, "dimension": dim, "n_ideas": int(ok.sum()), "leniency": lean,
                     "noise": noise, "agreement_spearman": rho,
                     "condition": sub.condition.iloc[0] if "condition" in sub else None})
    rep = pd.DataFrame(rows)
    wide = rep.pivot_table(index="rater_id", columns="dimension", values="leniency")
    if set(DIMENSIONS) <= set(wide.columns):
        gap = (wide["novelty"] - wide["usefulness"]).rename("novelty_minus_usefulness_leniency")
        rep = rep.merge(gap, left_on="rater_id", right_index=True, how="left")
    return rep


def judge_as_rater(judge_scores: pd.DataFrame, judge_col: str = "judge", condition: str = "judge") -> pd.DataFrame:
    """Turn judge scores into rater rows (first sample per idea and dimension)."""
    j = judge_scores.copy()
    if "sample_index" in j:
        j = j.sort_values("sample_index").drop_duplicates(["idea_id", "dimension", judge_col])
    return pd.DataFrame({"idea_id": j.idea_id, "rater_id": "judge:" + j[judge_col].astype(str),
                         "dimension": j.dimension, "score": j.score.astype(float), "condition": condition})


# --- Judges against the reference ----------------------------------------------------------

def judge_agreement(ratings: pd.DataFrame, judge_scores: pd.DataFrame, reference_conditions: set[str] | None = None,
                    judge_col: str = "judge") -> pd.DataFrame:
    """Per judge (and perturbation, if present) and dimension: Spearman and Pearson with the
    reference mean, leniency (mean difference) and n ideas."""
    ref = reference(ratings, reference_conditions)
    keys = [judge_col] + (["perturbation_id"] if "perturbation_id" in judge_scores else [])
    js = judge_scores.dropna(subset=["score"])
    if "sample_index" in js:
        js = js.groupby(keys + ["idea_id", "dimension"]).score.mean().reset_index()
    rows = []
    for key, sub in js.groupby(keys + ["dimension"]):
        key = key if isinstance(key, tuple) else (key,)
        m = sub.merge(ref, on=["idea_id", "dimension"])
        if len(m) < 3:
            continue
        rows.append({**dict(zip(keys + ["dimension"], key)), "n_ideas": len(m),
                     "spearman": float(stats.spearmanr(m.score, m["mean"]).statistic),
                     "pearson": float(stats.pearsonr(m.score, m["mean"]).statistic) if m.score.std() > 0 else np.nan,
                     "leniency": float((m.score - m["mean"]).mean())})
    return pd.DataFrame(rows)


def contested_detection(ratings: pd.DataFrame, judge_samples: pd.DataFrame, reference_conditions: set[str] | None = None,
                        judge_col: str = "judge") -> pd.DataFrame:
    """Spearman between a judge's SD across repeated samples and the human SD, over ideas."""
    ref = reference(ratings, reference_conditions)
    js = judge_samples.dropna(subset=["score"]).groupby([judge_col, "idea_id", "dimension"]).score.agg(judge_sd="std", samples="count").reset_index()
    js = js[js.samples > 1]
    rows = []
    for (j, dim), sub in js.groupby([judge_col, "dimension"]):
        m = sub.merge(ref, on=["idea_id", "dimension"]).dropna(subset=["judge_sd", "sd"])
        if len(m) < 3:
            continue
        rho = stats.spearmanr(m.judge_sd, m.sd).statistic if m.judge_sd.std() > 0 else np.nan
        rows.append({judge_col: j, "dimension": dim, "n_ideas": len(m), "spearman_judge_sd_vs_human_sd": float(rho),
                     "mean_judge_sd": float(m.judge_sd.mean()), "mean_human_sd": float(m.sd.mean())})
    return pd.DataFrame(rows)


def drift(judge_scores: pd.DataFrame, ratings: pd.DataFrame | None = None, baseline: str = "P0_baseline",
          judge_col: str = "judge", reference_conditions: set[str] | None = None) -> pd.DataFrame:
    """Per judge, perturbation and dimension: mean absolute change in score from the baseline
    prompt, and (with ratings) the change in Spearman agreement with the reference."""
    js = judge_scores.dropna(subset=["score"]).groupby([judge_col, "perturbation_id", "idea_id", "dimension"]).score.mean().reset_index()
    base = js[js.perturbation_id == baseline].rename(columns={"score": "base"}).drop(columns="perturbation_id")
    m = js.merge(base, on=[judge_col, "idea_id", "dimension"])
    out = m.groupby([judge_col, "perturbation_id", "dimension"]).apply(
        lambda g: pd.Series({"mean_abs_change": float((g.score - g.base).abs().mean()),
                             "mean_change": float((g.score - g.base).mean()), "n_ideas": len(g)}),
        include_groups=False).reset_index()
    agr = judge_agreement(ratings, js, reference_conditions, judge_col) if ratings is not None else pd.DataFrame()
    if len(agr):  # needs at least three ideas with human ratings
        b = agr[agr.perturbation_id == baseline][[judge_col, "dimension", "spearman"]].rename(columns={"spearman": "spearman_base"})
        agr = agr.merge(b, on=[judge_col, "dimension"])
        agr["spearman_change"] = agr.spearman - agr.spearman_base
        out = out.merge(agr[[judge_col, "perturbation_id", "dimension", "spearman", "spearman_change"]],
                        on=[judge_col, "perturbation_id", "dimension"], how="left")
    return out


# --- AI judges ------------------------------------------------------------------------------

def judge_position(ratings: pd.DataFrame, judge_scores: pd.DataFrame, reference_conditions: set[str] | None = None) -> pd.DataFrame:
    """Each judge (P0, first sample) entered as one more rater: its leniency, noise and agreement,
    and the percentile of its leniency and agreement among the human raters of the reference pool."""
    js = judge_scores[(judge_scores.perturbation_id == "P0_baseline") & (judge_scores.sample_index == 0)].dropna(subset=["score"])
    jr = judge_as_rater(js)
    rep = rater_report(pd.concat([ratings, jr], ignore_index=True), reference_conditions)
    humans = rep[rep.condition.isin(reference_conditions) if reference_conditions else rep.condition != "judge"]
    out = rep[rep.condition == "judge"].copy()
    out["judge"] = out.rater_id.str.removeprefix("judge:")
    for col in ("leniency", "agreement_spearman"):
        out[f"{col}_human_percentile"] = [
            float((humans[humans.dimension == d][col] < v).mean() * 100) for d, v in zip(out.dimension, out[col])]
    h = humans.groupby("dimension")[["leniency", "noise", "agreement_spearman"]].median().add_prefix("human_median_")
    return out.merge(h, left_on="dimension", right_index=True).drop(columns=["rater_id", "condition"])


# --- Judge diagnostics: uncertainty, rater equivalents, calibration, pooled spread ----------

def _ref_means(ratings: pd.DataFrame, raters: list[str], dim: str) -> pd.Series:
    r = ratings[(ratings.dimension == dim) & ratings.rater_id.isin(raters)]
    return r.groupby("idea_id").score.mean()


def bootstrap_agreement(ratings: pd.DataFrame, judge_scores: pd.DataFrame, reference_conditions: set[str] | None = None,
                        n_boot: int = 2000, seed: int = 1) -> pd.DataFrame:
    """Percentile 95% CI for each judge's Spearman agreement with the reference (P0), resampling
    ideas with replacement. Also the share of resamples in which the judge beats the median
    human rater's leave-one-out agreement computed on the same resample."""
    rng = np.random.default_rng(seed)
    pool = ratings if reference_conditions is None else ratings[ratings.condition.isin(reference_conditions)]
    js = judge_scores[(judge_scores.perturbation_id == "P0_baseline") & (judge_scores.sample_index == 0)].dropna(subset=["score"])
    raters = sorted(pool.rater_id.unique())
    rows = []
    for dim in sorted(js.dimension.unique()):
        wide = pool[pool.dimension == dim].pivot_table(index="idea_id", columns="rater_id", values="score")
        ideas = wide.index.to_numpy()
        tot, cnt = wide.sum(axis=1), wide.count(axis=1)
        loo = {r: (tot - wide[r].fillna(0)) / (cnt - wide[r].notna()) for r in raters}
        ref = wide.mean(axis=1)
        jw = js[js.dimension == dim].pivot_table(index="idea_id", columns="judge", values="score").reindex(ideas)
        boots = {j: [] for j in jw.columns}
        beats = {j: 0 for j in jw.columns}
        for _ in range(n_boot):
            idx = rng.integers(0, len(ideas), len(ideas))
            if len(set(idx)) < 4:
                continue
            refb = ref.to_numpy()[idx]
            hum = [stats.spearmanr(wide[r].to_numpy()[idx], loo[r].to_numpy()[idx]).statistic for r in raters]
            med = np.nanmedian(hum)
            for j in jw.columns:
                rho = stats.spearmanr(jw[j].to_numpy()[idx], refb).statistic
                boots[j].append(rho)
                beats[j] += int(rho > med)
        for j in jw.columns:
            b = np.array([x for x in boots[j] if not np.isnan(x)])
            rows.append({"judge": j, "dimension": dim, "spearman": float(stats.spearmanr(jw[j], ref).statistic),
                         "ci_low": float(np.percentile(b, 2.5)), "ci_high": float(np.percentile(b, 97.5)),
                         "share_resamples_above_median_human": beats[j] / max(len(boots[j]), 1)})
    return pd.DataFrame(rows)


def rater_equivalents(ratings: pd.DataFrame, judge_scores: pd.DataFrame, reference_conditions: set[str] | None = None,
                      k_max: int = 19, reps: int = 500, seed: int = 2) -> pd.DataFrame:
    """How many human raters, averaged, rank ideas as well as each judge? Each repetition splits
    the reference pool at random into a reference half and a candidate half; the mean of k
    candidates and each judge are scored (Spearman) against the same reference half. Returns the
    mean agreement for k = 1..k_max and, per judge, the smallest k whose mean agreement reaches
    the judge's."""
    rng = np.random.default_rng(seed)
    pool = ratings if reference_conditions is None else ratings[ratings.condition.isin(reference_conditions)]
    js = judge_scores[(judge_scores.perturbation_id == "P0_baseline") & (judge_scores.sample_index == 0)].dropna(subset=["score"])
    raters = np.array(sorted(pool.rater_id.unique()))
    half = len(raters) // 2
    k_max = min(k_max, half)
    curves, judge_rows = [], []
    for dim in sorted(js.dimension.unique()):
        wide = pool[pool.dimension == dim].pivot_table(index="idea_id", columns="rater_id", values="score")
        jw = js[js.dimension == dim].pivot_table(index="idea_id", columns="judge", values="score").reindex(wide.index)
        human = np.zeros((reps, k_max))
        judge = {j: np.zeros(reps) for j in jw.columns}
        for t in range(reps):
            perm = rng.permutation(raters)
            ref = wide[perm[half:]].mean(axis=1)
            cand = perm[:half]
            for k in range(1, k_max + 1):
                human[t, k - 1] = stats.spearmanr(wide[cand[:k]].mean(axis=1), ref).statistic
            for j in jw.columns:
                judge[j][t] = stats.spearmanr(jw[j], ref).statistic
        mean_curve = np.nanmean(human, axis=0)
        curves += [{"dimension": dim, "k": k, "human_mean_agreement": float(mean_curve[k - 1])} for k in range(1, k_max + 1)]
        for j in jw.columns:
            jm = float(np.nanmean(judge[j]))
            reach = [k for k in range(1, k_max + 1) if mean_curve[k - 1] >= jm]
            judge_rows.append({"judge": j, "dimension": dim, "judge_agreement_vs_half_reference": jm,
                               "rater_equivalents": reach[0] if reach else f">{k_max}"})
    return pd.DataFrame(curves), pd.DataFrame(judge_rows)


def leniency_by_perturbation(ratings: pd.DataFrame, judge_scores: pd.DataFrame,
                             reference_conditions: set[str] | None = None) -> pd.DataFrame:
    """Each judge's leniency (mean score minus reference mean) under every perturbation (first
    sample, all variants and seeds averaged per idea): how far calibration-like changes such as
    worked examples (P5) or batching (P4) move the level."""
    ref = reference(ratings, reference_conditions)[["idea_id", "dimension", "mean"]]
    js = judge_scores[judge_scores.sample_index == 0].dropna(subset=["score"])
    m = js.groupby(["judge", "perturbation_id", "idea_id", "dimension"]).score.mean().reset_index().merge(ref, on=["idea_id", "dimension"])
    out = m.assign(diff=m.score - m["mean"]).groupby(["judge", "perturbation_id", "dimension"])["diff"].mean().rename("leniency").reset_index()
    return out


def pooled_spread(ratings: pd.DataFrame, judge_scores: pd.DataFrame, reference_conditions: set[str] | None = None) -> pd.DataFrame:
    """Does spread across judges, or across a judge's prompt variants, track human disagreement?
    Spearman over ideas between human SD and (a) SD across the five judges at P0, (b) SD across
    perturbations P0-P5 within each judge, averaged over judges, (c) SD of all judges x
    perturbations pooled, (d) SD across one judge's five temperature-1 samples, averaged."""
    ref = reference(ratings, reference_conditions)[["idea_id", "dimension", "sd"]].rename(columns={"sd": "human_sd"})
    js = judge_scores.dropna(subset=["score"])
    first = js[js.sample_index == 0]
    p0 = first[first.perturbation_id == "P0_baseline"]
    grid = first[first.perturbation_id != "P6_temperature_sampled"]
    per_pert = grid.groupby(["judge", "perturbation_id", "idea_id", "dimension"]).score.mean().reset_index()
    p6 = js[js.perturbation_id == "P6_temperature_sampled"]
    sources = {
        "across judges (P0)": p0.groupby(["idea_id", "dimension"]).score.std(),
        "across prompt variants, within judge": per_pert.groupby(["judge", "idea_id", "dimension"]).score.std().groupby(["idea_id", "dimension"]).mean(),
        "all judges x variants pooled": per_pert.groupby(["idea_id", "dimension"]).score.std(),
        "temperature samples, within judge": p6.groupby(["judge", "idea_id", "dimension"]).score.std().groupby(["idea_id", "dimension"]).mean(),
    }
    rows = []
    for name, sd in sources.items():
        m = sd.rename("machine_sd").reset_index().merge(ref, on=["idea_id", "dimension"])
        for dim, g in m.groupby("dimension"):
            rho = stats.spearmanr(g.machine_sd, g.human_sd)
            rows.append({"source": name, "dimension": dim, "mean_machine_sd": float(g.machine_sd.mean()),
                         "mean_human_sd": float(g.human_sd.mean()), "spearman": float(rho.statistic), "p": float(rho.pvalue)})
    return pd.DataFrame(rows)


# --- The AI judge diagnostic: one report per judge on the four measures --------------------

def _icc_a1(df: pd.DataFrame, rater_col: str) -> float:
    import pingouin as pg

    try:
        with np.errstate(divide="ignore", invalid="ignore"):  # a judge that never varies has zero error variance
            t = pg.intraclass_corr(data=df, targets="idea_id", raters=rater_col, ratings="score", nan_policy="omit")
    except (AssertionError, ValueError):  # too few ideas or raters to estimate
        return float("nan")
    return float(t.set_index("Type").loc["ICC(A,1)", "ICC"])


def _first(js: pd.DataFrame) -> pd.DataFrame:
    if "sample_index" in js:
        js = js[js.sample_index == 0]
    return js


def agreement_ladder(ratings: pd.DataFrame, judge_scores: pd.DataFrame,
                     reference_conditions: set[str] | None = None,
                     baseline: str = "P0_baseline", sampled: str = "P6_temperature_sampled") -> pd.DataFrame:
    """ICC(A,1), two-way random, absolute agreement, per dimension, for three kinds of pair:
    person with person (the reference pool), AI judge with AI judge (all judges at the baseline
    prompt, temperature 0) and each AI judge with itself (its repeated samples at temperature 1).
    This is panel A of Figure 1: people disagree, judges agree with each other and with themselves."""
    pool = ratings if reference_conditions is None or "condition" not in ratings else ratings[ratings.condition.isin(reference_conditions)]
    js = judge_scores.dropna(subset=["score"])
    rows = []
    for dim in sorted(js.dimension.unique()):
        h = pool[pool.dimension == dim].dropna(subset=["score"])
        if h.rater_id.nunique() > 1:
            rows.append({"dimension": dim, "pair": "person with person", "judge": "", "n_raters": h.rater_id.nunique(),
                         "n_ideas": h.idea_id.nunique(), "icc_a1": _icc_a1(h, "rater_id")})
        p0 = _first(js[(js.perturbation_id == baseline) & (js.dimension == dim)])
        p0 = p0.groupby(["judge", "idea_id"]).score.mean().reset_index()
        if p0.judge.nunique() > 1:
            rows.append({"dimension": dim, "pair": "AI judge with AI judge", "judge": "", "n_raters": p0.judge.nunique(),
                         "n_ideas": p0.idea_id.nunique(), "icc_a1": _icc_a1(p0, "judge")})
        p6 = js[(js.perturbation_id == sampled) & (js.dimension == dim)]
        for j, g in p6.groupby("judge"):
            if g.sample_index.nunique() < 2:
                continue
            g = g.assign(sample_k=g.sample_index.astype(str))
            icc = _icc_a1(g, "sample_k") if g.score.std() > 0 else 1.0  # identical scores: perfect agreement
            rows.append({"dimension": dim, "pair": "AI judge with itself", "judge": j, "n_raters": g.sample_k.nunique(),
                         "n_ideas": g.idea_id.nunique(), "icc_a1": icc})
    return pd.DataFrame(rows)


def judge_diagnostics(ratings: pd.DataFrame, judge_scores: pd.DataFrame,
                      reference_conditions: set[str] | None = None,
                      baseline: str = "P0_baseline", sampled: str = "P6_temperature_sampled") -> pd.DataFrame:
    """The AI judge diagnostic: one row per judge and dimension, on the four measures.

    Agreement   Spearman of the judge's baseline scores with the reference mean, and its percentile
                among the human raters' leave-one-out agreement.
    Leniency    the judge's mean score minus the reference mean, and its percentile among the
                human raters' leniencies (a low percentile means harsher than most people).
    Stability   mean absolute change in score between the baseline prompt and each perturbation
                that should not matter (averaged over perturbations other than the sampled run).
    Variance    the judge's ICC(A,1) with itself across repeated samples (people: see
                ``agreement_ladder``), and the Spearman between its per-idea SD across samples and
                the human per-idea SD (does it know which ideas people disagree about?).
    Measures that need a run the data do not have (no perturbations, no repeated samples) are NaN."""
    js = judge_scores.dropna(subset=["score"])
    if "sample_index" not in js:
        js = js.assign(sample_index=0)
    if "perturbation_id" not in js:
        js = js.assign(perturbation_id=baseline)
    has_cond = "condition" in ratings
    r = ratings if has_cond else ratings.assign(condition="all")
    conds = reference_conditions if has_cond else None
    pos = judge_position(r, js, conds)
    keep = ["judge", "dimension", "n_ideas", "agreement_spearman", "agreement_spearman_human_percentile",
            "leniency", "leniency_human_percentile"]
    out = pos[keep].rename(columns={"agreement_spearman": "agreement", "agreement_spearman_human_percentile": "agreement_percentile",
                                    "leniency_human_percentile": "leniency_percentile"})
    grid = js[js.perturbation_id != sampled]
    if grid.perturbation_id.nunique() > 1:
        d = drift(grid, baseline=baseline)
        d = d[d.perturbation_id != baseline].groupby(["judge", "dimension"]).mean_abs_change.mean().rename("stability_mean_abs_change")
        out = out.merge(d.reset_index(), on=["judge", "dimension"], how="left")
    else:
        out["stability_mean_abs_change"] = np.nan
    lad = agreement_ladder(r, js, conds, baseline, sampled)
    self_icc = lad[lad.pair == "AI judge with itself"][["judge", "dimension", "icc_a1"]].rename(columns={"icc_a1": "variance_self_icc"})
    out = out.merge(self_icc, on=["judge", "dimension"], how="left")
    cont = contested_detection(r, js[js.perturbation_id == sampled], conds)
    if len(cont):
        cont = cont.rename(columns={"spearman_judge_sd_vs_human_sd": "variance_tracks_human_sd",
                                    "mean_judge_sd": "variance_judge_sd", "mean_human_sd": "variance_human_sd"})
        out = out.merge(cont[["judge", "dimension", "variance_judge_sd", "variance_human_sd", "variance_tracks_human_sd"]],
                        on=["judge", "dimension"], how="left")
    for c in ("variance_self_icc", "variance_judge_sd", "variance_human_sd", "variance_tracks_human_sd"):
        if c not in out:
            out[c] = np.nan
    return out.sort_values(["dimension", "judge"], ignore_index=True)


def diagnose(ratings: pd.DataFrame, judge_scores: pd.DataFrame,
             reference_conditions: set[str] | None = None) -> dict[str, pd.DataFrame]:
    """The AI judge diagnostic for any rated dataset: the human ceiling, per-idea disagreement,
    the agreement ladder, the four-measure diagnostic and drift. Inputs are the README long
    formats; nothing leaves the machine."""
    has_cond = "condition" in ratings
    r = ratings if has_cond else ratings.assign(condition="all")
    conds = reference_conditions if has_cond else None
    out = {"human_ceiling": human_ceiling(r[r.condition.isin(conds)] if conds else r, by_condition=False),
           "idea_disagreement": idea_summary(r, conds),
           "agreement_ladder": agreement_ladder(r, judge_scores, conds),
           "judge_diagnostics": judge_diagnostics(r, judge_scores, conds)}
    js = judge_scores.dropna(subset=["score"])
    if "perturbation_id" in js and js.perturbation_id.nunique() > 1:
        out["judge_drift"] = drift(_first(js), r, reference_conditions=conds)
    return out

# --- Command line ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from . import RESULTS_DIR
    from .load import EXAMPLE_DIR

    p = argparse.ArgumentParser(prog="python -m benchmark.agree",
                                description="The AI judge diagnostic for human ratings and AI judge scores.")
    p.add_argument("--ratings", default=str(EXAMPLE_DIR / "ratings_long.csv"), help="ratings_long.csv (default: the example)")
    p.add_argument("--scores", default=str(EXAMPLE_DIR / "judge_scores.csv"), help="judge_scores.csv from benchmark.judge")
    p.add_argument("--reference-condition", default=None,
                   help="build the reference from raters in this condition only (default: all raters)")
    p.add_argument("--out", default=None, help="output folder (default: results/diagnostics/<scores folder>)")
    a = p.parse_args(argv)
    conds = {a.reference_condition} if a.reference_condition else None
    res = diagnose(pd.read_csv(a.ratings), pd.read_csv(a.scores), conds)
    out = Path(a.out) if a.out else RESULTS_DIR / "diagnostics" / Path(a.scores).resolve().parent.name
    out.mkdir(parents=True, exist_ok=True)
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {out / (name + '.csv')}")
    print(res["judge_diagnostics"].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
