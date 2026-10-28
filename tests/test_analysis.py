"""Analysis pipeline on the example folder (synthetic ratings; runs in CI with no keys)."""
import pandas as pd
import pytest

from benchmark import REPO_DIR, agree, load
from benchmark.judge import DiskCache, FakeBackend, load_rubric, parse_batch, parse_scores, run
from benchmark.perturb import load_perturbations, plan_jobs

EXAMPLE = REPO_DIR / "examples" / "alternative_uses"
RUBRIC = REPO_DIR / "configs" / "rubrics" / "novelty_usefulness_v1.0.yaml"


@pytest.fixture(scope="module")
def ratings():
    return pd.read_csv(EXAMPLE / "ratings_long.csv")


def test_example_folder_is_valid(ratings):
    tables = load.read_folder(EXAMPLE)
    load.validate(tables)
    assert ratings.rater_id.nunique() == 39 and ratings.idea_id.nunique() == 15
    assert set(ratings.score.unique()) <= {1, 2, 3, 4, 5}
    assert set(tables["judge_scores"].judge) == {"claude-opus-5.5", "gpt-6.1-sol", "gemini-4-argon", "deepseek-v4-pro", "qwen3.8-27b"}


def test_ceiling_and_raters_needed(ratings):
    c = agree.human_ceiling(ratings, by_condition=False)
    assert set(c.dimension) == {"novelty", "usefulness"} and (c.icc_ak > c.icc_a1).all()
    assert agree.raters_needed(0.3, 0.8) == 10 and agree.raters_needed(0.5, 0.8) == 4
    assert abs(agree.spearman_brown(0.3, 10) - 0.811) < 0.001


def test_rater_report(ratings):
    rep = agree.rater_report(ratings)
    assert {"leniency", "noise", "agreement_spearman", "novelty_minus_usefulness_leniency"} <= set(rep.columns)
    assert abs(rep[rep.dimension == "novelty"].leniency.mean()) < 0.05   # leave-one-out reference is centred


def test_parsers():
    assert parse_scores('{"novelty": 4, "usefulness": "2"}', ["novelty", "usefulness"], (1, 5)) == {"novelty": 4.0, "usefulness": 2.0}
    assert parse_scores('{"novelty": 9, "usefulness": 2}', ["novelty", "usefulness"], (1, 5)) is None
    assert parse_scores("no json", ["novelty"], (1, 5)) is None
    out = parse_batch('[{"id": 1, "novelty": 3, "usefulness": 3}, {"id": 2, "novelty": 7}]', ["novelty", "usefulness"], (1, 5), 2)
    assert out == {1: {"novelty": 3.0, "usefulness": 3.0}}


def test_plan_jobs_cover_grid(ratings):
    rubric, _ = load_rubric(RUBRIC)
    ideas, tasks = pd.read_csv(EXAMPLE / "ideas.csv"), pd.read_csv(EXAMPLE / "tasks.csv")
    jobs = plan_jobs(load_perturbations("perturbations_v1.1"), ideas, tasks, rubric, None, agree.reference(ratings))
    by = pd.Series([j["perturbation_id"] for j in jobs]).value_counts().to_dict()
    assert by["P0_baseline"] == 15 and by["P1_rubric_reworded"] == 45 and by["P6_temperature_sampled"] == 75
    assert by["P5_few_shot"] == 12                       # three example ideas excluded
    swapped = next(j for j in jobs if j["perturbation_id"] == "P2_dimension_order_swapped")
    assert swapped["values"]["statement_block"].startswith("Statement 1 (usefulness)")


def test_run_with_fake_backend_uses_cache(tmp_path, ratings):
    judges = [{"label": "fake", "model": "fake/model"}]
    cache = DiskCache(tmp_path / "cache")
    kw = dict(rubric_path=RUBRIC, perturbation_ids=["P0_baseline", "P4_batch_order_shuffled"], cache=cache,
              out_dir=tmp_path / "out", backend_factory=lambda j: FakeBackend())
    df = run(str(EXAMPLE), judges, **kw)
    assert len(df) and df.score.notna().all()
    n_files = len(list((tmp_path / "cache").rglob("*.json")))
    df2 = run(str(EXAMPLE), judges, **kw)
    assert len(list((tmp_path / "cache").rglob("*.json"))) == n_files      # second run: all cached
    assert df.sort_values(list(df.columns)).reset_index(drop=True).equals(df2.sort_values(list(df2.columns)).reset_index(drop=True))
    j = df[df.perturbation_id == "P0_baseline"].assign(judge="fake")
    agr = agree.judge_agreement(ratings, j)
    assert set(agr.dimension) == {"novelty", "usefulness"}


def test_judge_tables_with_fake_judges(ratings):
    """Uncertainty, rater equivalents, calibration and pooled spread with two simulated judges."""
    import numpy as np

    ref = agree.reference(ratings)
    rng = np.random.default_rng(0)
    rows = []
    for judge, noise in (("good", 0.1), ("bad", 2.0)):
        for pid, samples in (("P0_baseline", 1), ("P4_batch_order_shuffled", 1), ("P6_temperature_sampled", 3)):
            for s in range(samples):
                for r in ref.itertuples():
                    rows.append({"judge": judge, "perturbation_id": pid, "sample_index": s, "idea_id": r.idea_id,
                                 "dimension": r.dimension, "score": r.mean + rng.normal(0, noise)})
    js = pd.DataFrame(rows)
    r = ratings.assign(condition="all")
    ci = agree.bootstrap_agreement(r, js, {"all"}, n_boot=200)
    good = ci[(ci.judge == "good") & (ci.dimension == "novelty")].iloc[0]
    assert good.ci_low <= good.spearman <= good.ci_high and good.spearman > 0.9
    curve, eq = agree.rater_equivalents(r, js, {"all"}, reps=30)
    assert curve.groupby("dimension").human_mean_agreement.apply(lambda s: s.iloc[-1] > s.iloc[0]).all()
    assert set(eq.judge) == {"good", "bad"}
    lp = agree.leniency_by_perturbation(r, js, {"all"})
    assert set(lp.perturbation_id) == {"P0_baseline", "P4_batch_order_shuffled", "P6_temperature_sampled"}
    ps = agree.pooled_spread(r, js, {"all"})
    assert {"across judges (P0)", "temperature samples, within judge"} <= set(ps.source)
