"""The AI judge diagnostic, the bring-your-own-data path, model-written rewordings, the variance
pipeline and the field test. Everything runs offline on the example folders and the demo data."""
import shutil

import numpy as np
import pandas as pd
import pytest

from benchmark import PROMPTS_DIR, REPO_DIR, agree, shadow, variance
from benchmark.judge import DiskCache, FakeBackend, dataset_tables, load_rubric, run
from benchmark.perturb import (DEFAULT_SET, _FakeReworder, generate_rewordings, load_perturbations, plan_jobs,
                               write_perturbation_set)

EXAMPLE = REPO_DIR / "examples" / "alternative_uses"
DEMO = REPO_DIR / "examples" / "demo"
SCREEN = REPO_DIR / "examples" / "screen"
RUBRIC = REPO_DIR / "configs" / "rubrics" / "novelty_usefulness_v1.0.yaml"
JUDGES = [{"label": f"j{k}", "model": f"fake-{k}"} for k in range(3)]


class NoisyFake(FakeBackend):
    """Fake judge whose score also depends on the model and, at temperature above 0, on the call,
    so judges differ and repeated samples differ."""

    def __init__(self, model):
        super().__init__(model)
        import itertools
        import threading
        self._n, self._lock = itertools.count(), threading.Lock()

    def complete(self, system, user, *, temperature, max_tokens):
        with self._lock:
            k = next(self._n)
        salt = self.model + (str(k) if temperature > 0 else "")
        return super().complete(system, user + salt, temperature=temperature, max_tokens=max_tokens)


@pytest.fixture(scope="module")
def example_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("example")
    df = run(str(EXAMPLE), JUDGES, rubric_path=RUBRIC, cache=DiskCache(tmp / "cache"), out_dir=tmp / "out",
             backend_factory=lambda j: NoisyFake(j["model"]), perturbation_ids=["P0_baseline", "P1_rubric_reworded", "P6_temperature_sampled"])
    return pd.read_csv(EXAMPLE / "ratings_long.csv"), df


def test_agreement_ladder_has_three_kinds_of_pair(example_run):
    ratings, js = example_run
    lad = agree.agreement_ladder(ratings, js)
    nov = lad[lad.dimension == "novelty"]
    assert set(nov.pair) == {"person with person", "AI judge with AI judge", "AI judge with itself"}
    assert nov[nov.pair == "AI judge with itself"].judge.nunique() == 3
    assert nov[nov.pair == "person with person"].n_raters.iloc[0] == 39


def test_judge_diagnostic_reports_four_measures(example_run):
    ratings, js = example_run
    d = agree.judge_diagnostics(ratings, js)
    assert len(d) == 6  # 3 judges x 2 dimensions
    for col in ("agreement", "leniency", "stability_mean_abs_change", "variance_self_icc", "variance_tracks_human_sd",
                "human_disagreement_reliability", "variance_ceiling"):
        assert col in d and d[col].notna().any(), col
    assert d.leniency_percentile.between(0, 100).all()
    assert d.variance_ceiling.between(0, 1).all()
    assert d.variance_tracks_human_sd_corrected.dropna().between(-1, 1).all()


def test_every_diagnostic_reports_how_well_disagreement_was_measured(example_run):
    """Risk 1 of the proposal: every diagnostic reports how much of the human disagreement could
    be measured at all."""
    ratings, js = example_run
    res = agree.diagnose(ratings, js)
    rel = res["disagreement_reliability"]
    whole = rel[rel.raters_per_panel.isna()].set_index("dimension")
    assert set(whole.index) == {"novelty", "usefulness"}
    assert whole.full_pool_reliability.between(0, 1).all() and whole.ceiling_for_variance_spearman.between(0, 1).all()
    assert (rel.raters_per_panel.dropna() <= 19).all()       # crossed example: the k-rater curve up to half the pool


def test_disagreement_reliability_works_when_raters_see_different_ideas():
    """Planned-missing designs: each idea has its own raters."""
    rng = np.random.default_rng(3)
    sd = np.linspace(0.3, 1.6, 30)
    rows = [{"idea_id": f"i{i}", "rater_id": f"r{i}_{k}", "dimension": "novelty", "score": float(np.clip(np.rint(3 + rng.normal(0, sd[i])), 1, 5))}
            for i in range(30) for k in range(40 if i < 10 else 10)]
    rel = agree.disagreement_reliability(pd.DataFrame(rows))
    assert len(rel) == 1 and rel.raters_per_panel.isna().all()   # not crossed: no k-rater curve
    assert rel.full_pool_reliability.iloc[0] > 0.3                # strongly different spreads are measurable


def test_diagnose_runs_on_demo_data_without_conditions(tmp_path):
    ratings = pd.read_csv(DEMO / "ratings_long.csv").drop(columns="condition")
    js = run(str(DEMO), JUDGES[:2], rubric_path=RUBRIC, cache=DiskCache(tmp_path / "c"), out_dir=tmp_path / "o",
             backend_factory=lambda j: NoisyFake(j["model"]), perturbation_ids=["P0_baseline", "P6_temperature_sampled"])
    res = agree.diagnose(ratings, js)
    assert {"human_ceiling", "idea_disagreement", "agreement_ladder", "judge_diagnostics", "disagreement_reliability"} <= set(res)
    assert len(res["judge_diagnostics"]) == 4


def test_data_dir_is_a_dataset():
    ideas, tasks, ref = dataset_tables(str(DEMO))
    assert len(ideas) == 6 and len(tasks) == 2 and ref is not None


def test_reword_writes_a_new_versioned_file(tmp_path):
    for f in ("perturbations_v1.1.yaml", "perturbations_v1.2.yaml", "pointwise_v1.1.yaml"):
        shutil.copy(PROMPTS_DIR / f, tmp_path / f)
    rubric, _ = load_rubric(RUBRIC)
    variants, prov = generate_rewordings(rubric, _FakeReworder(), n=3, cache=DiskCache(tmp_path / "cache"))
    assert len(variants) == 3 and set(variants[0]) == {"novelty", "usefulness"}
    out = write_perturbation_set(variants, prov, out_name="perturbations_v9.9", prompts_dir=tmp_path, date="2026-10-26")
    p1 = next(p for p in load_perturbations("perturbations_v9.9", tmp_path)["perturbations"] if p["id"] == "P1_rubric_reworded")
    assert p1["generated_by"]["generator_model"] == "fake-reworder" and len(p1["variants"]) == 3
    with pytest.raises(FileExistsError):
        write_perturbation_set(variants, prov, out_name="perturbations_v9.9", prompts_dir=tmp_path)
    assert out.exists()


def test_variance_sources_and_reliability(example_run):
    ratings, js = example_run
    res = variance.evaluate(ratings, js)
    src = res["variance_sources"]
    assert {"model_families", "temperature", "prompt_variants"} <= set(src.source)
    rel = res["disagreement_reliability"]
    assert (rel.ceiling_for_variance_spearman.dropna().between(0, 1)).all()
    test = res["pipeline_test"]
    assert set(test.dimension) == {"novelty", "usefulness"} and (test.splits == 200).all()
    assert test.test_auc_contested.between(0, 1).all()


def test_pipeline_is_judged_on_contested_ideas_it_has_not_seen(example_run):
    """The proposal: judge the pipeline by how well it picks out the contested ideas among ideas it
    has not seen. A single judge that predicts the spread is a candidate too."""
    ratings, js = example_run
    ideas = sorted(ratings.idea_id.unique())
    human = ratings[ratings.dimension == "novelty"].groupby("idea_id").score.std()
    expected = pd.DataFrame({"judge": "oracle", "idea_id": ideas, "dimension": "novelty",
                             "forecast_mean": 3.0, "forecast_sd": [human[i] for i in ideas]})
    cands = variance.candidate_spreads(js, expected)
    assert "expected spread: oracle" in cands and any(n.startswith("model families:") for n in cands)
    cv = variance.cross_validated_search(js, ratings, expected=expected, n_splits=50)
    assert len(cv) == 50 and set(cv.columns) >= {"chosen", "test_auc", "test_spearman", "all_judges_test_auc"}
    assert (cv.chosen == "expected spread: oracle").mean() > 0.5     # a single judge that predicts spread wins
    assert cv.test_auc.mean() > 0.8


def test_rater_twin_and_perspective_jobs(example_run):
    ratings, _ = example_run
    ideas, tasks = pd.read_csv(EXAMPLE / "ideas.csv"), pd.read_csv(EXAMPLE / "tasks.csv")
    rubric, _ = load_rubric(RUBRIC)
    train = list(ideas.idea_id[:10])
    raters = sorted(ratings.rater_id.unique())[:3]
    jobs = variance.plan_rater_twin_jobs(ideas, tasks, rubric, ratings, train, raters)
    assert len(jobs) == 3 * (len(ideas) - 10)
    assert all(j["idea_ids"][0] not in train for j in jobs)
    assert "this same rater" in jobs[0]["values"]["few_shot_block"]
    pj = variance.plan_perspective_jobs(ideas, tasks, rubric)
    assert len(pj) == 3 * len(ideas) and "investor" in {j["variant"] for j in pj}


def test_expected_spread_parser():
    out = variance.parse_expected_spread('{"novelty": {"1": 0, "2": 0, "3": 5, "4": 5, "5": 0}, "usefulness": {"1": 10}}',
                                         ["novelty", "usefulness"], (1, 5))
    assert out["novelty"] == (3.5, 0.5) and out["usefulness"] == (1.0, 0.0)
    assert variance.parse_expected_spread("not json", ["novelty"], (1, 5)) is None


def test_panel_output_flags_contested():
    m = pd.DataFrame({"member": ["a", "b"] * 4, "idea_id": np.repeat(["i1", "i2", "i3", "i4"], 2), "dimension": "novelty",
                      "score": [1, 5, 3, 3, 2, 3, 4, 4]})
    out = variance.panel_output(m)
    assert out.set_index("idea_id").contested["i1"] and not out.set_index("idea_id").contested["i2"]


def test_variance_run_makes_the_new_sources(tmp_path):
    res = variance.run_sources(EXAMPLE, JUDGES[:2], rubric_path=RUBRIC, n_twins=3, out_dir=tmp_path,
                               backend_factory=lambda j: FakeBackend(j["model"]), cache=DiskCache(tmp_path / "cache"))
    js = res["judge_scores"]
    assert set(js.perturbation_id) == {"V_perspective", "V_rater_twin"}
    assert set(js[js.perturbation_id == "V_perspective"].variant) == {"customer", "engineer", "investor"}
    assert js[js.perturbation_id == "V_rater_twin"].variant.nunique() == 3
    assert res["expected_spread"].forecast_sd.notna().all()
    assert (tmp_path / "judge_scores.csv").exists() and (tmp_path / "expected_spread.csv").exists()


def test_llm_rewordings_are_the_default_and_cover_the_rubric():
    raw = load_perturbations()
    assert raw["version"] == "1.2" and DEFAULT_SET == "perturbations_v1.2"
    p1 = next(p for p in raw["perturbations"] if p["id"] == "P1_rubric_reworded")
    assert p1["generated_by"]["generator_model"] and len(p1["variants"]) == 3


def test_a_rubric_the_rewordings_do_not_cover_is_stopped(tmp_path):
    rub = tmp_path / "r.yaml"
    rub.write_text(RUBRIC.read_text().replace("id: usefulness", "id: feasibility"))
    rubric, _ = load_rubric(rub)
    ideas, tasks = pd.read_csv(EXAMPLE / "ideas.csv"), pd.read_csv(EXAMPLE / "tasks.csv")
    with pytest.raises(ValueError, match="benchmark.perturb reword"):
        plan_jobs(load_perturbations(), ideas, tasks, rubric, ["P1_rubric_reworded"])
    assert plan_jobs(load_perturbations(), ideas, tasks, rubric, ["P0_baseline"])     # other conditions still run


def test_shadow_screen_counts_dropped_ideas_and_their_disagreement():
    ideas = pd.DataFrame({"idea_id": list("abcdef"), "idea_text": list("abcdef"), "advanced": [1, 1, 1, 0, 0, 0]})
    rows = []
    for i, (m, s) in zip("abcdef", [(4, 0), (4, 1.5), (4, 1.5), (2, 0), (2, 0), (3, 0)]):
        for r, d in enumerate([-1, 1, -1, 1]):
            rows.append({"idea_id": i, "rater_id": f"e{r}", "dimension": "novelty", "score": m + d * s / 1.0})
    js = pd.DataFrame({"judge": "j", "idea_id": list("abcdef"), "dimension": "novelty", "score": [5, 2, 3, 4, 1, 3],
                       "perturbation_id": "P0_baseline", "sample_index": 0})
    res = shadow.shadow_screen(ideas, pd.DataFrame(rows), js)
    j = res["shadow_summary"].set_index("judge").loc["j"]
    # AI top 3: a (5), d (4), then c or f tied at 3 -> c advanced with probability 1/2; b dropped
    assert j.organization_advanced == 3 and j.certainly_dropped == 1 and j.ai_would_drop == pytest.approx(1.5)
    assert j.evaluator_sd_of_dropped > j.evaluator_sd_of_kept      # the dropped advanced ideas were the contested ones
    sheet = res["debrief_sheet"].set_index("idea_id")
    assert sheet.loc["c", "ai_would_advance: j"] == 0.5 and "ai_score: all judges" in sheet


def test_shadow_screen_needs_every_decision():
    ideas = pd.DataFrame({"idea_id": ["a", "b"], "idea_text": ["a", "b"], "advanced": [1, None]})
    with pytest.raises(ValueError, match="decision"):
        shadow.shadow_screen(ideas, pd.DataFrame(columns=["idea_id", "rater_id", "dimension", "score"]),
                             pd.DataFrame(columns=["judge", "idea_id", "dimension", "score"]))


def test_screen_example_runs_end_to_end(tmp_path):
    from benchmark.load import read_folder, validate

    t = read_folder(SCREEN)
    validate(t)
    js = run(str(SCREEN), JUDGES, rubric_path=RUBRIC, cache=DiskCache(tmp_path / "c"), out_dir=tmp_path / "o",
             backend_factory=lambda j: FakeBackend(j["model"]), perturbation_ids=["P0_baseline"])
    res = shadow.shadow_screen(t["ideas"], t["ratings_long"], js)
    assert set(res["shadow_summary"].judge) == {"j0", "j1", "j2", "all judges"}
    assert (res["shadow_summary"].organization_advanced == int(t["ideas"].advanced.sum())).all()
