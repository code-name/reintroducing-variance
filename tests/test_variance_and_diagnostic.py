"""The AI judge diagnostic, the bring-your-own-data path, model-written rewordings and the variance
pipeline. Everything runs offline on the example folder and the demo data."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark import PROMPTS_DIR, REPO_DIR, agree, variance
from benchmark.judge import DiskCache, FakeBackend, dataset_tables, load_rubric, run
from benchmark.perturb import _FakeReworder, generate_rewordings, load_perturbations, write_perturbation_set

EXAMPLE = REPO_DIR / "examples" / "alternative_uses"
DEMO = REPO_DIR / "examples" / "demo"
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
    for col in ("agreement", "leniency", "stability_mean_abs_change", "variance_self_icc", "variance_tracks_human_sd"):
        assert col in d and d[col].notna().any(), col
    assert d.leniency_percentile.between(0, 100).all()


def test_diagnose_runs_on_demo_data_without_conditions(tmp_path):
    ratings = pd.read_csv(DEMO / "ratings_long.csv").drop(columns="condition")
    js = run(str(DEMO), JUDGES[:2], rubric_path=RUBRIC, cache=DiskCache(tmp_path / "c"), out_dir=tmp_path / "o",
             backend_factory=lambda j: NoisyFake(j["model"]), perturbation_ids=["P0_baseline", "P6_temperature_sampled"])
    res = agree.diagnose(ratings, js)
    assert {"human_ceiling", "idea_disagreement", "agreement_ladder", "judge_diagnostics"} <= set(res)
    assert len(res["judge_diagnostics"]) == 4


def test_data_dir_is_a_dataset():
    ideas, tasks, ref = dataset_tables(str(DEMO))
    assert len(ideas) == 6 and len(tasks) == 2 and ref is not None


def test_reword_writes_a_new_versioned_file(tmp_path):
    for f in ("perturbations_v1.1.yaml", "pointwise_v1.1.yaml"):
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
    assert (rel.ceiling_for_panel_spearman.dropna().between(0, 1)).all()
    cv = res["panel_search_cross_validated"]
    assert cv.splits.iloc[0] == 200


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
