"""Repository-level checks that run without data or API keys: prompts load and are versioned,
rendering keeps the JSON format intact, the perturbation set is consistent, the gitignore protects
participant data and secrets, and requirements are pinned."""
from pathlib import Path

import pytest
import yaml

from benchmark import PROMPTS_DIR, REPO_DIR
from benchmark.judge import PromptSpec, cache_key, load_prompt, render_prompt
from benchmark.perturb import load_perturbations

PROMPT_FILES = sorted(p.stem for p in PROMPTS_DIR.glob("*_v*.yaml") if not p.stem.startswith("perturbations"))


def test_prompt_files_exist():
    for name in ("pointwise_v1.1", "pointwise_batch_v1.1", "pairwise_v1.0"):
        assert name in PROMPT_FILES


@pytest.mark.parametrize("name", PROMPT_FILES)
def test_prompt_loads_and_is_versioned(name):
    spec = load_prompt(name)
    assert isinstance(spec, PromptSpec) and spec.name == name
    assert spec.status in {"draft", "released", "retired"}
    raw = yaml.safe_load(spec.path.read_text(encoding="utf-8"))
    assert raw["changelog"], "every prompt needs a changelog entry"
    assert "output_schema" in raw and raw["output_schema"].get("required")


def test_render_keeps_json_braces_and_rejects_bad_variables():
    spec = load_prompt("pointwise_v1.1")
    vals = dict(task_prompt="t", idea_text="A doorstop for a tent.", instructions="i", scale_min=1, scale_max=5,
                anchor_text="a", statement_block="s", json_format='{"novelty": <integer 1-5>}')
    _, user = render_prompt(spec, **vals)
    assert '{"novelty":' in user and "A doorstop for a tent." in user
    with pytest.raises(KeyError):
        render_prompt(spec, **vals, not_a_variable=1)
    with pytest.raises(KeyError):
        render_prompt(spec, task_prompt="x")


def test_cache_key_is_stable_and_sensitive():
    k1 = cache_key("openrouter", "m", "pointwise_v1.1", "P0_baseline:", "s", "u", 0.0, 0)
    assert k1 == cache_key("openrouter", "m", "pointwise_v1.1", "P0_baseline:", "s", "u", 0.0, 0)
    assert k1 != cache_key("openrouter", "m", "pointwise_v1.1", "P0_baseline:", "s", "u", 1.0, 0)
    assert len(k1) == 64


def test_perturbation_set_v11_is_the_stability_grid():
    raw = load_perturbations("perturbations_v1.1")
    ids = [e["id"] for e in raw["perturbations"]]
    assert ids == ["P0_baseline", "P1_rubric_reworded", "P2_dimension_order_swapped", "P3_anchors_removed",
                   "P4_batch_order_shuffled", "P5_few_shot", "P6_temperature_sampled"]
    assert "bias_probes" not in raw and raw["base_prompt"] == "pointwise_v1.1"


def test_modules_import():
    import benchmark.agree, benchmark.figures, benchmark.judge, benchmark.load, benchmark.perturb, benchmark.variance  # noqa: E401,F401


def test_gitignore_protects_participant_data_and_secrets():
    lines = (REPO_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    for needed in ("data/raw/", "data/processed/", "data/app/", "results/*", ".env"):
        assert needed in lines, needed


def test_requirements_are_pinned():
    for line in (REPO_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            assert "==" in line, f"unpinned requirement: {line}"
