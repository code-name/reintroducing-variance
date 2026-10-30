"""Prompt-perturbation generator: the stability grid.

``load_perturbations`` reads ``benchmark/prompts/perturbations_v<ver>.yaml``. ``rubric_values``
turns a rubric into the template variables of ``pointwise_v1.1`` (with a perturbation's rewording,
swapped order or removed anchors applied). ``plan_jobs`` expands a perturbation set over a set of
ideas into concrete jobs: one per prompt to send, carrying the template values, the ideas covered,
the temperature and the sample index. Rendering and sending happen in ``benchmark.judge``.

The default set is ``perturbations_v1.2``: its P1 rewordings were written by an LLM
(qwen/qwen3.8-27b, prompt ``reword_v1.0``) and frozen. ``perturbations_v1.1`` holds the
hand-written rewordings used in the first pilot runs. A rubric whose statements the set does not
cover cannot run P1 until an LLM has reworded it:

    python -m benchmark.perturb reword --rubric my_rubric.yaml --out perturbations_my_study

``generate_rewordings`` asks a model (any backend of ``benchmark.judge``) to reword the rubric's
statements without changing their meaning, and ``write_perturbation_set`` freezes the result into a
new versioned perturbation file, recording the generating model, prompt and date. Every later run
then uses the same text, so the rewording itself is reproducible even though a model wrote it.

    python -m benchmark.perturb reword --model qwen/qwen3.8-27b --n 3 --out perturbations_v1.2
    python -m benchmark.perturb reword --backend fake --out perturbations_demo   # offline try-out
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import PROMPTS_DIR


DEFAULT_SET = "perturbations_v1.2"


def load_perturbations(name: str = DEFAULT_SET, prompts_dir: Path = PROMPTS_DIR) -> dict[str, Any]:
    """Load and check a perturbation set: unique ids, a ``P0_baseline`` entry, and a
    ``base_prompt`` that exists."""
    path = prompts_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No perturbation file {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = list(raw.get("perturbations", [])) + list(raw.get("bias_probes", []))
    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate perturbation ids")
    if "P0_baseline" not in ids:
        raise ValueError(f"{path.name}: no P0_baseline entry")
    base = raw.get("base_prompt")
    if not base or not (prompts_dir / f"{base}.yaml").exists():
        raise ValueError(f"{path.name}: base_prompt '{base}' not found in {prompts_dir}")
    return raw


def rubric_values(rubric, *, statements: dict[str, str] | None = None, swap: bool = False,
                  anchors: bool = True) -> dict[str, str]:
    """Template variables from a rubric (``benchmark.rating.rubrics.RubricModel``)."""
    dims = list(rubric.dimensions)
    if swap:
        dims = dims[::-1]
    lines = []
    for k, d in enumerate(dims, start=1):
        text = (statements or {}).get(d.id, d.statement)
        lines.append(f"Statement {k} ({d.id}): {text}")
    if anchors and rubric.anchors:
        anchor_text = ", ".join(f"{v} = {rubric.anchors[v]}" for v in sorted(rubric.anchors))
    else:
        anchor_text = f"{rubric.scale_min} is the lowest and {rubric.scale_max} the highest"
    rng = f"<integer from {rubric.scale_min} to {rubric.scale_max}>"
    json_format = "{" + ", ".join(f'"{d.id}": {rng}' for d in dims) + ', "rationale": "<one sentence>"}'
    item_format = "{" + '"id": <number>, ' + ", ".join(f'"{d.id}": {rng}' for d in dims) + "}"
    return {"instructions": rubric.instructions.strip(), "scale_min": str(rubric.scale_min),
            "scale_max": str(rubric.scale_max), "anchor_text": anchor_text,
            "statement_block": "\n".join(lines), "json_format": json_format, "json_item_format": item_format}


def few_shot_examples(reference: pd.DataFrame, ideas: pd.DataFrame, dims: list[str], n: int = 3,
                      sort_dim: str = "novelty") -> tuple[str, list[str]]:
    """Lowest, middle and highest ideas on ``sort_dim`` by human mean, rendered as a worked-example
    block. Returns (block, idea_ids used)."""
    wide = reference.pivot_table(index="idea_id", columns="dimension", values="mean").dropna()
    wide = wide.sort_values(sort_dim)
    picks = [wide.index[0], wide.index[len(wide) // 2], wide.index[-1]][:n]
    text = ideas.set_index("idea_id").idea_text
    lines = ["Here are three ideas with the average scores that other raters gave them:"]
    for i in picks:
        scores = ", ".join(f'"{d}": {int(round(wide.loc[i, d]))}' for d in dims)
        lines.append(f'- "{text[i]}" -> {{{scores}}}')
    return "\n".join(lines) + "\n", list(picks)


def plan_jobs(perturbations: dict, ideas: pd.DataFrame, tasks: pd.DataFrame, rubric,
              ids: list[str] | None = None, reference: pd.DataFrame | None = None) -> list[dict]:
    """Expand the perturbation set over ``ideas`` into jobs.

    Each job: perturbation_id, variant, seed, sample_index, temperature, kind ("single" or
    "batch"), idea_ids, task_prompt, idea_text (single) or idea_list (batch), and template values.
    P5 is skipped when no human ``reference`` is given."""
    task_text = tasks.set_index("task_id").task_prompt_verbatim
    dims = [d.id for d in rubric.dimensions]
    jobs: list[dict] = []
    for p in perturbations["perturbations"]:
        pid = p["id"]
        if ids and pid not in ids:
            continue
        over = p.get("run_overrides", {})
        temp, n_samples = float(over.get("temperature", 0.0)), int(over.get("n_samples", 1))
        t = p.get("transform", "none")
        if t == "reword_rubric":
            missing = sorted({d for st in p["variants"].values() for d in dims if d not in st})
            if missing:
                raise ValueError(
                    f"{pid}: the rewordings in this perturbation set do not cover the rubric's {missing} "
                    f"statement(s). Have an LLM reword your rubric first: python -m benchmark.perturb reword "
                    f"--rubric <your rubric.yaml> --out perturbations_<name>, then pass --perturbation-set "
                    f"perturbations_<name> (or leave {pid} out with --perturbations).")
            variants = [(name, rubric_values(rubric, statements=st)) for name, st in p["variants"].items()]
        elif t == "swap_dimensions":
            variants = [("", rubric_values(rubric, swap=True))]
        elif t == "remove_anchors":
            variants = [("", rubric_values(rubric, anchors=False))]
        else:
            variants = [("", rubric_values(rubric))]
        if t == "batch_shuffle":
            prm = p.get("params", {})
            size, seeds = int(prm.get("batch_size", 10)), prm.get("seeds", [1])
            for seed in seeds:
                for task_id, grp in ideas.groupby("task_id"):
                    order = list(grp.idea_id)
                    random.Random(f"{seed}:{task_id}").shuffle(order)
                    for start in range(0, len(order), size):
                        chunk = order[start:start + size]
                        txt = grp.set_index("idea_id").idea_text
                        listing = "\n".join(f"[{k}] {txt[i]}" for k, i in enumerate(chunk, start=1))
                        jobs.append({"perturbation_id": pid, "variant": "", "seed": seed, "sample_index": 0,
                                     "temperature": temp, "kind": "batch", "idea_ids": chunk,
                                     "values": {**variants[0][1], "task_prompt": task_text[task_id], "idea_list": listing}})
            continue
        few_block, excluded = "", []
        if t == "few_shot":
            if reference is None:
                continue
            few_block, excluded = few_shot_examples(reference, ideas, dims, int(p.get("params", {}).get("n_examples", 3)))
        for variant, vals in variants:
            for row in ideas.itertuples():
                if row.idea_id in excluded:
                    continue
                for s in range(n_samples):
                    jobs.append({"perturbation_id": pid, "variant": variant, "seed": None, "sample_index": s,
                                 "temperature": temp, "kind": "single", "idea_ids": [row.idea_id],
                                 "values": {**vals, "task_prompt": task_text[row.task_id], "idea_text": row.idea_text,
                                            "few_shot_block": few_block}})
    return jobs


def describe(jobs: list[dict]) -> pd.DataFrame:
    """Jobs per perturbation and kind, for planning."""
    df = pd.DataFrame([{"perturbation_id": j["perturbation_id"], "kind": j["kind"], "ideas": len(j["idea_ids"])} for j in jobs])
    return df.groupby(["perturbation_id", "kind"]).agg(calls=("ideas", "size"), idea_scores=("ideas", "sum")).reset_index()


def dumps(job: dict) -> str:
    return json.dumps(job, sort_keys=True, ensure_ascii=False)


# --- Rewordings written by a model ---------------------------------------------------------

def generate_rewordings(rubric, backend, *, n: int = 3, cache=None, prompt_name: str = "reword_v1.0") -> tuple[list[dict[str, str]], dict]:
    """Ask ``backend`` for ``n`` rewordings of the rubric's statements. Returns (variants,
    provenance). Each variant maps dimension id to a statement. The call is cached like any judge
    call, so re-running with the same model and rubric returns the same text."""
    from .judge import DiskCache, cache_key, load_prompt, render_prompt

    spec = load_prompt(prompt_name)
    dims = [d.id for d in rubric.dimensions]
    values = {"instructions": rubric.instructions.strip(),
              "statement_block": "\n".join(f"{d.id}: {d.statement}" for d in rubric.dimensions),
              "n_variants": str(n),
              "json_format": "[" + ", ".join(['{' + ", ".join(f'"{d}": "<one sentence>"' for d in dims) + '}'] * min(n, 2))
                             + (", ..." if n > 2 else "") + "]"}
    system, user = render_prompt(spec, **values)
    temperature = float(spec.run_defaults.get("temperature", 0.7))
    cache = cache or DiskCache()
    key = cache_key(backend.name, backend.model, spec.name, "reword", system, user, temperature, 0)
    rec = cache.get(spec.name, key)
    if rec is None:
        rec = {"model": backend.model, "prompt": spec.name, "system": system, "user": user,
               **backend.complete(system, user, temperature=temperature, max_tokens=int(spec.run_defaults.get("max_tokens", 800)))}
        cache.put(spec.name, key, rec)
    variants = _parse_rewordings(rec["text"], dims)
    if len(variants) < n:
        raise ValueError(f"model returned {len(variants)} usable rewordings, needed {n}: {rec['text'][:300]}")
    prov = {"generator_backend": backend.name, "generator_model": rec.get("model_served") or backend.model,
            "generator_prompt": spec.name, "cache_key": key}
    return variants[:n], prov


def _parse_rewordings(text: str, dims: list[str]) -> list[dict[str, str]]:
    import re

    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and all(isinstance(it.get(d), str) and it[d].strip() for d in dims):
            out.append({d: it[d].strip() for d in dims})
    return out


def write_perturbation_set(variants: list[dict[str, str]], provenance: dict, *, out_name: str,
                           base: str = DEFAULT_SET, prompts_dir=None, date: str | None = None) -> Path:
    """Copy ``base`` and replace the P1 variants with the model-written rewordings. The new file
    records where the rewordings came from. Refuses to overwrite an existing file (released
    prompt files are never edited)."""
    from datetime import date as _date

    prompts_dir = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
    raw = load_perturbations(base, prompts_dir)
    out = prompts_dir / f"{out_name}.yaml"
    if out.exists():
        raise FileExistsError(f"{out} exists; released perturbation files are never edited")
    version = out_name.rsplit("_v", 1)[-1] if "_v" in out_name else out_name
    today = date or _date.today().isoformat()
    raw["version"], raw["status"], raw["created"] = version, "draft", today
    raw["changelog"] = [f"{version} ({today}): P1 rewordings written by {provenance['generator_model']} with "
                        f"{provenance['generator_prompt']} and frozen here; everything else as {base}."] + list(raw.get("changelog", []))
    for p in raw["perturbations"]:
        if p["id"] == "P1_rubric_reworded":
            p["description"] = "The dimension statements reworded by a model, with the same meaning. Each variant run separately."
            p["variants"] = {chr(97 + k): v for k, v in enumerate(variants)}
            p["generated_by"] = provenance
    out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from . import REPO_DIR
    from .judge import DiskCache, OpenRouterBackend, load_env, load_judges, load_rubric

    p = argparse.ArgumentParser(prog="python -m benchmark.perturb")
    p.add_argument("command", choices=["reword"])
    p.add_argument("--model", default="qwen/qwen3.8-27b", help="OpenRouter model id that writes the rewordings")
    p.add_argument("--backend", choices=["openrouter", "fake"], default="openrouter")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--rubric", default=None, help="rubric YAML (default: configs/judges.yaml)")
    p.add_argument("--base", default=DEFAULT_SET)
    p.add_argument("--out", required=True, help="new perturbation file name, e.g. perturbations_v1.2")
    a = p.parse_args(argv)
    load_env()
    rubric, _ = load_rubric(Path(a.rubric) if a.rubric else REPO_DIR / load_judges()["rubric"])
    if a.backend == "fake":
        backend, cache = _FakeReworder(), DiskCache(REPO_DIR / "results" / "cache_fake")
    else:
        backend, cache = OpenRouterBackend(a.model, reasoning_effort="low"), None
    variants, prov = generate_rewordings(rubric, backend, n=a.n, cache=cache)
    print(write_perturbation_set(variants, prov, out_name=a.out, base=a.base))
    for v in variants:
        print(" ", v)
    return 0


class _FakeReworder:
    """Offline stand-in for trying the reword command and for tests."""

    name, model = "fake", "fake-reworder"

    def complete(self, system: str, user: str, *, temperature: float, max_tokens: int) -> dict:
        import re

        dims = re.findall(r"^(\w+): ", user, re.M)
        n = int(re.search(r"Write (\d+) different", user).group(1))
        items = [{d: f"(fake rewording {k + 1}) The idea is {d}." for d in dims} for k in range(n)]
        return {"text": json.dumps(items), "usage": {}, "cost_usd": 0.0, "provider": "fake",
                "model_served": self.model, "temperature_dropped": False}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
