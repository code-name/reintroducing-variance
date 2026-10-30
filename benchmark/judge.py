"""Judge harness: score ideas with LLM judges through OpenRouter, caching every call to disk.

Prompts are versioned YAML (``benchmark/prompts/``) rendered from the same rubric the human raters
see (``configs/rubrics/``). Judges are listed in ``configs/judges.yaml``. Every call is cached in
``results/cache/<prompt>/<key>.json`` before it is used; the key hashes everything that could
change the answer (backend, model, prompt, perturbation, rendered text, temperature, sample index),
so re-running costs nothing and no call is ever made twice. Spend is tracked per call from the
provider's reported cost and the run stops at ``JUDGE_BUDGET_USD``.

    python -m benchmark.judge plan                     # calls and estimated cost, no network
    python -m benchmark.judge run --data-dir my_study/ --rubric my_rubric.yaml --judges qwen3.8-27b
                                                       # your own ideas, tasks, ratings and rubric
    python -m benchmark.judge run --data-dir examples/demo --backend fake   # try it with no key

A data directory holds ``ideas.csv`` (idea_id, idea_text, task_id), ``tasks.csv`` (task_id,
task_prompt_verbatim) and, optionally, ``ratings_long.csv`` (idea_id, rater_id, dimension, score;
needed only for the worked-example perturbation and for the diagnostic). Nothing is uploaded
anywhere except the prompts sent to the model provider you choose.

Output: ``results/judge_runs/<dataset>/judge_scores.csv`` (judge, perturbation_id, variant, seed,
sample_index, idea_id, dimension, score, cache_key) and ``cost_log.csv``. The scores file loads
into the rating app with ``python -m benchmark.rating.cli import-judge-scores``.

The API key comes from the environment only (``OPENROUTER_API_KEY``, or ``.env``); it is never
written to the cache, the logs or the repository.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import yaml

from . import CACHE_DIR, PROMPTS_DIR, REPO_DIR

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class PromptSpec:
    """A loaded, versioned judge prompt."""

    id: str
    version: str
    status: str
    system: str
    user: str
    variables: tuple[str, ...]
    defaults: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    run_defaults: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def name(self) -> str:
        return f"{self.id}_v{self.version}"


def load_prompt(name: str, prompts_dir: Path = PROMPTS_DIR) -> PromptSpec:
    """Load ``<name>.yaml`` (e.g. ``pointwise_v1.0``) from the prompts directory.

    Raises ``FileNotFoundError`` for an unknown name and ``ValueError`` if required keys are
    missing or the version in the file does not match the filename.
    """
    path = prompts_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No prompt file {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("id", "version", "status", "system", "user", "variables"):
        if key not in raw:
            raise ValueError(f"{path.name}: missing required key '{key}'")
    expected = f"{raw['id']}_v{raw['version']}"
    if expected != name:
        raise ValueError(f"{path.name}: id/version '{expected}' does not match filename '{name}'")
    return PromptSpec(
        id=raw["id"],
        version=str(raw["version"]),
        status=raw["status"],
        system=raw["system"],
        user=raw["user"],
        variables=tuple(raw["variables"]),
        defaults=dict(raw.get("defaults", {})),
        output_schema=dict(raw.get("output_schema", {})),
        run_defaults=dict(raw.get("run_defaults", {})),
        path=path,
    )


def render_prompt(spec: PromptSpec, **values: Any) -> tuple[str, str]:
    """Substitute declared variables into the system and user templates.

    Only names listed in ``spec.variables`` are substituted, so literal braces in the template
    (for example the JSON reply format) are left untouched. Unknown names raise ``KeyError``;
    declared variables without a value or default raise ``KeyError`` as well. Returns
    ``(system_text, user_text)``.
    """
    unknown = set(values) - set(spec.variables)
    if unknown:
        raise KeyError(f"Values given for undeclared variables: {sorted(unknown)}")
    merged = {**spec.defaults, **values}
    missing = [v for v in spec.variables if v not in merged]
    if missing:
        raise KeyError(f"Missing values for variables: {missing}")

    def sub(template: str) -> str:
        return _PLACEHOLDER.sub(
            lambda m: str(merged[m.group(1)]) if m.group(1) in merged else m.group(0), template
        )

    return sub(spec.system), sub(spec.user)


def cache_key(
    backend_name: str,
    model_version: str,
    prompt_name: str,
    perturbation_id: str,
    system_text: str,
    user_text: str,
    temperature: float,
    sample_index: int,
) -> str:
    """SHA-256 over everything that could change a model's answer. Stable across runs."""
    payload = json.dumps(
        {
            "backend": backend_name,
            "model_version": model_version,
            "prompt": prompt_name,
            "perturbation": perturbation_id,
            "system": system_text,
            "user": user_text,
            "temperature": temperature,
            "sample_index": sample_index,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeBackend(Protocol):
    """What a backend must provide."""

    name: str
    model: str

    def complete(self, system: str, user: str, *, temperature: float, max_tokens: int) -> dict[str, Any]:
        """Return {"text", "usage", "cost_usd", "provider", "model_served", "temperature_dropped"}."""
        ...


class OpenRouterBackend:
    """OpenAI-compatible chat completions at OpenRouter. Records the provider and exact model that
    served each call and the cost OpenRouter reports."""

    name = "openrouter"

    def __init__(self, model: str, *, api_key: str | None = None, base_url: str = "https://openrouter.ai/api/v1",
                 reasoning_effort: str | None = "low", timeout: float = 120.0) -> None:
        import os

        from openai import OpenAI

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env or the environment)")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)

    def complete(self, system: str, user: str, *, temperature: float, max_tokens: int) -> dict[str, Any]:
        from openai import BadRequestError
        from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

        extra: dict[str, Any] = {"usage": {"include": True}}
        if self.reasoning_effort:
            extra["reasoning"] = {"effort": self.reasoning_effort, "exclude": True}
        dropped = False
        limit = max_tokens

        @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60),
               retry=retry_if_not_exception_type(BadRequestError), reraise=True)
        def call(with_temperature: bool):
            kwargs: dict[str, Any] = {"model": self.model, "max_tokens": limit, "extra_body": extra,
                                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if with_temperature:
                kwargs["temperature"] = temperature
            return self.client.chat.completions.create(**kwargs)

        try:
            resp = call(True)
        except BadRequestError as exc:  # some reasoning models reject temperature
            if "temperature" not in str(exc).lower():
                raise
            resp, dropped = call(False), True
        spent = 0.0
        choice = resp.choices[0] if resp.choices else None
        if choice is None or not choice.message.content or choice.finish_reason == "length":
            # A reasoning model can use up the limit before it replies: try once with more room.
            spent = float((resp.usage.model_dump() if resp.usage else {}).get("cost") or 0.0)
            limit = max_tokens * 4
            resp = call(not dropped)
        usage = resp.usage.model_dump() if resp.usage else {}
        return {"text": (resp.choices[0].message.content or "") if resp.choices else "", "usage": usage,
                "cost_usd": spent + float(usage.get("cost") or 0.0), "provider": getattr(resp, "provider", None),
                "model_served": resp.model, "temperature_dropped": dropped}


class FakeBackend:
    """Deterministic offline backend for tests and try-outs: scores are a hash of the model and the
    prompt, so different fake judges differ. It reads the dimensions from the reply format in the
    prompt, so it works with any rubric, and it answers the expected-spread prompt with counts."""

    name = "fake"

    def __init__(self, model: str = "fake-judge", dims: tuple[str, ...] = ("novelty", "usefulness")) -> None:
        self.model, self.dims = model, dims

    def complete(self, system: str, user: str, *, temperature: float, max_tokens: int) -> dict[str, Any]:
        h = hashlib.sha256((self.model + user + str(temperature)).encode()).digest()
        spread_dims = re.findall(r'"(\w+)": \{"\d+": <count>', user)
        dims = re.findall(r'"(\w+)": <integer', user) or list(self.dims)
        dims = list(dict.fromkeys(d for d in dims if d != "id"))
        if spread_dims:  # expected-spread prompt: ten raters over the scale points
            points = re.findall(r'"(\d+)": <count>', user)
            points = list(dict.fromkeys(points))
            out = {}
            for i, d in enumerate(dict.fromkeys(spread_dims)):
                counts = [h[(i * 7 + j) % len(h)] % 4 for j in range(len(points))]
                total = sum(counts) or 1
                scaled = [round(10 * c / total) for c in counts]
                scaled[-1] += 10 - sum(scaled)
                out[d] = {p: max(0, c) for p, c in zip(points, scaled)}
            text = json.dumps(out)
        elif "[1]" in user:  # batch prompt
            n = user.count("\n[") + 1
            items = [{"id": k, **{d: 1 + h[(k * 2 + i) % len(h)] % 5 for i, d in enumerate(dims)}} for k in range(1, n + 1)]
            text = json.dumps(items)
        else:
            text = json.dumps({**{d: 1 + h[i] % 5 for i, d in enumerate(dims)}, "rationale": "fake"})
        return {"text": text, "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": 30}, "cost_usd": 0.0,
                "provider": "fake", "model_served": self.model, "temperature_dropped": False}


class DiskCache:
    """One JSON file per call under ``root/<prompt_name>/<key>.json``, written atomically."""

    def __init__(self, root: Path = CACHE_DIR) -> None:
        self.root = Path(root)

    def _path(self, prompt_name: str, key: str) -> Path:
        return self.root / prompt_name / f"{key}.json"

    def get(self, prompt_name: str, key: str) -> dict[str, Any] | None:
        p = self._path(prompt_name, key)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, prompt_name: str, key: str, record: dict[str, Any]) -> None:
        p = self._path(prompt_name, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        import uuid

        tmp = p.with_name(f"{p.stem}.{uuid.uuid4().hex}.tmp")  # unique: two threads may write the same call
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)


_JSON_OBJ = re.compile(r"\{.*\}", re.S)
_JSON_ARR = re.compile(r"\[.*\]", re.S)


def parse_scores(text: str, dims: list[str], scale: tuple[int, int]) -> dict[str, float] | None:
    """Scores from a pointwise reply; None if any dimension is missing or off the scale."""
    m = _JSON_OBJ.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for d in dims:
        v = obj.get(d)
        if isinstance(v, str) and v.strip().lstrip("-").isdigit():
            v = int(v)
        if not isinstance(v, (int, float)) or not scale[0] <= v <= scale[1]:
            return None
        out[d] = float(v)
    return out


def parse_batch(text: str, dims: list[str], scale: tuple[int, int], n: int) -> dict[int, dict[str, float]]:
    """Scores from a batch reply keyed by 1-based position; items that fail to parse are omitted."""
    m = _JSON_ARR.search(text or "")
    if not m:
        return {}
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out: dict[int, dict[str, float]] = {}
    for item in arr if isinstance(arr, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int) or not 1 <= item["id"] <= n:
            continue
        s = parse_scores(json.dumps(item), dims, scale)
        if s is not None:
            out[item["id"]] = s
    return out


# --- Configuration ---------------------------------------------------------------------

def load_judges(path: Path = REPO_DIR / "configs" / "judges.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_rubric(path: Path):
    from .rating.rubrics import parse_rubric

    model, _, _, sha = parse_rubric(Path(path).read_text(encoding="utf-8"))
    return model, sha


def load_env(path: Path = REPO_DIR / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (existing variables win)."""
    import os

    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# --- Running ---------------------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    pass


def dataset_tables(dataset: str) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame | None"]:
    """(ideas, tasks, human reference or None) for a data folder: ideas.csv, tasks.csv and
    optionally ratings_long.csv. A relative path is looked up from the working directory, then
    from the repository root (so ``examples/alternative_uses`` works from anywhere)."""
    import pandas as pd

    from .agree import reference

    d = Path(dataset)
    if not d.is_dir() and (REPO_DIR / d).is_dir():
        d = REPO_DIR / d
    if not d.is_dir():
        raise FileNotFoundError(f"no data folder at {dataset}")
    ideas, tasks = pd.read_csv(d / "ideas.csv"), pd.read_csv(d / "tasks.csv")
    if not (d / "ratings_long.csv").exists():
        return ideas, tasks, None
    ratings = pd.read_csv(d / "ratings_long.csv")
    conds = {"baseline"} if "condition" in ratings and "baseline" in set(ratings.condition) else None
    return ideas, tasks, reference(ratings, conds)


def run(dataset: str, judges: list[dict], *, rubric_path: Path, perturbation_set: str = "perturbations_v1.2",
        perturbation_ids: list[str] | None = None, out_dir: Path | None = None, cache: DiskCache | None = None,
        budget_usd: float = 50.0, dry_run: bool = False, backend_factory=None, max_workers: int = 8,
        write: bool = False, jobs: list[dict] | None = None) -> "pd.DataFrame":
    """Score every job for every judge. Cached calls are reused. Returns the scores table.
    ``dry_run`` makes no calls (cached results only); with ``write`` it still writes the scores
    file, which is how parallel per-judge runs are assembled into one table. ``jobs`` replaces the
    perturbation grid with jobs built elsewhere (``benchmark.variance`` uses this)."""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    import pandas as pd

    from . import RESULTS_DIR
    from .perturb import load_perturbations, plan_jobs

    rubric, rubric_sha = load_rubric(rubric_path)
    dims = [d.id for d in rubric.dimensions]
    scale = (rubric.scale_min, rubric.scale_max)
    ideas, tasks, ref = dataset_tables(dataset)
    pert = load_perturbations(perturbation_set)
    if jobs is None:
        jobs = plan_jobs(pert, ideas, tasks, rubric, perturbation_ids, ref)
    single = load_prompt(pert["base_prompt"])
    batch = load_prompt("pointwise_batch_v1.1")
    cache = cache or DiskCache()
    out_dir = out_dir or RESULTS_DIR / "judge_runs" / Path(dataset).name
    spent, lock = [0.0], Lock()
    rows, costs = [], []

    def do(judge: dict, job: dict, backend) -> None:
        spec = batch if job["kind"] == "batch" else single
        system, user = render_prompt(spec, **{k: v for k, v in job["values"].items() if k in spec.variables})
        key = cache_key("openrouter", judge["model"], spec.name, f"{job['perturbation_id']}:{job['variant']}",
                        system, user, job["temperature"], job["sample_index"])
        rec = cache.get(spec.name, key)
        if rec is None:
            if dry_run:
                return
            with lock:
                if spent[0] >= budget_usd:
                    raise BudgetExceeded(f"budget {budget_usd} USD reached")
            max_tokens = int(judge.get("max_tokens", spec.run_defaults.get("max_tokens", 800)))
            res = backend.complete(system, user, temperature=job["temperature"], max_tokens=max_tokens)
            rec = {"judge": judge["label"], "model": judge["model"], "prompt": spec.name, "rubric_sha256": rubric_sha,
                   "perturbation_id": job["perturbation_id"], "variant": job["variant"], "seed": job["seed"],
                   "sample_index": job["sample_index"], "temperature": job["temperature"], "idea_ids": job["idea_ids"],
                   "system": system, "user": user, **res, "created_at": _now()}
            cache.put(spec.name, key, rec)
            with lock:
                spent[0] += rec.get("cost_usd") or 0.0
                costs.append({"judge": judge["label"], "key": key, "cost_usd": rec.get("cost_usd"),
                              "prompt_tokens": (rec.get("usage") or {}).get("prompt_tokens"),
                              "completion_tokens": (rec.get("usage") or {}).get("completion_tokens")})
        if job["kind"] == "batch":
            parsed = parse_batch(rec["text"], dims, scale, len(job["idea_ids"]))
            pairs = [(iid, parsed.get(k + 1)) for k, iid in enumerate(job["idea_ids"])]
        else:
            pairs = [(job["idea_ids"][0], parse_scores(rec["text"], dims, scale))]
        with lock:
            for iid, sc in pairs:
                for d in dims:
                    rows.append({"judge": judge["label"], "model": judge["model"], "perturbation_id": job["perturbation_id"],
                                 "variant": job["variant"], "seed": job["seed"], "sample_index": job["sample_index"],
                                 "temperature": job["temperature"], "idea_id": iid, "dimension": d,
                                 "score": None if sc is None else sc[d], "scale_min": scale[0], "scale_max": scale[1],
                                 "cache_key": key})

    for judge in judges:
        if dry_run:
            backend = None
        elif backend_factory is not None:
            backend = backend_factory(judge)
        else:
            backend = OpenRouterBackend(judge["model"], reasoning_effort=judge.get("reasoning_effort"))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(do, judge, job, backend) for job in jobs]
            for f in as_completed(futures):
                f.result()
    df = pd.DataFrame(rows)
    if write or not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(df):
            df.sort_values(["judge", "perturbation_id", "variant", "seed", "sample_index", "idea_id", "dimension"],
                           na_position="first").to_csv(out_dir / "judge_scores.csv", index=False)
        if costs:
            log = out_dir / "cost_log.csv"
            pd.DataFrame(costs).to_csv(log, mode="a", header=not log.exists(), index=False)
    return df


def plan(dataset: str, judges: list[dict], *, rubric_path: Path, perturbation_set: str = "perturbations_v1.2",
         perturbation_ids: list[str] | None = None, out_tokens: int = 400) -> "pd.DataFrame":
    """Calls and an upper-bound cost estimate per judge, from rendered prompt length and the
    prices in configs/judges.yaml. No network."""
    import pandas as pd

    from .perturb import load_perturbations, plan_jobs

    rubric, _ = load_rubric(rubric_path)
    ideas, tasks, ref = dataset_tables(dataset)
    pert = load_perturbations(perturbation_set)
    jobs = plan_jobs(pert, ideas, tasks, rubric, perturbation_ids, ref)
    single, batch = load_prompt(pert["base_prompt"]), load_prompt("pointwise_batch_v1.1")
    in_tok = 0
    for j in jobs:
        spec = batch if j["kind"] == "batch" else single
        s, u = render_prompt(spec, **{k: v for k, v in j["values"].items() if k in spec.variables})
        in_tok += (len(s) + len(u)) // 4
    rows = []
    for judge in judges:
        p_in, p_out = judge.get("price_in_per_m", 0), judge.get("price_out_per_m", 0)
        rows.append({"dataset": dataset, "judge": judge["label"], "calls": len(jobs), "input_tokens": in_tok,
                     "est_cost_usd": round(in_tok / 1e6 * p_in + len(jobs) * out_tokens / 1e6 * p_out, 2)})
    return pd.DataFrame(rows)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    p = argparse.ArgumentParser(prog="python -m benchmark.judge")
    p.add_argument("command", choices=["plan", "run"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--judges", default=None, help="comma-separated labels from configs/judges.yaml")
    p.add_argument("--perturbations", default=None, help="comma-separated perturbation ids")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--assemble", action="store_true", help="no calls; write the scores table from the cache")
    p.add_argument("--out-suffix", default="", help="write to results/judge_runs/<dataset><suffix>/ (for per-judge parallel runs)")
    p.add_argument("--data-dir", default=None, help="your own data: a folder with ideas.csv, tasks.csv, ratings_long.csv")
    p.add_argument("--rubric", default=None, help="rubric YAML (default: the one in configs/judges.yaml)")
    p.add_argument("--perturbation-set", default="perturbations_v1.2",
                   help="perturbation file in benchmark/prompts/ (v1.2: LLM-written rewordings; v1.1: the hand-written ones of the first pilot runs)")
    p.add_argument("--backend", choices=["openrouter", "fake"], default="openrouter",
                   help="fake = deterministic offline scores, for trying the pipeline without a key")
    a = p.parse_args(argv)
    if a.data_dir:
        a.dataset = a.data_dir
    load_env()
    cfg = load_judges()
    judges = [j for j in cfg["judges"] if not a.judges or j["label"] in a.judges.split(",")]
    runs = cfg["runs"] if a.dataset is None else [r for r in cfg["runs"] if r["dataset"] == a.dataset]
    if a.dataset is not None and not runs:  # a folder not in the plan: the full grid
        runs = [{"dataset": a.dataset, "perturbations": None}]
    rubric_path = Path(a.rubric) if a.rubric else REPO_DIR / cfg["rubric"]
    if a.command == "plan":
        import pandas as pd

        frames = [plan(r["dataset"], judges, rubric_path=rubric_path, perturbation_set=a.perturbation_set,
                       perturbation_ids=(a.perturbations.split(",") if a.perturbations else r.get("perturbations"))) for r in runs]
        df = pd.concat(frames, ignore_index=True)
        print(df.to_string(index=False))
        print(f"\nTotal calls {df.calls.sum():,}; estimated upper-bound cost {df.est_cost_usd.sum():.2f} USD")
        return 0
    budget = float(os.environ.get("JUDGE_BUDGET_USD", cfg.get("budget_usd", 50)))
    for r in runs:
        from . import RESULTS_DIR

        factory = (lambda j: FakeBackend(j["label"])) if a.backend == "fake" else None
        df = run(r["dataset"], judges, rubric_path=rubric_path, dry_run=a.dry_run or a.assemble, write=a.assemble,
                 perturbation_set=a.perturbation_set, backend_factory=factory,
                 cache=DiskCache(RESULTS_DIR / "cache_fake") if a.backend == "fake" else None,
                 budget_usd=budget, out_dir=RESULTS_DIR / "judge_runs" / f"{Path(r['dataset']).name}{a.out_suffix}",
                 perturbation_ids=(a.perturbations.split(",") if a.perturbations else r.get("perturbations")))
        missing = int(df.score.isna().sum()) if len(df) else 0
        print(f"{r['dataset']}: {len(df):,} score rows, {missing:,} unparsed{' (from cache only)' if a.dry_run or a.assemble else ''}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
