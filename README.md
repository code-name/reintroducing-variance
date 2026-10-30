# Reintroducing variance into idea evaluation

**An open instrument for trusting AI judges.**

Goran Calic, Hila Lifshitz, Alex Thabane, Vikram Arora and Matthew Zychowicz. Research artifact for the *Organization Science* special issue "AI-Enabled Frontiers in Organizational Science".

> **Status.** An early pilot of the tool. The four modules work end to end on the example folder and on your own data. The variance pipeline and the field test are prototypes. Tests run in CI on Python 3.11 to 3.13 with no data or API keys.

## What this is

Researchers and organizations now hand the rating of ideas to AI judges: large language models asked to score novelty, usefulness or creativity. People disagree about new ideas, and that disagreement carries information about which ideas are contested. The usual check on an AI judge, a single correlation with average human ratings, says nothing about whether the judge sees that disagreement.

This repository holds an instrument that checks an AI judge against human raters on four measures, a prototype pipeline that tries to put the human variance back into AI judgment, and a field test that runs an AI judge in shadow next to an organization's own evaluators.

## The instrument: four modules

1. **Rating platform** (`benchmark/rating/`). Human raters score ideas blind: they never see where an idea came from, any model's score or any other rater's score. The reference for an idea is the distribution of many independent ratings (mean and spread), never one adjudicated answer. Held-out ideas never have their scores released.
2. **AI judges** (`benchmark/judge.py`). The same ideas go to AI judges with the same rubric file the human raters read. Every call is cached with the exact prompt, model, provider and cost, so every response can be checked.
3. **Stability** (`benchmark/perturb.py`). An LLM rewrites the rating instructions in ways that should not matter, the instructions are also changed in other such ways (swapped dimension order, removed anchors, batching, worked examples, repeated sampling), and the change in the AI judges' scores is recorded. The rewordings are written by an LLM (`python -m benchmark.perturb reword`) and frozen into a versioned file that records the model and the call, so every later run uses the same text. The default grid, `perturbations_v1.2.yaml`, was written this way; for your own rubric, run the same command with `--rubric`.
4. **AI judge diagnostic** (`benchmark/agree.py`). One report per judge on four measures:
   - **variance**: does the judge know which ideas people disagree about? (its agreement with itself across repeated samples, and whether its spread tracks the human spread)
   - **agreement**: does it rank ideas in the same order as people?
   - **leniency**: does it score ideas lower or higher than people, and where does it sit among them?
   - **stability**: does it score the same way when the instructions change in ways that should not matter?

   Every report also says how much of the human disagreement could be measured at all. Disagreement is a property of a group of raters, and small groups measure it poorly: each idea's raters are split in two at random and the two halves' spreads are compared across ideas. The reliability that comes out puts a ceiling on the variance measure, and the diagnostic prints it next to every judge.

## The variance pipeline (`benchmark/variance.py`, prototype)

A pipeline that combines and prompts LLMs so that AI judgment shows which ideas people would disagree about. More disagreement is not the goal: a panel that disagrees about everything adds noise. A single AI judge can also pass, if it rates consistently and still flags the right ideas as contested, for example by predicting the spread of scores people would give. The pipeline:

1. measures human disagreement per idea and how reliably the raters pin it down;
2. builds candidates from different sources of disagreement: panels of different model families, a judge's repeated samples, prompt variants, judges rating from the view of a customer, an engineer or an investor, judges shown one human rater's past ratings, and judges asked for the spread of scores they expect from people;
3. searches over the candidates, choosing on some ideas;
4. judges the choice by how well it picks out the contested ideas (the top quarter of human disagreement) among the ideas it was not chosen on;
5. returns, for every idea, a mean, a spread and a contested flag, so a screen can send contested ideas to people instead of discarding them.

The first three sources reuse the stability runs. The other three need their own calls:

```bash
python -m benchmark.variance run --data-dir my_study/ --judges qwen3.8-27b,deepseek-v4-pro   # perspectives, rater twins, expected spread
python -m benchmark.variance --data-dir my_study/                                            # evaluate every source and test the pipeline
```

The pipeline needs many rated ideas to choose well; the example's 15 ideas only show the procedure.

## Field test: an AI judge in shadow (`benchmark/shadow.py`, prototype)

An organization that already screens submitted ideas keeps doing so. An AI judge scores every idea next to the usual evaluators, without affecting their decisions. Afterwards the field test counts how many of the ideas the organization advanced the AI judge would have dropped (if it had advanced the same number), and checks whether those are the ideas the evaluators disagreed about. It also writes a debrief sheet with the AI judges' scores, to show the evaluators after they have decided.

The data folder is the usual one, with the organization's decision in an `advanced` column of `ideas.csv` and the evaluators' own scores in `ratings_long.csv`. `examples/screen/` is an invented example:

```bash
python -m benchmark.judge run --data-dir examples/screen --backend fake --perturbations P0_baseline --judges qwen3.8-27b,deepseek-v4-pro
python -m benchmark.shadow --data-dir examples/screen

## Quick start

```bash
git clone https://github.com/code-name/reintroducing-variance && cd reintroducing-variance
python3.13 -m venv .venv && source .venv/bin/activate      # CI tests 3.11 to 3.13
pip install -r requirements.txt
pytest                                                     # no data or keys needed
python -m benchmark.agree                                  # the AI judge diagnostic on the example
python -m benchmark.variance                               # the variance pipeline on the example
python -m benchmark.figures                                # the diagnostic figure -> results/figure.png
```

The example folder, `examples/alternative_uses/`, holds 15 ideas, synthetic ratings from 39 raters and the scores of five AI judges, so every step runs without an API key. Its ratings are synthetic; use it to see how the tools work, not to estimate anything.

## Run it on your own ideas

Everything runs on your machine. Your ideas and ratings need not be shared with us or posted anywhere; the only thing that leaves your machine is the prompt sent to the model provider you choose (or nothing, with a local open-weight model behind an OpenAI-compatible endpoint).

Put three CSV files in a folder:

- `ideas.csv`: `idea_id, idea_text, task_id`, and optionally `source` (human or AI), `split`, `target_raters` (for ideas that need more raters than the rest) and `advanced` (the organization's decision, for the field test).
- `tasks.csv`: `task_id, task_prompt_verbatim`.
- `ratings_long.csv`: one row per idea, rater and dimension: `idea_id, rater_id, dimension, score, scale_min, scale_max`, with optional `condition`. Averaged scores are not enough; the diagnostic needs each rater's own scores.

Check the folder with `python -m benchmark.load my_study/`. Write your rubric as a YAML file like `configs/rubrics/novelty_usefulness_v1.0.yaml` (any dimensions, any scale). Then:

```bash
cp .env.example .env                                       # add OPENROUTER_API_KEY
python -m benchmark.judge plan --data-dir my_study/ --rubric my_rubric.yaml      # calls and cost, no network
python -m benchmark.judge run  --data-dir my_study/ --rubric my_rubric.yaml --judges qwen3.8-27b,deepseek-v4-pro
python -m benchmark.agree    --ratings my_study/ratings_long.csv --scores results/judge_runs/my_study/judge_scores.csv
python -m benchmark.variance --ratings my_study/ratings_long.csv --scores results/judge_runs/my_study/judge_scores.csv
```

To try the whole path without an API key, use the illustrative demo data and the offline fake judge:

```bash
python -m benchmark.judge run --data-dir examples/demo --backend fake --judges qwen3.8-27b,deepseek-v4-pro
python -m benchmark.agree --ratings examples/demo/ratings_long.csv --scores results/judge_runs/demo/judge_scores.csv
```

## Modules

| Module | What it does |
|---|---|
| `benchmark/rating/` | Rating platform (below) |
| `benchmark/judge.py` | Judge harness through OpenRouter or any OpenAI-compatible endpoint: rubric-rendered prompts, disk cache for every call, cost tracking and budget stop, your own data folder |
| `benchmark/perturb.py` | The stability grid, and LLM-written rewordings frozen into versioned files |
| `benchmark/agree.py` | Human ceiling, contested ideas, how reliably disagreement was measured, rater report, the agreement ladder and the AI judge diagnostic, drift |
| `benchmark/variance.py` | The variance pipeline (prototype) |
| `benchmark/shadow.py` | The field test: an AI judge in shadow next to an organization's evaluators (prototype) |
| `benchmark/load.py` | Reads and checks a data folder |
| `benchmark/figures.py` | The diagnostic figure |

## The rating platform

`benchmark/rating/` collects and stores ratings. It is a small self-hosted web app plus a command line, adapted from NSDA Corp's annotation apps (see `NOTICE`). Three rules distinguish it from an annotation tool:

- **The reference is an aggregate, not a resolution.** Each idea is rated independently by k raters (default 10); an idea's own `target_raters` overrides k, so a subset can be rated by more people (for example 40) and disagreement measured well. Nothing is adjudicated to one answer; the reference is each idea's mean, spread and standard error. The spread is kept as a measured property of the idea: how contested it is.
- **Raters rate blind.** The rating screen shows the task, the idea and the rubric. It never shows whether the idea came from a person or a model, any model score, or any other rater's score. Analytics are admin-only.
- **Held-out ideas stay held out.** Every idea has a split (`calibration` or `heldout`). A release export publishes calibration ratings and lists held-out ideas without their scores.

Try it locally with the illustrative demo data in `examples/demo/`:

```bash
python -m benchmark.rating.cli init-db
python -m benchmark.rating.cli seed-dev-users          # admin, rater1-3; localhost only
python -m benchmark.rating.cli add-rubric --file configs/rubrics/novelty_usefulness_v1.0.yaml
python -m benchmark.rating.cli create-dataset --name demo --rubric novelty_usefulness_v1.0 --target-raters 3
python -m benchmark.rating.cli import-tasks   --dataset demo --file examples/demo/tasks.csv
python -m benchmark.rating.cli import-ideas   --dataset demo --file examples/demo/ideas.csv
python -m benchmark.rating.cli import-ratings --dataset demo --file examples/demo/ratings_long.csv
python -m benchmark.rating.cli serve                   # http://127.0.0.1:8000, password in configs/rating_app.yaml
python -m benchmark.rating.cli export --dataset demo           # everything -> data/processed/demo/
python -m benchmark.rating.cli export --dataset demo --release # publishable -> data/release/demo/
```

Existing rater-level data loads with `import-ratings`; its raters become `external` users who cannot log in. AI judge scores load with `import-judge-scores` and are identified by the sha256 of their run configuration. The database lives in `data/app/` and is not tracked by git.

## Repository layout

```
reintroducing-variance/
├── README.md
├── LICENSE                   # MIT
├── NOTICE                    # credit for the NSDA Corp code release
├── requirements.txt          # pinned
├── pyproject.toml
├── .env.example              # API keys and endpoints, copied to .env (not tracked)
├── configs/
│   ├── rating_app.yaml       # rating app settings (no secrets)
│   ├── judges.yaml           # AI judges (OpenRouter model ids, prices) and the run plan
│   └── rubrics/              # versioned rubrics shared by human raters and AI judges
├── examples/
│   ├── alternative_uses/     # 15 ideas, synthetic ratings and five AI judges' scores
│   ├── screen/               # twelve invented ideas, evaluators and decisions for the field test
│   └── demo/                 # six invented ideas for trying the rating app
├── benchmark/
│   ├── rating/               # rating platform: models, sampler, imports, aggregates, exports, CLI, web app
│   ├── judge.py              # judge harness: OpenRouter, disk cache, cost and budget
│   ├── perturb.py            # the stability grid and LLM-written rewordings
│   ├── agree.py              # the AI judge diagnostic and the statistics behind it
│   ├── variance.py           # the variance pipeline (prototype)
│   ├── shadow.py             # the field test (prototype)
│   ├── load.py               # reads and checks a data folder
│   ├── figures.py            # the diagnostic figure
│   └── prompts/              # every prompt sent to a model, versioned, never edited after release
├── data/                     # working folders for your own data and the rating app; not tracked
├── results/                  # everything the tools write; not tracked
└── tests/
```

## Prompts, caching and reproducibility

- Every judge prompt lives in `benchmark/prompts/` as a versioned YAML file. A released version is never edited; changes go in a new file with a new version number and an entry in the file's own changelog.
- Every model call is cached in `results/cache/` under a key built from the backend, model, prompt id and version, perturbation, rendered prompt, temperature and sample index. Re-running is free; nothing is called twice.
- Two of the five configured judges are open-weight models whose weights are public, so judge results can be re-run without a commercial account. Model ids are pinned in `configs/judges.yaml`, and the provider and exact model that served each call are cached with it.
- Human raters and AI judges read the same rubric file, so the wording is identical.
- API keys come from environment variables only; `.env` is not tracked.

## License and citation

Code is released under the MIT License (see `LICENSE` and `NOTICE`). Citation details will be added on publication.
