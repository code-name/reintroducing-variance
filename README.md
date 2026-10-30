# Reintroducing variance into idea evaluation

**An open instrument for trusting AI judges.** Benchmarks LLMs that rate ideas against human raters, including whether a model can identify which ideas people disagree about.

Goran Calic, Hila Lifshitz, Alex Thabane, Vikram Arora and Matthew Zychowicz. Research artifact for the *Organization Science* special issue "AI-Enabled Frontiers in Organizational Science".

> **Status:** early pilot. The variance pipeline and the field test are prototypes.

## Getting started

Requires Python 3.11, 3.12 or 3.13.

**Windows (PowerShell)**

```powershell
git clone https://github.com/code-name/reintroducing-variance.git
cd reintroducing-variance
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Mac or Linux**

```bash
git clone https://github.com/code-name/reintroducing-variance.git
cd reintroducing-variance
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the bundled example (no API key):

```bash
python -m benchmark.agree      # the AI judge diagnostic
python -m benchmark.variance   # the variance pipeline
python -m benchmark.figures    # the diagnostic figure, saved to results/figure.png
pytest                         # the tests
```

`examples/alternative_uses/` has 15 ideas, synthetic ratings from 39 raters and saved scores from five AI judges. Use it to test the tools, not to estimate anything. `benchmark.variance` may print a `ConstantInputWarning` on this small sample; it is safe to ignore.

Start the rating app (no API key):

```bash
python -m benchmark.rating.cli init-db          # first time only
python -m benchmark.rating.cli seed-dev-users   # test logins, localhost only
python -m benchmark.rating.cli serve
```

Open http://127.0.0.1:8000 and log in as `admin` or `rater1`. The password is in `configs/rating_app.yaml`. The app starts empty; to load data, see [The rating platform](#the-rating-platform).

## When you need an API key

An [OpenRouter](https://openrouter.ai) key is required only for commands that call a model. One key covers all five judges.

- `python -m benchmark.judge run`
- `python -m benchmark.variance run`
- `python -m benchmark.perturb reword`

Add `--backend fake` to run any of them offline. Scores are fake, so use this only to check your setup.

## Run it on your own ideas

### 1. Prepare your data

Put three CSV files in a folder, such as `my_study/`:

| File | Columns |
|---|---|
| `ideas.csv` | `idea_id, idea_text, task_id`. Optional: `source` (human or AI), `split`, `target_raters`, `advanced` |
| `tasks.csv` | `task_id, task_prompt_verbatim` |
| `ratings_long.csv` | One row per idea, rater and dimension: `idea_id, rater_id, dimension, score, scale_min, scale_max`. Optional: `condition` |

Rater-level scores are required; averages are not enough.

### 2. Write your rubric

A rubric is a YAML file read by both human raters and AI judges. Copy `configs/rubrics/novelty_usefulness_v1.0.yaml` and edit it. Any dimensions and any scale work.

### 3. Run the commands

```powershell
# Check your data
python -m benchmark.load my_study/

# Copy the settings file, then open .env and paste in your API key
copy .env.example .env                  # Mac or Linux: cp .env.example .env

# Have an AI reword your rubric
python -m benchmark.perturb reword --rubric my_rubric.yaml --out perturbations_my_study

# See the calls and cost for all five judges
python -m benchmark.judge plan --data-dir my_study/ --rubric my_rubric.yaml --perturbation-set perturbations_my_study

# Score the ideas
python -m benchmark.judge run --data-dir my_study/ --rubric my_rubric.yaml --perturbation-set perturbations_my_study --judges qwen3.8-27b,deepseek-v4-pro

# Run the diagnostic
python -m benchmark.agree --ratings my_study/ratings_long.csv --scores results/judge_runs/my_study/judge_scores.csv

# Run the variance pipeline (optional)
python -m benchmark.variance run --data-dir my_study/ --rubric my_rubric.yaml --judges qwen3.8-27b,deepseek-v4-pro
python -m benchmark.variance --data-dir my_study/
```

**Always reword a new rubric.** The run fails if a dimension has no rewordings. If you keep the dimension names but change their wording, it does not fail: it silently uses the old rewordings, which invalidates the P1 stability test.

- **Rewordings** are generated once and frozen to a versioned file, so every run uses the same text.
- **Budget:** `JUDGE_BUDGET_USD` in `.env` sets a hard spend limit.
- **Caching:** every call is cached, so reruns do not pay twice.
- **Outputs** go to `results/`.
- **Privacy:** see [Data privacy](#data-privacy) before sending confidential ideas.

## How the diagnostic works

`benchmark.agree` writes one report per AI judge on four measures:

| Measure | Question |
|---|---|
| **Variance** | Does the judge know which ideas people disagree about? |
| **Agreement** | Does it rank ideas in the same order as people? |
| **Leniency** | Does it score ideas higher or lower than people? |
| **Stability** | Does it score the same way when the instructions change in ways that should not matter? |

Each report also gives the reliability of the human disagreement measure, which caps the variance score.

Stability runs each judge under small prompt changes: reworded statements, swapped dimension order, removed scale labels, batching and worked examples.

## The variance pipeline (prototype)

`benchmark/variance.py` searches for a way to make AI judgment flag the ideas people would disagree about. The target is the right ideas, not more disagreement.

- **Candidates** come from six sources: panels of different models, repeated samples, prompt variants, judges rating as a customer, engineer or investor, judges shown one rater's past ratings, and judges asked to predict how people would split.
- **Selection:** the best candidate is chosen on some ideas and tested on unseen ones.
- **Output:** a mean, a spread and a contested flag per idea, so contested ideas can go to people.

Selection needs many rated ideas. The example's 15 ideas only demonstrate the steps.

## The field test (prototype)

`benchmark/shadow.py` compares an AI judge with an organization's evaluators without affecting their decisions. It reports how many advanced ideas the judge would have dropped, whether those are the ideas the evaluators disagreed about, and a debrief sheet to share after decisions are made.

Input is a standard data folder with the decisions in an `advanced` column of `ideas.csv`. Run the invented example:

```bash
python -m benchmark.judge run --data-dir examples/screen --backend fake --perturbations P0_baseline --judges qwen3.8-27b,deepseek-v4-pro
python -m benchmark.shadow --data-dir examples/screen
```

For real data, use your own folder and leave out `--backend fake`.

## The rating platform

`benchmark/rating/` is a self-hosted web app and CLI for collecting human ratings, adapted from NSDA Corp's annotation apps (see `NOTICE`).

- **Many raters per idea.** Each idea gets 10 raters by default, and `target_raters` can set more. The reference is each idea's mean, spread and standard error, not one agreed answer.
- **Raters rate blind.** They never see where an idea came from, any model's score or another rater's score.
- **Held-out ideas stay held out.** A release export lists them without their scores.

Run it with the invented data in `examples/demo/`:

```bash
python -m benchmark.rating.cli init-db
python -m benchmark.rating.cli seed-dev-users          # admin, rater1-3; localhost only
python -m benchmark.rating.cli add-rubric --file configs/rubrics/novelty_usefulness_v1.0.yaml
python -m benchmark.rating.cli create-dataset --name demo --rubric novelty_usefulness_v1.0 --target-raters 3
python -m benchmark.rating.cli import-tasks   --dataset demo --file examples/demo/tasks.csv
python -m benchmark.rating.cli import-ideas   --dataset demo --file examples/demo/ideas.csv
python -m benchmark.rating.cli import-ratings --dataset demo --file examples/demo/ratings_long.csv
python -m benchmark.rating.cli serve                   # http://127.0.0.1:8000, password in configs/rating_app.yaml
python -m benchmark.rating.cli export --dataset demo           # everything, to data/processed/demo/
python -m benchmark.rating.cli export --dataset demo --release # publishable, to data/release/demo/
```

## Data privacy

The tool runs on your machine and your data is not shared with the authors. Only calls to a real AI judge send data out.

| Step | What leaves your machine |
|---|---|
| `judge run` | The task, rubric and idea text. The P5 test adds other ideas with their human mean scores |
| `variance run` | The same, and rater twins add one rater's scores on other ideas, without the rater's identity |
| `perturb reword` | The rubric, with no ideas or ratings |
| Everything else, and `--backend fake` | Nothing |

Prompts go to OpenRouter and the model provider. The tool sets no data-retention options, so check their terms before sending confidential ideas, such as an organization's submissions in the field test.

- **Stored locally:** prompts and replies are cached in `results/cache/`, so the cache contains idea text.
- **Kept out of git:** `data/` (including the rating database), `results/` and `.env`.
- **Raters:** use a code for `rater_id`, not a name. Rating exports replace usernames with pseudonyms.

## Forking and extending

The code is MIT licensed, so you can fork and adapt it. Keep the `LICENSE` and `NOTICE` files.

- **New rubric or judge:** no code needed. Add a rubric YAML file, or add any OpenRouter model to `configs/judges.yaml`.
- **Local model or another provider:** add a backend class in `benchmark/judge.py`. Every model call goes through its `complete()` method, which returns the reply, usage and cost. `OpenRouterBackend` is the reference, and `FakeBackend` is the smallest example. Add it to the `--backend` option and give it its own cache folder, as `fake` does.
- **Private work:** a fork of a public repository is public. Keep data in `data/` and `results/`, or work in a separate private repository.
- **Comparable results:** do not edit released prompt files; add a new version. Report the commit you ran.

## Roadmap

The tool is designed to run beyond OpenRouter. Planned:

- Local open-weight models, so that nothing leaves the machine
- Direct APIs for other model providers

## What's in the repository

| Path | What it is |
|---|---|
| `benchmark/rating/` | Rating platform |
| `benchmark/judge.py` | Sends ideas to AI judges through OpenRouter |
| `benchmark/perturb.py` | The stability tests and rubric rewordings |
| `benchmark/agree.py` | The AI judge diagnostic |
| `benchmark/variance.py` | The variance pipeline (prototype) |
| `benchmark/shadow.py` | The field test (prototype) |
| `benchmark/load.py` | Checks a data folder |
| `benchmark/figures.py` | The diagnostic figure |
| `benchmark/prompts/` | Every prompt sent to a model |
| `configs/` | The AI judges, rubrics and rating app settings |
| `examples/` | Example data |
| `data/`, `results/` | Your data and outputs (not tracked by git) |

## Reproducibility

- Prompts are versioned files. A released version is never edited; changes go in a new file.
- Every model call is saved in `results/cache/`, with the exact prompt, model, provider and cost.
- Model versions are pinned in `configs/judges.yaml`.
- API keys come from `.env` or the environment, and `.env` is not tracked by git.

## License and citation

Code is released under the MIT License (see `LICENSE` and `NOTICE`). Citation details will be added on publication.
