# Reintroducing variance into idea evaluation

**An open instrument for trusting AI judges.** Benchmarks LLMs that rate ideas against human raters, including whether a model can identify which ideas people disagree about.

Goran Calic, Hila Lifshitz, Alex Thabane, Vikram Arora and Matthew Zychowicz. Research artifact for the *Organization Science* special issue "AI-Enabled Frontiers in Organizational Science".

> **Status:** early pilot. The variance pipeline and the field test are prototypes.

## Getting started

Requires Python 3.11, 3.12 or 3.13. All commands are for Windows PowerShell.

```powershell
git clone https://github.com/code-name/reintroducing-variance.git
cd reintroducing-variance
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Clone into a short folder path, because Windows limits path length. Run every command from this folder with `.venv` active. In a new terminal, run `.venv\Scripts\Activate.ps1` again. If PowerShell blocks it, run `Set-ExecutionPolicy -Scope Process Bypass` first.

### Run the example

No API key needed:

```powershell
python -m benchmark.agree      # the AI judge diagnostic
python -m benchmark.variance   # the variance pipeline
python -m benchmark.figures    # the diagnostic figure, saved to results/figure.png
pytest                         # the tests
```

`examples/alternative_uses/` has 15 ideas, synthetic ratings and saved scores from five AI judges. These are not the human ratings from the pilot in our proposal, so the numbers will differ. Use it to test the tools, not to estimate anything. `benchmark.variance` may print a `ConstantInputWarning` on this small sample; it is safe to ignore.

### Start the rating app

No API key needed:

```powershell
python -m benchmark.rating.cli init-db          # first time only
python -m benchmark.rating.cli seed-dev-users   # test logins for your own computer only

# The app starts empty, so load data (the invented demo here)
python -m benchmark.rating.cli add-rubric --file configs/rubrics/novelty_usefulness_v1.0.yaml
python -m benchmark.rating.cli create-dataset --name demo --rubric novelty_usefulness_v1.0 --target-raters 3
python -m benchmark.rating.cli import-tasks   --dataset demo --file examples/demo/tasks.csv
python -m benchmark.rating.cli import-ideas   --dataset demo --file examples/demo/ideas.csv
python -m benchmark.rating.cli import-ratings --dataset demo --file examples/demo/ratings_long.csv

# Start the app
python -m benchmark.rating.cli serve
```

Open http://127.0.0.1:8000 and log in as `admin` or `rater1`. The password is in `configs/rating_app.yaml`.

To export ratings, stop the app with `Ctrl+C`, then run:

```powershell
python -m benchmark.rating.cli export --dataset demo            # everything, to data/processed/demo/
python -m benchmark.rating.cli export --dataset demo --release  # publishable, to data/release/demo/
```

## When you need an API key

Only these commands call a real model, and each needs an [OpenRouter](https://openrouter.ai) key. One key covers all five judges.

- `python -m benchmark.judge run`
- `python -m benchmark.variance run`
- `python -m benchmark.perturb reword`

Add `--backend fake` to run any of them offline with fake scores, to check your setup.

### Set and test your key

Create a key at [openrouter.ai](https://openrouter.ai) and add credit. Then:

```powershell
# Copy the settings file, then open .env and paste your key after OPENROUTER_API_KEY=
copy .env.example .env

# Test the key: six real calls, less than one cent
python -m benchmark.judge run --data-dir examples/demo --perturbations P0_baseline --judges qwen3.8-27b
```

| Last line you see | What it means |
|---|---|
| `examples/demo: 12 score rows, 0 unparsed` | The key works |
| `OPENROUTER_API_KEY is not set` | The key is missing from `.env`, or the file was not copied |
| `AuthenticationError` with `401` (after about 20 seconds) | The key is wrong |

## Run it on your own ideas

### 1. Prepare your data

Create a folder such as `my_study/` with three CSV files:

| File | Columns |
|---|---|
| `ideas.csv` | `idea_id, idea_text, task_id`. `source` (`human` or `ai`) is required for the rating app. Optional: `split` (`calibration` or `heldout`), `target_raters`, `advanced` |
| `tasks.csv` | `task_id, task_prompt_verbatim` |
| `ratings_long.csv` | `idea_id, rater_id, dimension, score, scale_min, scale_max`. Optional: `condition` |

Use one row per idea, rater and dimension. Averaged scores are not enough.

### 2. Write your rubric

```powershell
copy configs\rubrics\novelty_usefulness_v1.0.yaml my_rubric.yaml
```

Edit the copy. Any dimensions and any scale work, but each dimension `id` must match the `dimension` column in `ratings_long.csv`. A mismatch gives no error: that dimension is left out of the report. `benchmark.figures` draws the example's novelty figure only.

### 3. Run the commands

Set and test your key first (see above).

```powershell
# Check your data
python -m benchmark.load my_study/

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

**Always reword a new rubric.** If a rubric keeps the dimension names but changes their wording, the run does not fail: it silently uses the old rewordings, which invalidates the P1 stability test. `--out` needs a new name each time, because existing files are never overwritten.

Outputs go to `results/`. Calls are cached, so reruns do not pay twice. `JUDGE_BUDGET_USD` in `.env` stops a run once it has spent that amount. The limit applies to each run, and calls already under way still finish.

## How it works

Inputs are your ideas, tasks and rubric. Every other box has the same name as a heading below. Dashed boxes are prototypes.

```mermaid
flowchart TD
    I["Inputs"] --> M1["Module 1: the rating platform"]
    I --> M2["Module 2: AI judges"]
    M2 --> M3["Module 3: stability"]
    M1 -- human ratings --> M4["Module 4: the AI judge diagnostic"]
    M3 -- AI scores --> M4
    M1 --> V["The variance pipeline (prototype)"]
    M3 --> V
    M1 --> F["The field test (prototype)"]
    M3 --> F
    classDef proto stroke-dasharray: 5 4
    class V,F proto
```

### Module 1: the rating platform

`benchmark/rating/` is a self-hosted web app and CLI for collecting human ratings, adapted from NSDA Corp's annotation apps (see `NOTICE`). Raters rate blind, each idea gets 10 raters by default, and release exports withhold held-out scores.

### Module 2: AI judges

`benchmark/judge.py` sends the same ideas to AI judges with the same rubric the raters read, through OpenRouter. Every call is cached with its prompt, model, provider and cost.

### Module 3: stability

`benchmark/perturb.py` runs each judge under small prompt changes that should not change its scores: reworded statements, swapped dimension order, removed scale labels, batching and worked examples.

### Module 4: the AI judge diagnostic

`benchmark/agree.py` writes one report per AI judge:

| Measure | Question |
|---|---|
| **Variance** | Does the judge know which ideas people disagree about? |
| **Agreement** | Does it rank ideas in the same order as people? |
| **Leniency** | Does it score ideas higher or lower than people? |
| **Stability** | Does it score the same way when the prompt changes in ways that should not matter? |

### The variance pipeline (prototype)

`benchmark/variance.py` searches for a way to make AI judgment flag the ideas people would disagree about. It chooses the best candidate on some ideas, tests it on unseen ones, and reports how well it finds the contested ideas. Producing a mean, a spread and a contested flag for each idea is in the code (`panel_output`), but has no command yet. It needs many rated ideas; the example's 15 only demonstrate the steps.

### The field test (prototype)

`benchmark/shadow.py` compares an AI judge with an organization's evaluators without affecting their decisions. It reports how many advanced ideas the judge would have dropped, and whether those are the ideas the evaluators disagreed about.

Add the decisions in an `advanced` column of `ideas.csv`. To run the invented example:

```powershell
python -m benchmark.judge run --data-dir examples/screen --backend fake --perturbations P0_baseline --judges qwen3.8-27b,deepseek-v4-pro
python -m benchmark.shadow --data-dir examples/screen
```

For real data, use your own folder and leave out `--backend fake`.

## Data privacy

The tool runs on your machine and your data is not shared with the authors. Only real model calls send data out, to OpenRouter and the model provider:

| Command | What leaves your machine |
|---|---|
| `judge run` | Task, rubric and idea text; P5 adds other ideas with human mean scores |
| `variance run` | The same; rater twins add one rater's scores, without the rater's identity |
| `perturb reword` | The rubric only |
| Anything else, or `--backend fake` | Nothing |

The tool sets no data-retention options, so check their terms before sending confidential ideas. The cache in `results/cache/` contains idea text. `data/`, `results/` and `.env` are not tracked by git. Use a code for `rater_id`, not a name.

## Forking and extending

The code is MIT licensed. Keep the `LICENSE` and `NOTICE` files.

- **New rubric or judge:** no code needed. Add a rubric YAML file, or add an OpenRouter model to `configs/judges.yaml`.
- **Local model or another provider:** add a backend class in `benchmark/judge.py`, using `OpenRouterBackend` and `FakeBackend` as examples. Add it to the `--backend` options in `judge.py`, `perturb.py` and `variance.py`, and give it its own cache folder, as `fake` does.
- **Private work:** a fork of a public repository is public. Keep data in `data/` and `results/`, or use a separate private repository.

## Roadmap

- Local open-weight models, so that nothing leaves the machine
- Direct APIs for other model providers

## License and citation

Code is released under the MIT License (see `LICENSE` and `NOTICE`). Citation details will be added on publication.
