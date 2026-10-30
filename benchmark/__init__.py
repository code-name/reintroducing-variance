"""Reintroducing variance into idea evaluation: an open instrument for trusting AI judges.

The instrument has four modules:

- ``benchmark.rating``   the rating platform: blind multi-rater collection, coverage sampler,
                         imports, per-idea aggregates (mean, spread, standard error) and exports,
                         with a web app and CLI (adapted from NSDA Corp, see NOTICE).
- ``benchmark.judge``    the AI judges: the same rubric as the human raters, pluggable backends,
                         a disk cache for every call, a cost log, and a lab's own data folder.
- ``benchmark.perturb``  stability: an LLM rewrites the rating instructions in ways that should not
                         matter (frozen into versioned files), alongside the other changes of the
                         stability grid.
- ``benchmark.agree``    the AI judge diagnostic: agreement, leniency, stability and variance for
                         every judge, plus the human ceiling, contested ideas and how much of the
                         human disagreement could be measured at all.

Around them: ``benchmark.variance`` (a prototype pipeline that combines and prompts LLMs so that
AI judgment shows which ideas people would disagree about, judged on contested ideas it has not
seen), ``benchmark.shadow`` (a prototype field test: an AI judge in shadow next to an
organization's evaluators), ``benchmark.load`` (reads and checks a data folder) and
``benchmark.figures`` (the diagnostic figure).
"""
from pathlib import Path

__version__ = "0.4.0"

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
DATA_DIR = REPO_DIR / "data"
RESULTS_DIR = REPO_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"
