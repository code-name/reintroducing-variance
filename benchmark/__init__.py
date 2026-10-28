"""Reintroducing variance into idea evaluation: an open instrument for trusting AI judges.

The instrument has four modules:

- ``benchmark.rating``   the rating platform: blind multi-rater collection, coverage sampler,
                         imports, per-idea aggregates (mean, spread, standard error) and exports,
                         with a web app and CLI (adapted from NSDA Corp, see NOTICE).
- ``benchmark.judge``    the AI judges: the same rubric as the human raters, pluggable backends,
                         a disk cache for every call, a cost log, and a lab's own data folder.
- ``benchmark.perturb``  stability: changes to the instructions that should not matter, including
                         rewordings written by an LLM and frozen into versioned files.
- ``benchmark.agree``    the AI judge diagnostic: agreement, leniency, stability and variance for
                         every judge, plus the human ceiling and contested ideas.

Around them: ``benchmark.variance`` (an early prototype of a pipeline that builds panels of AI
judges that disagree where people disagree), ``benchmark.load`` (reads and checks a data folder)
and ``benchmark.figures`` (the diagnostic figure).
"""
from pathlib import Path

__version__ = "0.3.0"

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
DATA_DIR = REPO_DIR / "data"
RESULTS_DIR = REPO_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"
