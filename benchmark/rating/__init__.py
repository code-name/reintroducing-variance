"""Rating platform: the plumbing that collects, stores and exports idea ratings.

Adapted from NSDA Corp's internal annotation apps (``nsdalab_ocr_labelling_app`` and
``nsdalab_regdoc-requirements_labelling_app``), released for this project by NSDA Corp. See
``NOTICE``. What was kept and what changed:

Kept (the plumbing)
    Multi-user web app with session login and argon2 password hashes; every judgment stored as
    its own row with rater, time and session; an audit log; content-hashed configurations
    (rubrics and judge runs are identified by the sha256 of their canonical form, as NSDA
    runner configs were); a coverage sampler that never assigns a rater the same item twice and
    logs every assignment with its method and priority; admin pages; exports served to the
    browser and to machine clients over HTTP Basic auth.

Changed (because ideas are not OCR pages)
    - Truth is an aggregate, not a resolution. Ratings are never adjudicated to one answer; each
      idea gets many independent ratings and the reference is their mean, spread and
      uncertainty (``aggregate.py``). The NSDA disagreement-resolution and tie-break workflow is
      not ported. Only categorical calls (invalid / off-task / duplicate) are recorded as flags.
    - Raters rate blind. The rating screen never shows an idea's source (human or AI), any model
      score, or any other rater's score. NSDA showed model output to annotators; here that would
      anchor raters on the very judge being tested.
    - The coverage target is k raters per idea (default 10), not two plus a tie-break.
    - Ideas carry a ``split`` (calibration / heldout). Held-out scores are never released.

Modules
    config, auth, models     configuration, password hashing, SQLAlchemy models
    rubrics                  versioned rubric YAML, validated and content-hashed
    importers                tasks, ideas, historical ratings and judge scores from CSV
    sampler                  coverage sampler (k raters per idea, least-covered first)
    ratings                  record a rating or a flag, with audit trail
    aggregate                per-idea mean, spread, standard error; coverage summaries
    exports                  ideas.csv, tasks.csv, ratings_long.csv, judge_scores.csv, release
    cli                      command-line entry points
    web/                     FastAPI + Jinja2 web app
"""
