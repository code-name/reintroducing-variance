"""Command-line entry points for the rating platform.

    python -m benchmark.rating.cli init-db
    python -m benchmark.rating.cli seed-dev-users                     # localhost only
    python -m benchmark.rating.cli create-user --username u [--admin] [--lab L] [--expertise E]
    python -m benchmark.rating.cli add-rubric --file configs/rubrics/novelty_usefulness_v1.0.yaml
    python -m benchmark.rating.cli create-dataset --name demo --rubric novelty_usefulness_v1.0 [--target-raters 10]
    python -m benchmark.rating.cli import-tasks   --dataset demo --file tasks.csv
    python -m benchmark.rating.cli import-ideas   --dataset demo --file ideas.csv
    python -m benchmark.rating.cli import-ratings --dataset demo --file ratings_long.csv
    python -m benchmark.rating.cli import-judge-scores --dataset demo --file scores.csv --name run1 \
        --backend openrouter --model-name M --prompt pointwise_v1.1 [--model-version V]
    python -m benchmark.rating.cli coverage --dataset demo
    python -m benchmark.rating.cli export --dataset demo [--release] [--out DIR]
    python -m benchmark.rating.cli serve [--port 8000]

Ported from NSDA ``cli.py`` (init-db, create-user, seed-dev-users) and extended.
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

from .auth import hash_password
from .config import REPO_ROOT, load_config
from .models import EXPERTISE, Dataset, User, init_db


def _db(cfg):
    return init_db(cfg.db_path)()


def _dataset(db, name: str) -> Dataset:
    ds = db.query(Dataset).filter_by(name=name).one_or_none()
    if ds is None:
        sys.exit(f"No dataset named {name!r}. Create it with create-dataset.")
    return ds


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - flat command dispatch
    p = argparse.ArgumentParser(prog="python -m benchmark.rating.cli")
    p.add_argument("--config", default=None, help="path to configs/rating_app.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db")
    sub.add_parser("seed-dev-users")
    c = sub.add_parser("create-user")
    c.add_argument("--username", required=True)
    c.add_argument("--password", default=None, help="omit to be prompted")
    c.add_argument("--admin", action="store_true")
    c.add_argument("--display-name", default=None)
    c.add_argument("--lab", default=None)
    c.add_argument("--expertise", default="unknown", choices=EXPERTISE)
    c.add_argument("--domain", default=None)
    r = sub.add_parser("add-rubric")
    r.add_argument("--file", required=True)
    d = sub.add_parser("create-dataset")
    d.add_argument("--name", required=True)
    d.add_argument("--rubric", required=True, help="<id>_v<version> or sha256")
    d.add_argument("--target-raters", type=int, default=None)
    d.add_argument("--description", default=None)
    for name in ("import-tasks", "import-ideas", "import-ratings"):
        x = sub.add_parser(name)
        x.add_argument("--dataset", required=True)
        x.add_argument("--file", required=True)
    j = sub.add_parser("import-judge-scores")
    j.add_argument("--dataset", required=True)
    j.add_argument("--file", required=True)
    j.add_argument("--name", required=True)
    j.add_argument("--backend", required=True)
    j.add_argument("--model-name", required=True)
    j.add_argument("--model-version", default=None)
    j.add_argument("--prompt", required=True, help="judge prompt name, e.g. pointwise_v1.1")
    j.add_argument("--perturbation", default="P0_baseline")
    j.add_argument("--temperature", type=float, default=0.0)
    cv = sub.add_parser("coverage")
    cv.add_argument("--dataset", required=True)
    e = sub.add_parser("export")
    e.add_argument("--dataset", required=True)
    e.add_argument("--release", action="store_true", help="calibration ratings only, held-out scores withheld")
    e.add_argument("--out", default=None)
    s = sub.add_parser("serve")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)

    a = p.parse_args(argv)
    cfg = load_config(a.config)

    if a.command == "init-db":
        init_db(cfg.db_path)
        print(f"Database ready: {cfg.db_path}")
        return 0

    if a.command == "serve":
        import os

        import uvicorn

        # uvicorn's factory takes no arguments, so pass the chosen config through the
        # environment; create_app() -> load_config() reads ORGSCI_RATING_CONFIG.
        os.environ["ORGSCI_RATING_CONFIG"] = str(cfg.config_path)
        uvicorn.run("benchmark.rating.web.main:create_app", factory=True,
                    host=a.host or cfg.bind.host, port=a.port or cfg.bind.port)
        return 0

    db = _db(cfg)
    if a.command == "seed-dev-users":
        made = 0
        for seed in cfg.dev_seed_users:
            if db.query(User).filter_by(username=seed.username).one_or_none():
                print(f"exists: {seed.username}")
                continue
            db.add(User(username=seed.username, display_name=seed.display_name, role=seed.role,
                        lab=seed.lab, expertise=seed.expertise or "unknown",
                        password_hash=hash_password(cfg.dev_password)))
            made += 1
            print(f"created: {seed.username} ({seed.role})")
        db.commit()
        print(f"Seeded {made} user(s) with the dev password (localhost only).")
        return 0

    if a.command == "create-user":
        if db.query(User).filter_by(username=a.username.lower()).one_or_none():
            sys.exit(f"User already exists: {a.username}")
        pw = a.password or getpass.getpass(f"Password for {a.username}: ")
        db.add(User(username=a.username.lower(), display_name=a.display_name or a.username,
                    role="admin" if a.admin else "rater", password_hash=hash_password(pw),
                    lab=a.lab, expertise=a.expertise, domain=a.domain))
        db.commit()
        print(f"Created user: {a.username}")
        return 0

    if a.command == "add-rubric":
        from .rubrics import register_rubric_file

        row = register_rubric_file(db, Path(a.file))
        print(f"Rubric {row.rubric_id}_v{row.version}  sha256={row.sha256}")
        return 0

    if a.command == "create-dataset":
        from .rubrics import find_rubric

        rub = find_rubric(db, a.rubric)
        if rub is None:
            sys.exit(f"No rubric {a.rubric!r}. Register it with add-rubric first.")
        if db.query(Dataset).filter_by(name=a.name).one_or_none():
            sys.exit(f"Dataset exists: {a.name}")
        db.add(Dataset(name=a.name, description=a.description, rubric_sha256=rub.sha256,
                       target_raters=a.target_raters or cfg.default_target_raters))
        db.commit()
        print(f"Created dataset {a.name} under rubric {rub.rubric_id}_v{rub.version}")
        return 0

    if a.command in ("import-tasks", "import-ideas", "import-ratings"):
        from . import importers

        ds = _dataset(db, a.dataset)
        fn = {"import-tasks": importers.import_tasks, "import-ideas": importers.import_ideas,
              "import-ratings": importers.import_ratings}[a.command]
        n = fn(db, ds, Path(a.file))
        print(f"{a.command}: {n} new row(s) into {ds.name}")
        return 0

    if a.command == "import-judge-scores":
        from .importers import import_judge_scores

        ds = _dataset(db, a.dataset)
        run = import_judge_scores(db, ds, Path(a.file), name=a.name, backend=a.backend,
                                  model_name=a.model_name, model_version=a.model_version,
                                  prompt_name=a.prompt, perturbation_id=a.perturbation,
                                  temperature=a.temperature)
        print(f"Judge run {run.name}  config_sha256={run.config_sha256}")
        return 0

    if a.command == "coverage":
        from .aggregate import coverage_summary

        print(json.dumps(coverage_summary(db, _dataset(db, a.dataset)), indent=2))
        return 0

    if a.command == "export":
        from .exports import write_internal, write_release

        ds = _dataset(db, a.dataset)
        if a.release:
            out = Path(a.out) if a.out else REPO_ROOT / "data" / "release" / ds.name
            manifest = write_release(db, ds, out)
        else:
            out = Path(a.out) if a.out else REPO_ROOT / "data" / "processed" / ds.name
            manifest = write_internal(db, ds, out)
        print(f"Wrote {manifest['kind']} export to {out}")
        for f in manifest["files"]:
            print(f"  {f['file']}: {f['rows']} rows")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
