"""Admin routes: overview, datasets (create, upload tasks and ideas), analytics, users,
judge runs, exports. Analytics are admin-only so that raters stay blind."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import RedirectResponse

from ... import importers
from ...aggregate import coverage_summary, idea_aggregates, most_contested
from ...exports import rater_code
from ...models import (AuditLog, Dataset, Idea, IdeaFlag, JudgeRun, JudgeScore, Rating, Rubric,
                       User)
from ...rubrics import RubricError, register_rubric, rubric_model
from ..templating import templates
from .auth import require_admin

router = APIRouter(prefix="/admin")


def _page(request, name, user, nav, **ctx):
    return templates.TemplateResponse(request=request, name=name,
                                      context={"user": user, "nav": nav, **ctx})


@router.get("")
async def overview(request: Request, user=Depends(require_admin)):
    with request.app.state.session_factory() as db:
        stats = {"datasets": db.query(Dataset).count(), "ideas": db.query(Idea).count(),
                 "ratings": db.query(Rating).count(), "flags": db.query(IdeaFlag).count(),
                 "raters": db.query(User).filter(User.role != "admin").count(),
                 "rubrics": db.query(Rubric).count(), "judge_runs": db.query(JudgeRun).count()}
    return _page(request, "admin.html", user, "admin", page_title="Admin", stats=stats)


@router.get("/datasets")
async def datasets(request: Request, msg: str | None = None, user=Depends(require_admin)):
    with request.app.state.session_factory() as db:
        rows = [{"ds": d, "cov": coverage_summary(db, d), "rubric": db.get(Rubric, d.rubric_sha256)}
                for d in db.query(Dataset).order_by(Dataset.created_at)]
        rubrics = db.query(Rubric).order_by(Rubric.created_at).all()
    return _page(request, "admin_datasets.html", user, "admin-datasets", page_title="Datasets",
                 rows=rows, rubrics=rubrics, msg=msg)


@router.post("/datasets/create")
async def create_dataset(request: Request, user=Depends(require_admin)):
    form = await request.form()
    name = (form.get("name") or "").strip()
    with request.app.state.session_factory() as db:
        rub = db.get(Rubric, form.get("rubric_sha256") or "")
        if not name or rub is None:
            return RedirectResponse(url="/admin/datasets?msg=Name and rubric are required", status_code=303)
        if db.query(Dataset).filter_by(name=name).one_or_none():
            return RedirectResponse(url="/admin/datasets?msg=Dataset exists", status_code=303)
        target = int(form.get("target_raters") or request.app.state.cfg.default_target_raters)
        db.add(Dataset(name=name, rubric_sha256=rub.sha256, target_raters=target))
        db.add(AuditLog(user_id=user.id, entity="dataset", entity_id=name, action="create", payload={}))
        db.commit()
    return RedirectResponse(url="/admin/datasets?msg=Created " + name, status_code=303)


@router.post("/rubrics/upload")
async def upload_rubric(request: Request, file: UploadFile, user=Depends(require_admin)):
    text = (await file.read()).decode("utf-8")
    with request.app.state.session_factory() as db:
        try:
            row = register_rubric(db, text, source=f"upload:{file.filename}")
        except RubricError as exc:
            return RedirectResponse(url=f"/admin/datasets?msg={str(exc)[:150]}", status_code=303)
    return RedirectResponse(url=f"/admin/datasets?msg=Rubric {row.rubric_id}_v{row.version} registered",
                            status_code=303)


@router.post("/datasets/{dataset_id}/upload")
async def upload_csv(request: Request, dataset_id: str, kind: str, file: UploadFile,
                     user=Depends(require_admin)):
    fn = {"tasks": importers.import_tasks, "ideas": importers.import_ideas,
          "ratings": importers.import_ratings}.get(kind)
    with request.app.state.session_factory() as db:
        ds = db.get(Dataset, dataset_id)
        if ds is None or fn is None:
            return RedirectResponse(url="/admin/datasets?msg=Unknown dataset or kind", status_code=303)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / Path(file.filename or f"{kind}.csv").name
            path.write_bytes(await file.read())
            try:
                n = fn(db, ds, path)
            except (importers.ImportErrorWithRow, ValueError) as exc:
                db.rollback()
                return RedirectResponse(url=f"/admin/datasets?msg={str(exc)[:150]}", status_code=303)
    return RedirectResponse(url=f"/admin/datasets?msg=Imported {n} {kind} into {ds.name}", status_code=303)


@router.get("/analytics")
async def analytics(request: Request, dataset: str | None = None, user=Depends(require_admin)):
    with request.app.state.session_factory() as db:
        all_ds = db.query(Dataset).order_by(Dataset.created_at).all()
        ds = next((d for d in all_ds if d.name == dataset), all_ds[0] if all_ds else None)
        ctx = {"datasets": all_ds, "dataset": ds}
        if ds is not None:
            rub = rubric_model(db.get(Rubric, ds.rubric_sha256))
            aggs = idea_aggregates(db, ds.id, ds.rubric_sha256)
            raters = {}
            for (rid,) in db.query(Rating.rater_id).join(Idea, Idea.id == Rating.idea_id).filter(
                    Idea.dataset_id == ds.id).distinct():
                u = db.get(User, rid)
                n = db.query(Rating.idea_id).filter_by(rater_id=rid).distinct().count()
                raters[rid] = {"code": rater_code(rid), "lab": u.lab, "expertise": u.expertise,
                               "role": u.role, "ideas": n}
            ctx.update({"cov": coverage_summary(db, ds), "rubric": rub,
                        "contested": {d.id: most_contested(aggs, d.id) for d in rub.dimensions},
                        "raters": sorted(raters.values(), key=lambda r: -r["ideas"]),
                        "flags": db.query(IdeaFlag).join(Idea, Idea.id == IdeaFlag.idea_id)
                        .filter(Idea.dataset_id == ds.id).count()})
    return _page(request, "analytics.html", user, "admin-analytics", page_title="Analytics", **ctx)


@router.get("/users")
async def users(request: Request, user=Depends(require_admin)):
    with request.app.state.session_factory() as db:
        rows = db.query(User).order_by(User.role, User.username).all()
    return _page(request, "admin_users.html", user, "admin-users", page_title="Users", users=rows)


@router.get("/runs")
async def runs(request: Request, user=Depends(require_admin)):
    with request.app.state.session_factory() as db:
        rows = [{"run": r, "scores": db.query(JudgeScore).filter_by(run_id=r.id).count(),
                 "dataset": db.get(Dataset, r.dataset_id).name}
                for r in db.query(JudgeRun).order_by(JudgeRun.created_at.desc())]
    return _page(request, "admin_runs.html", user, "admin-runs", page_title="Judge runs", rows=rows)


@router.get("/exports")
async def exports(request: Request, user=Depends(require_admin)):
    with request.app.state.session_factory() as db:
        rows = db.query(Dataset).order_by(Dataset.created_at).all()
    return _page(request, "admin_exports.html", user, "admin-exports", page_title="Exports", datasets=rows)
