"""Rater-facing routes: dashboard and the blind rating screen.

Blinding rules (tested in tests/rating/test_web.py):
- the rating screen shows the task prompt, the idea text and the rubric, nothing else;
- it never shows the idea's source (human or AI), model name, any judge score, any other
  rater's score, or aggregate statistics;
- raters never see analytics; only admins do.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse

from ... import sampler
from ...models import Assignment, Dataset, Idea, Rubric, Task, utcnow
from ...ratings import RatingError, record_flag, record_ratings
from ...rubrics import rubric_model
from ..templating import templates
from .auth import require_user

router = APIRouter()


def _rater_progress(db, user_id: str) -> list[dict]:
    rows = []
    for ds in db.query(Dataset).order_by(Dataset.created_at):
        done = db.query(Assignment).filter_by(dataset_id=ds.id, rater_id=user_id, status="done").count()
        rows.append({"dataset": ds, "done": done})
    return rows


@router.get("/dashboard")
async def dashboard(request: Request, user=Depends(require_user)):
    with request.app.state.session_factory() as db:
        progress = _rater_progress(db, user.id)
    return templates.TemplateResponse(request=request, name="dashboard.html", context={
        "user": user, "nav": "dashboard", "page_title": "Dashboard", "progress": progress})


@router.get("/rate")
async def rate(request: Request, dataset: str | None = None, msg: str | None = None,
               user=Depends(require_user)):
    with request.app.state.session_factory() as db:
        if dataset is None:
            open_ds = db.query(Dataset).filter_by(status="open").order_by(Dataset.created_at).all()
            if len(open_ds) == 1:
                return RedirectResponse(url=f"/rate?dataset={open_ds[0].name}", status_code=302)
            return templates.TemplateResponse(request=request, name="rate_pick.html", context={
                "user": user, "nav": "rate", "page_title": "Rate", "datasets": open_ds})
        ds = db.query(Dataset).filter_by(name=dataset).one_or_none()
        if ds is None:
            return RedirectResponse(url="/rate", status_code=302)
        assignment = sampler.next_assignment(db, ds, user.id)
        done = db.query(Assignment).filter_by(dataset_id=ds.id, rater_id=user.id, status="done").count()
        ctx = {"user": user, "nav": "rate", "page_title": "Rate", "dataset": ds, "done": done,
               "msg": msg, "rubric": rubric_model(db.get(Rubric, ds.rubric_sha256))}
        if assignment is None:
            return templates.TemplateResponse(request=request, name="rate.html",
                                              context={**ctx, "assignment": None})
        idea = db.get(Idea, assignment.idea_id)
        task = db.get(Task, idea.task_id)
        # Only what a blind rater may see. Do not add idea.source / model fields here.
        ctx.update({"assignment": assignment.id, "idea_text": idea.text,
                    "task_prompt": task.prompt_verbatim})
    return templates.TemplateResponse(request=request, name="rate.html", context=ctx)


@router.post("/rate/submit")
async def rate_submit(request: Request, user=Depends(require_user)):
    form = await request.form()
    with request.app.state.session_factory() as db:
        a = db.get(Assignment, form.get("assignment_id", ""))
        if a is None or a.rater_id != user.id or a.status != "pending":
            return RedirectResponse(url="/rate", status_code=302)
        ds = db.get(Dataset, a.dataset_id)
        rub = rubric_model(db.get(Rubric, ds.rubric_sha256))
        action = form.get("action", "submit")
        if action == "skip":
            sampler.complete(db, a, status="skipped")
            return RedirectResponse(url=f"/rate?dataset={ds.name}", status_code=303)
        flag = (form.get("flag") or "").strip()
        scores = {d.id: form.get(f"score_{d.id}") for d in rub.dimensions}
        given = {k: v for k, v in scores.items() if v not in (None, "")}
        try:
            if flag:
                record_flag(db, idea_id=a.idea_id, rater_id=user.id, flag=flag,
                            note=(form.get("note") or "").strip() or None)
            if given or not flag:
                duration = int((utcnow() - a.served_at).total_seconds() * 1000)
                record_ratings(db, idea_id=a.idea_id, rater_id=user.id,
                               rubric_sha256=ds.rubric_sha256,
                               scores={k: float(v) for k, v in given.items()},
                               duration_ms=duration, session_id=request.session.get("session_id"))
        except (RatingError, ValueError) as exc:
            db.rollback()
            return RedirectResponse(url=f"/rate?dataset={ds.name}&msg={str(exc)[:120]}", status_code=303)
        sampler.complete(db, a, status="done")
        return RedirectResponse(url=f"/rate?dataset={ds.name}", status_code=303)
