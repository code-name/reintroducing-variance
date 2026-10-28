"""Login and logout: session-cookie auth against the users table (ported from NSDA)."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse

from ...auth import verify_password
from ...models import User
from ..templating import templates

router = APIRouter()


def current_user(request: Request) -> User | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    with request.app.state.session_factory() as db:
        user = db.get(User, user_id)
        if user is None or not user.active or user.role == "external":
            return None
        db.expunge(user)
        return user


def require_user(request: Request) -> User:
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    return user


def require_admin(request: Request) -> User:
    user = require_user(request)
    if user.role != "admin":
        raise HTTPException(status_code=302, headers={"Location": "/dashboard"})
    return user


@router.get("/login")
async def login_form(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse(request=request, name="login.html", context={"error": None})


@router.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    with request.app.state.session_factory() as db:
        user = db.query(User).filter_by(username=username.strip().lower()).one_or_none()
        if (user is None or not user.active or user.role == "external"
                or not verify_password(password, user.password_hash)):
            return templates.TemplateResponse(request=request, name="login.html",
                                              context={"error": "Invalid username or password."},
                                              status_code=401)
        user_id = user.id
    request.session.clear()
    request.session["user_id"] = user_id
    request.session["session_id"] = str(uuid.uuid4())
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)
