"""FastAPI application factory (server-rendered Jinja2 + vanilla JS, as in the NSDA apps).

Run with::

    python -m benchmark.rating.cli serve
    # or: uvicorn benchmark.rating.web.main:create_app --factory --port 8000

Unlike the NSDA app, importing this module has no side effects: the database is opened only
when ``create_app`` is called.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from ..config import AppConfig, load_config, session_secret
from ..models import init_db
from .routes import admin, api, auth, pages
from .templating import STATIC_DIR


class RevalidatingStaticFiles(StaticFiles):
    """StaticFiles with Cache-Control: no-cache, so edited assets are revalidated (NSDA rule)."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    session_factory = init_db(cfg.db_path)

    app = FastAPI(title="OrgSci idea rating", version="0.1.0")
    app.state.cfg = cfg
    app.state.session_factory = session_factory
    app.add_middleware(SessionMiddleware, secret_key=session_secret(cfg), max_age=60 * 60 * 12,
                       same_site="lax")
    app.include_router(auth.router)
    app.include_router(pages.router)
    app.include_router(admin.router)
    app.include_router(api.router)
    app.mount("/static", RevalidatingStaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def root(request: Request):
        return RedirectResponse(url="/dashboard" if request.session.get("user_id") else "/login", status_code=302)

    return app
