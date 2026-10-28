"""Shared Jinja2 environment. Ported from NSDA ``app/templating.py``: ``asset_url`` appends the
file mtime to /static URLs so an edited CSS/JS file is a new URL and a refresh picks it up."""
from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def asset_url(path: str) -> str:
    clean = path.lstrip("/").removeprefix("static/")
    try:
        version = int((STATIC_DIR / clean).stat().st_mtime)
    except OSError:
        return path
    return f"{path}?v={version}"


def fmt(value, digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


templates.env.globals["asset_url"] = asset_url
templates.env.filters["fmt"] = fmt
