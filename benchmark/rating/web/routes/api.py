"""Machine-facing export API. One route serves the browser (session cookie) and machine
clients (HTTP Basic auth), as in NSDA ``app/routes/api.py``. Admin only.

    curl -u admin:PASSWORD "http://localhost:8000/api/export/ratings.csv?dataset=demo"
    curl -u admin:PASSWORD ".../api/export/ratings.csv?dataset=demo&split=calibration"
"""
from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from ...auth import verify_password
from ...exports import render
from ...models import Dataset, User
from .auth import current_user

router = APIRouter()
TABLES = ("ideas", "tasks", "ratings", "judge_scores")


def _api_user(request: Request) -> User | None:
    user = current_user(request)
    if user is not None:
        return user
    header = request.headers.get("Authorization") or ""
    if header.lower().startswith("basic "):
        try:
            username, _, password = base64.b64decode(header.split(" ", 1)[1]).decode("utf8").partition(":")
        except Exception:  # noqa: BLE001 - malformed header is a 401
            return None
        with request.app.state.session_factory() as db:
            row = db.query(User).filter_by(username=username.lower()).one_or_none()
            if row is not None and row.active and verify_password(password, row.password_hash):
                db.expunge(row)
                return row
    return None


@router.get("/api/export/{table}.csv")
async def export_table(request: Request, table: str, dataset: str, split: str | None = None):
    user = _api_user(request)
    if user is None:
        raise HTTPException(status_code=401, headers={"WWW-Authenticate": 'Basic realm="orgsci export"'})
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="admin only")
    if table not in TABLES:
        raise HTTPException(status_code=404, detail=f"table must be one of {TABLES}")
    with request.app.state.session_factory() as db:
        ds = db.query(Dataset).filter_by(name=dataset).one_or_none()
        if ds is None:
            raise HTTPException(status_code=404, detail="no such dataset")
        body = render(db, ds, table, splits={split} if split else None)
    name = f"{ds.name}_{table}{'_' + split if split else ''}.csv"
    return Response(content=body, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
