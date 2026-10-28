"""Configuration loading for the rating app (``configs/rating_app.yaml``).

Ported from NSDA ``core/config.py``. Secrets come from environment variables, never from
committed files. Environment variables use the prefix ``ORGSCI_RATING_``.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "rating_app.yaml"


@dataclass
class BindConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class DevUserSeed:
    username: str
    display_name: str
    role: str = "rater"
    lab: str | None = None
    expertise: str | None = None


@dataclass
class AppConfig:
    bind: BindConfig = field(default_factory=BindConfig)
    db_path: Path = REPO_ROOT / "data" / "app" / "rating.db"
    data_dir: Path = REPO_ROOT / "data" / "app"
    rubrics_dir: Path = REPO_ROOT / "configs" / "rubrics"
    dev_seed_users: list[DevUserSeed] = field(default_factory=list)
    dev_password: str = "change-me-locally"
    default_target_raters: int = 10
    config_path: Path = DEFAULT_CONFIG_PATH


def _resolve(base: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load the YAML config. ``ORGSCI_RATING_CONFIG`` overrides the default path.

    Relative paths inside the file are resolved against the repository root (the parent of
    ``configs/``), so the app behaves the same from any working directory.
    """
    cfg_path = Path(path or os.environ.get("ORGSCI_RATING_CONFIG", DEFAULT_CONFIG_PATH))
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    raw: dict = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    base = cfg_path.parent.parent

    cfg = AppConfig(config_path=cfg_path)
    bind = raw.get("bind") or {}
    cfg.bind = BindConfig(host=bind.get("host", cfg.bind.host), port=int(bind.get("port", cfg.bind.port)))
    cfg.db_path = _resolve(base, (raw.get("db") or {}).get("path", "data/app/rating.db"))
    cfg.data_dir = _resolve(base, raw.get("data_dir", "data/app"))
    cfg.rubrics_dir = _resolve(base, raw.get("rubrics_dir", "configs/rubrics"))
    sampler = raw.get("sampler") or {}
    cfg.default_target_raters = int(sampler.get("target_raters_per_idea", cfg.default_target_raters))
    dev = raw.get("dev") or {}
    cfg.dev_seed_users = [
        DevUserSeed(
            username=u["username"],
            display_name=u.get("display_name", u["username"]),
            role=u.get("role", "rater"),
            lab=u.get("lab"),
            expertise=u.get("expertise"),
        )
        for u in dev.get("seed_users", [])
    ]
    cfg.dev_password = os.environ.get("ORGSCI_RATING_DEV_PASSWORD", dev.get("dev_password", cfg.dev_password))
    return cfg


def session_secret(cfg: AppConfig) -> str:
    """Session signing secret: env var first, else a key generated once under the data dir."""
    env = os.environ.get("ORGSCI_RATING_SESSION_SECRET")
    if env:
        return env
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    key_path = cfg.data_dir / "secret_key"
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    key = secrets.token_urlsafe(32)
    key_path.write_text(key, encoding="utf-8")
    return key
