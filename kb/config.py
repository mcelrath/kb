"""
kb/config.py — single config resolver (R3, epic kb-ez9).

Resolves KB_* configuration in this precedence (highest first):
  (a) environment variables — if an env var overrides the toml value, logs a
      one-line note so stale-session overrides are never silent (outage fix).
  (b) ~/.config/kb/config.toml — written by `kb configure`.
  (c) hardcoded defaults (ash:8081 llamacpp 4096, tardis:9510, ~/.cache/kb/).

Module-level singleton: call `load_config()` for the resolved config.
Pass `force_reload=True` to bypass the cache (tests / reconfigure).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KbConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class KbConfig:
    db_path: Path
    embedding_url: str
    embedding_dim: int
    embedding_format: str
    embedding_model: str
    embedding_key: str
    llm_url: str
    summary_mode: str
    tracker_backend: str | None = None  # host-wide kbt backend ([tracker] backend); None = unset


# ---------------------------------------------------------------------------
# Hardcoded defaults (same as the old constants.py literals)
# ---------------------------------------------------------------------------

_DEFAULTS = KbConfig(
    db_path=Path.home() / ".cache" / "kb" / "knowledge.db",
    embedding_url="http://ash:8081/embedding",
    embedding_dim=4096,
    embedding_format="llamacpp",
    embedding_model="",
    embedding_key="",
    llm_url="http://tardis:9510/completion",
    summary_mode="extractive",
    tracker_backend=None,
)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config: KbConfig | None = None


def load_config(force_reload: bool = False) -> KbConfig:
    """Return the resolved KbConfig singleton.

    Precedence (highest first):
      1. Environment variables (KB_*)
      2. ~/.config/kb/config.toml
      3. Hardcoded defaults

    If an environment variable overrides a toml value, a one-line INFO log is
    emitted so stale-session env overrides are never silent.
    """
    global _config
    if _config is not None and not force_reload:
        return _config

    _config = _resolve_config()
    return _config


def _resolve_config() -> KbConfig:
    """Build a KbConfig by layering toml then env on top of defaults."""
    # Start from defaults
    cfg = KbConfig(
        db_path=_DEFAULTS.db_path,
        embedding_url=_DEFAULTS.embedding_url,
        embedding_dim=_DEFAULTS.embedding_dim,
        embedding_format=_DEFAULTS.embedding_format,
        embedding_model=_DEFAULTS.embedding_model,
        embedding_key=_DEFAULTS.embedding_key,
        llm_url=_DEFAULTS.llm_url,
        summary_mode=_DEFAULTS.summary_mode,
        tracker_backend=_DEFAULTS.tracker_backend,
    )

    # Layer (b): toml file
    toml_path = _toml_path()
    toml_values: dict[str, str] = {}
    if toml_path.exists():
        toml_values = _load_toml(toml_path)
        cfg = _apply_toml(cfg, toml_values)

    # Layer (a): environment variables — override toml + defaults; log if overriding toml
    cfg = _apply_env(cfg, toml_values)

    return cfg


def _toml_path() -> Path:
    return Path.home() / ".config" / "kb" / "config.toml"


def _load_toml(path: Path) -> dict[str, str]:
    """Parse the config.toml and return a flat dict of string values.

    Uses tomllib (stdlib, Python 3.11+). Returns {} on any parse error.
    """
    try:
        import tomllib  # type: ignore[import]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[import,no-redef]
        except ImportError:
            return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        # Flatten: only top-level [embedding] and [llm] sections + bare keys
        result: dict[str, str] = {}
        emb = data.get("embedding", {})
        llm = data.get("llm", {})
        db = data.get("database", {})
        tracker = data.get("tracker", {})
        if "url" in emb:
            result["embedding_url"] = str(emb["url"])
        if "dim" in emb:
            result["embedding_dim"] = str(emb["dim"])
        if "format" in emb:
            result["embedding_format"] = str(emb["format"])
        if "model" in emb:
            result["embedding_model"] = str(emb["model"])
        # key is secret — NOT written to toml (stays in settings.local.json)
        if "url" in llm:
            result["llm_url"] = str(llm["url"])
        if "summary_mode" in llm:
            result["summary_mode"] = str(llm["summary_mode"])
        if "db_path" in db:
            result["db_path"] = str(db["db_path"])
        if "backend" in tracker:
            result["tracker_backend"] = str(tracker["backend"])
        return result
    except Exception:
        return {}


def _apply_toml(cfg: KbConfig, toml: dict[str, str]) -> KbConfig:
    """Apply toml values to cfg (only for keys present in toml)."""
    if "embedding_url" in toml:
        cfg.embedding_url = toml["embedding_url"]
    if "embedding_dim" in toml:
        try:
            cfg.embedding_dim = int(toml["embedding_dim"])
        except (ValueError, TypeError):
            pass
    if "embedding_format" in toml:
        cfg.embedding_format = toml["embedding_format"]
    if "embedding_model" in toml:
        cfg.embedding_model = toml["embedding_model"]
    if "llm_url" in toml:
        cfg.llm_url = toml["llm_url"]
    if "summary_mode" in toml:
        cfg.summary_mode = toml["summary_mode"]
    if "db_path" in toml:
        cfg.db_path = Path(toml["db_path"])
    if "tracker_backend" in toml:
        cfg.tracker_backend = toml["tracker_backend"]
    return cfg


def _apply_env(cfg: KbConfig, toml: dict[str, str]) -> KbConfig:
    """Apply KB_* env vars to cfg; log if overriding a toml value."""
    env = os.environ

    url = env.get("KB_EMBEDDING_URL", "")
    if url:
        toml_v = toml.get("embedding_url", "")
        if toml_v and url != toml_v:
            _log.info("kb config: KB_EMBEDDING_URL env=%r overrides toml=%r", url, toml_v)
            print(f"kb config: KB_EMBEDDING_URL env={url!r} overrides toml={toml_v!r}", file=sys.stderr)
        cfg.embedding_url = url

    dim_str = env.get("KB_EMBEDDING_DIM", "")
    if dim_str:
        try:
            dim = int(dim_str)
            toml_v = toml.get("embedding_dim", "")
            if toml_v and dim_str != toml_v:
                _log.info("kb config: KB_EMBEDDING_DIM env=%r overrides toml=%r", dim_str, toml_v)
                print(f"kb config: KB_EMBEDDING_DIM env={dim_str!r} overrides toml={toml_v!r}", file=sys.stderr)
            cfg.embedding_dim = dim
        except (ValueError, TypeError):
            pass

    fmt = env.get("KB_EMBEDDING_FORMAT", "")
    if fmt:
        toml_v = toml.get("embedding_format", "")
        if toml_v and fmt != toml_v:
            _log.info("kb config: KB_EMBEDDING_FORMAT env=%r overrides toml=%r", fmt, toml_v)
            print(f"kb config: KB_EMBEDDING_FORMAT env={fmt!r} overrides toml={toml_v!r}", file=sys.stderr)
        cfg.embedding_format = fmt

    model = env.get("KB_EMBEDDING_MODEL", "")
    if model:
        toml_v = toml.get("embedding_model", "")
        if toml_v and model != toml_v:
            _log.info("kb config: KB_EMBEDDING_MODEL env=%r overrides toml=%r", model, toml_v)
            print(f"kb config: KB_EMBEDDING_MODEL env={model!r} overrides toml={toml_v!r}", file=sys.stderr)
        cfg.embedding_model = model

    key = env.get("KB_EMBEDDING_KEY", "")
    if key:
        cfg.embedding_key = key

    llm = env.get("KB_LLM_URL", "")
    if llm:
        cfg.llm_url = llm

    sm = env.get("KB_SUMMARY_MODE", "")
    if sm:
        cfg.summary_mode = sm

    db_env = env.get("KB_DB", "")
    if db_env:
        cfg.db_path = Path(db_env)

    return cfg
