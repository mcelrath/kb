"""R3 (kb-228cfb): _generate_summary routes through load_config().summary_mode,
not os.environ directly — so config.toml [llm] summary_mode is honored, and the
KB_SUMMARY_MODE env override (resolved by load_config) still wins."""

from pathlib import Path

import kb.config as config_mod
from kb.config import KbConfig
from kb.facade import KnowledgeBase


def _cfg(mode: str) -> KbConfig:
    return KbConfig(
        db_path=Path("/tmp/unused.db"),
        embedding_url="",
        embedding_dim=4,
        embedding_format="llamacpp",
        embedding_model="",
        embedding_key="",
        llm_url="",
        summary_mode=mode,
    )


def _bare_kb() -> KnowledgeBase:
    # _generate_summary for mode in {none, extractive} touches no instance state,
    # so a bare instance (no DB/embedding init) is sufficient to exercise routing.
    return object.__new__(KnowledgeBase)


def test_config_summary_mode_none_honored_when_env_unset(monkeypatch):
    monkeypatch.delenv("KB_SUMMARY_MODE", raising=False)
    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: _cfg("none"))
    # Old code defaulted to 'extractive' from os.environ and returned a blurb;
    # the fix reads config -> 'none' -> None.
    assert _bare_kb()._generate_summary("Some real content here.", None) is None


def test_env_override_wins_via_load_config(monkeypatch):
    # load_config layers env on top of toml; here it resolves to 'none'.
    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: _cfg("none"))
    assert _bare_kb()._generate_summary("Some real content here.", None) is None


def test_extractive_returns_blurb(monkeypatch):
    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: _cfg("extractive"))
    out = _bare_kb()._generate_summary("First sentence here. Second one.", None)
    assert out and isinstance(out, str)
