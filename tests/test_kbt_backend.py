"""resolve_backend precedence tests (kb-sg0.14, beadless backend selection).

Precedence (highest first):
  1. KBT_BACKEND env
  2. per-project .kbt/config.toml [tracker] backend  (walk up)
  3. legacy .beads/config.yaml backend:              (walk up; deprecation warn)
  4. host-wide ~/.config/kb/config.toml [tracker] backend
  5. default: dolt-if-bd (transitional notice) else kb

The two walk-ups are INDEPENDENT and SEQUENTIAL (.kbt to root first, then
.beads to root), so a per-project .kbt marker at a HIGHER ancestor beats a
legacy .beads at a LOWER ancestor — the per-project isolation guarantee.
"""
import pytest

from kb import issue_cli


@pytest.fixture(autouse=True)
def _no_host_tracker(monkeypatch):
    """Default: host-global [tracker] backend is unset (step 4 → None), so tests
    exercise steps 1-3 and 5 deterministically. Individual tests override."""
    class _Cfg:
        tracker_backend = None
    monkeypatch.setattr("kb.config.load_config", lambda force_reload=False: _Cfg())
    monkeypatch.delenv("KBT_BACKEND", raising=False)


def _write_kbt(d, backend):
    (d / ".kbt").mkdir(parents=True, exist_ok=True)
    (d / ".kbt" / "config.toml").write_text(f'[tracker]\nbackend = "{backend}"\n')


def _write_beads(d, backend):
    (d / ".beads").mkdir(parents=True, exist_ok=True)
    (d / ".beads" / "config.yaml").write_text(f"backend: {backend}\n")


def test_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("KBT_BACKEND", "kb")
    _write_beads(tmp_path, "dolt")
    assert issue_cli.resolve_backend(tmp_path) == "kb"


def test_kbt_marker_wins_over_beads_same_dir(tmp_path):
    _write_kbt(tmp_path, "kb")
    _write_beads(tmp_path, "dolt")
    assert issue_cli.resolve_backend(tmp_path) == "kb"


def test_kbt_higher_ancestor_beats_beads_lower(tmp_path):
    # THE WALK-ORDER TRAP: .kbt at root, .beads in a deeper subdir, cwd=subdir.
    # Independent sequential walks → .kbt (farther) must win over .beads (nearer).
    _write_kbt(tmp_path, "kb")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    _write_beads(sub, "dolt")
    assert issue_cli.resolve_backend(sub) == "kb"


def test_legacy_beads_read_and_warns(tmp_path, capsys):
    _write_beads(tmp_path, "dolt")
    assert issue_cli.resolve_backend(tmp_path) == "dolt"
    err = capsys.readouterr().err
    assert "deprecated" in err and "bead-migrate" in err


def test_host_global_toml(tmp_path, monkeypatch):
    class _Cfg:
        tracker_backend = "kb"
    monkeypatch.setattr("kb.config.load_config", lambda force_reload=False: _Cfg())
    # no per-project markers → falls through to host-global
    assert issue_cli.resolve_backend(tmp_path) == "kb"


def test_default_dolt_if_bd_with_notice(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(issue_cli.shutil, "which", lambda name: "/usr/bin/bd")
    assert issue_cli.resolve_backend(tmp_path) == "dolt"
    assert "bead-migrate" in capsys.readouterr().err


def test_default_kb_if_no_bd(tmp_path, monkeypatch):
    monkeypatch.setattr(issue_cli.shutil, "which", lambda name: None)
    assert issue_cli.resolve_backend(tmp_path) == "kb"


def test_kbt_marker_empty_backend_falls_through(tmp_path, monkeypatch):
    # A .kbt with no backend value must not short-circuit; falls to default.
    (tmp_path / ".kbt").mkdir()
    (tmp_path / ".kbt" / "config.toml").write_text("[tracker]\n")
    monkeypatch.setattr(issue_cli.shutil, "which", lambda name: None)
    assert issue_cli.resolve_backend(tmp_path) == "kb"


def test_config_toml_tracker_section_parsed(tmp_path):
    # B1 fix: config.py must read [tracker] backend (was silently dropped).
    from kb import config
    p = tmp_path / "config.toml"
    p.write_text('[tracker]\nbackend = "kb"\n')
    flat = config._load_toml(p)
    assert flat.get("tracker_backend") == "kb"
    import dataclasses
    cfg = config._apply_toml(dataclasses.replace(config._DEFAULTS), flat)
    assert cfg.tracker_backend == "kb"
    assert config._DEFAULTS.tracker_backend is None  # singleton not mutated
