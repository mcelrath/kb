"""resolve_backend precedence tests (kb-sg0.14, beadless backend selection).

Precedence (highest first):
  1. KBT_BACKEND env
  2. per-project .kbt/config.toml [tracker] backend  (walk up)
  3. legacy .beads/config.yaml with EXPLICIT backend: (escape hatch; deprecation warn)
  4. host-wide ~/.config/kb/config.toml [tracker] backend
  5. default: kb (warns if a non-empty legacy .beads is present)

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


def test_legacy_beads_explicit_backend_honored_and_warns(tmp_path, capsys):
    # An EXPLICIT backend: in .beads/config.yaml is still honored (escape hatch).
    _write_beads(tmp_path, "dolt")
    assert issue_cli.resolve_backend(tmp_path) == "dolt"
    err = capsys.readouterr().err
    assert "deprecated" in err and "bead-migrate" in err


def test_beads_without_explicit_backend_falls_through_to_kb(tmp_path, monkeypatch):
    # A .beads/config.yaml with NO backend: key must NOT route to dolt anymore.
    (tmp_path / ".beads").mkdir()
    (tmp_path / ".beads" / "config.yaml").write_text("dolt:\n  shared-server: true\n")
    monkeypatch.setattr(issue_cli.shutil, "which", lambda name: "/usr/bin/bd")
    assert issue_cli.resolve_backend(tmp_path) == "kb"


def test_nonempty_beads_warns_but_defaults_kb(tmp_path, monkeypatch, capsys):
    # Non-empty .beads (dolt DB dir present), no explicit backend, bd on PATH:
    # default to kb AND emit a migration-suggesting warning.
    (tmp_path / ".beads" / "dolt").mkdir(parents=True)
    monkeypatch.setattr(issue_cli.shutil, "which", lambda name: "/usr/bin/bd")
    assert issue_cli.resolve_backend(tmp_path) == "kb"
    err = capsys.readouterr().err
    assert "bead-migrate" in err and "ASK THE USER" in err


def test_empty_beads_no_warning(tmp_path, monkeypatch, capsys):
    # A config-only .beads (no dolt dir, no issues.jsonl) is NOT "non-empty":
    # default to kb with no migration warning.
    (tmp_path / ".beads").mkdir()
    (tmp_path / ".beads" / "config.yaml").write_text("dolt:\n  shared-server: true\n")
    monkeypatch.setattr(issue_cli.shutil, "which", lambda name: None)
    assert issue_cli.resolve_backend(tmp_path) == "kb"
    assert "bead-migrate" not in capsys.readouterr().err


def test_host_global_toml(tmp_path, monkeypatch):
    class _Cfg:
        tracker_backend = "kb"
    monkeypatch.setattr("kb.config.load_config", lambda force_reload=False: _Cfg())
    # no per-project markers → falls through to host-global
    assert issue_cli.resolve_backend(tmp_path) == "kb"


def test_default_kb_even_with_bd(tmp_path, monkeypatch, capsys):
    # Cutover: bd on PATH no longer routes to dolt; no .beads → kb, no warning.
    monkeypatch.setattr(issue_cli.shutil, "which", lambda name: "/usr/bin/bd")
    assert issue_cli.resolve_backend(tmp_path) == "kb"
    assert "bead-migrate" not in capsys.readouterr().err


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


# --------------------------------------------------------------------------
# T5: 'deferred' status accepted by ISSUE_STATUSES and IssuesRepository
# --------------------------------------------------------------------------
def test_deferred_status_accepted_by_issue_statuses():
    """T5: ISSUE_STATUSES must include 'deferred' so create/set_status accept it."""
    from kb.entities.issues import ISSUE_STATUSES
    assert "deferred" in ISSUE_STATUSES


def test_deferred_status_create_and_set(tmp_path):
    """T5: IssuesRepository.create and set_status must accept 'deferred'."""
    from kb.bd_import import _build_test_kb
    from kb.entities.issues import IssuesRepository
    kb = _build_test_kb(tmp_path / "t5.db")
    repo = IssuesRepository(kb.conn, kb._embedding)

    # create with status=deferred must not raise
    result = repo.create(title="deferred task", status="deferred", prefix="tst")
    issue_id = result["id"]
    assert result["is_new"]

    # set_status to deferred on an existing issue must not raise
    row = kb.conn.execute("SELECT status FROM issues WHERE id=?", (issue_id,)).fetchone()
    assert row[0] == "deferred"

    # transition back to open must work
    repo.set_status(issue_id, "open")
    row = kb.conn.execute("SELECT status FROM issues WHERE id=?", (issue_id,)).fetchone()
    assert row[0] == "open"

    # set to deferred again
    repo.set_status(issue_id, "deferred")
    row = kb.conn.execute("SELECT status FROM issues WHERE id=?", (issue_id,)).fetchone()
    assert row[0] == "deferred"
