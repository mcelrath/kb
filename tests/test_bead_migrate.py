"""kbt bead-migrate safety-gate tests (kb-sg0.15) + importer status normalization.

The migrate command must NEVER write the marker or delete .beads/ unless the
import is verified complete against an INDEPENDENT live count and per-issue
fidelity passes (B4), and it must archive .beads/ before removal (B5). These
tests mock the `bd` subprocess so no live dolt server is required.
"""
import json
import subprocess
import types

import pytest

from kb import issue_cli
from kb import bd_import


# --------------------------------------------------------------------------
# importer status normalization (the deferred → open CHECK fix)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("given,expect", [
    ("open", "open"), ("in_progress", "in_progress"), ("blocked", "blocked"),
    ("closed", "closed"), ("deferred", "open"), ("pinned", "open"),
    (None, "open"), ("DEFERRED", "open"),
])
def test_normalize_status(given, expect):
    assert bd_import._normalize_status(given) == expect


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def test_write_kbt_marker(tmp_path):
    m = issue_cli._write_kbt_marker(tmp_path)
    assert m == tmp_path / ".kbt" / "config.toml"
    assert 'backend = "kb"' in m.read_text()
    assert issue_cli.resolve_backend(tmp_path) == "kb"  # round-trips


def test_find_beads_dir(tmp_path):
    (tmp_path / ".beads").mkdir()
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert issue_cli._find_beads_dir(sub) == tmp_path / ".beads"


# --------------------------------------------------------------------------
# bead-migrate gates (mocked bd)
# --------------------------------------------------------------------------
def _args(dry_run=False, keep_beads=False):
    return types.SimpleNamespace(dry_run=dry_run, keep_beads=keep_beads, project=None)


def _fake_run_factory(list_json, export_lines, list_rc=0, export_rc=0):
    """Build a subprocess.run replacement that answers bd list / bd export and
    delegates git/tar to the real subprocess.run."""
    real = subprocess.run

    def fake(cmd, *a, **k):
        if cmd[:2] == ["bd", "list"]:
            return types.SimpleNamespace(returncode=list_rc, stdout=json.dumps(list_json), stderr="")
        if cmd[:2] == ["bd", "export"]:
            # cmd = ["bd","export","-o",path]
            out = cmd[3]
            open(out, "w").write("\n".join(export_lines) + ("\n" if export_lines else ""))
            return types.SimpleNamespace(returncode=export_rc, stdout="", stderr="")
        return real(cmd, *a, **k)
    return fake


def _issue_line(i):
    return json.dumps({"_type": "issue", "id": f"x-{i}", "status": "open",
                       "issue_type": "task", "title": f"t{i}", "priority": 2})


def test_bd_absent_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(issue_cli.shutil, "which", lambda n: None)
    monkeypatch.chdir(tmp_path)
    assert issue_cli.cmd_bead_migrate(_args(), kb=None) == 0


def test_bd_list_failure_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(issue_cli.shutil, "which", lambda n: "/usr/bin/bd")
    monkeypatch.setattr(subprocess, "run", _fake_run_factory([], [], list_rc=1))
    monkeypatch.chdir(tmp_path)
    assert issue_cli.cmd_bead_migrate(_args(), kb=None) == 1


def test_truncated_export_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(issue_cli.shutil, "which", lambda n: "/usr/bin/bd")
    # live says 2; export has one good line + one truncated (unparseable) line
    fake = _fake_run_factory([{"id": "x-0"}, {"id": "x-1"}],
                             [_issue_line(0), '{"_type":"issue","id":"x-1"'])  # truncated
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.chdir(tmp_path)
    assert issue_cli.cmd_bead_migrate(_args(), kb=None) == 1


def test_count_mismatch_aborts_no_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(issue_cli.shutil, "which", lambda n: "/usr/bin/bd")
    (tmp_path / ".beads").mkdir()
    # live=3 but export only yields 2 valid issue lines
    fake = _fake_run_factory([{"id": "x-0"}, {"id": "x-1"}, {"id": "x-2"}],
                             [_issue_line(0), _issue_line(1)])
    monkeypatch.setattr(subprocess, "run", fake)
    # import would report 2 imported (mock so we don't need a real kb)
    monkeypatch.setattr(issue_cli, "import_bd_export_safe",
                        lambda kb, p, dry_run, project: {"issues_imported": 2,
                                                         "deps_imported": 0, "comments_imported": 0})
    monkeypatch.setattr(bd_import, "verify_fidelity", lambda kb, p: [])
    monkeypatch.setattr(bd_import, "_build_test_kb", lambda p: object())
    monkeypatch.chdir(tmp_path)
    rc = issue_cli.cmd_bead_migrate(_args(dry_run=True), kb=None)
    assert rc == 1
    assert not (tmp_path / ".kbt").exists()        # no marker on abort
    assert (tmp_path / ".beads").exists()           # .beads untouched


def test_fidelity_failure_aborts_no_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(issue_cli.shutil, "which", lambda n: "/usr/bin/bd")
    (tmp_path / ".beads").mkdir()
    fake = _fake_run_factory([{"id": "x-0"}], [_issue_line(0)])
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(issue_cli, "import_bd_export_safe",
                        lambda kb, p, dry_run, project: {"issues_imported": 1,
                                                         "deps_imported": 0, "comments_imported": 0})
    monkeypatch.setattr(bd_import, "verify_fidelity",
                        lambda kb, p: [{"id": "x-0", "diffs": ["status mismatch"]}])
    monkeypatch.setattr(bd_import, "_build_test_kb", lambda p: object())
    monkeypatch.chdir(tmp_path)
    rc = issue_cli.cmd_bead_migrate(_args(dry_run=True), kb=None)
    assert rc == 1
    assert (tmp_path / ".beads").exists()


def test_happy_path_writes_marker_and_archives_beads(tmp_path, monkeypatch):
    # real temp git repo + real temp kb db + fake bd (git delegated to real run)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / ".beads").mkdir()
    (tmp_path / ".beads" / "issues.jsonl").write_text("{}\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "--no-gpg-sign", "-m", "init"], check=True)
    # make .beads dirty so the archive step has something to commit (the real-repo case)
    (tmp_path / ".beads" / "issues.jsonl").write_text('{"changed":1}\n')

    monkeypatch.setattr(issue_cli.shutil, "which", lambda n: "/usr/bin/bd")
    monkeypatch.setattr(subprocess, "run",
                        _fake_run_factory([{"id": "x-0"}, {"id": "x-1"}],
                                          [_issue_line(0), _issue_line(1)]))
    monkeypatch.chdir(tmp_path)

    from kb.bd_import import _build_test_kb
    kb = _build_test_kb(tmp_path / "kb.db")

    rc = issue_cli.cmd_bead_migrate(_args(), kb=kb)
    assert rc == 0
    assert (tmp_path / ".kbt" / "config.toml").exists()      # marker written
    assert not (tmp_path / ".beads").exists()                # .beads removed
    # archive commit landed before delete (COMMIT-BEFORE-CLOBBER); the fake
    # subprocess.run delegates git to the real one, so this inspects real history.
    out = subprocess.run(["git", "-C", str(tmp_path), "log", "--oneline"],
                         capture_output=True, text=True)
    assert "bead-migrate" in out.stdout
