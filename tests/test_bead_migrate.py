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
    ("closed", "closed"), ("deferred", "deferred"), ("pinned", "open"),
    (None, "open"), ("DEFERRED", "deferred"),
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


def test_live_exceeds_export_warns_and_continues(tmp_path, monkeypatch, capsys):
    """T1: live bd count > export issue count is a WARNING, not an abort.

    Real-world skew: bd export omits closed ephemeral *-wisp-* molecule sub-tasks
    that bd list --all counts (claude delta=19, secular delta=486). The migrate
    path must complete (return 0 in dry-run), print a WARNING, and NOT write the
    marker (dry-run skips step 5).
    """
    monkeypatch.setattr(issue_cli.shutil, "which", lambda n: "/usr/bin/bd")
    (tmp_path / ".beads").mkdir()
    # live=3 but export only has 2 issue lines (benign wisp-task delta)
    fake = _fake_run_factory([{"id": "x-0"}, {"id": "x-1"}, {"id": "x-2"}],
                             [_issue_line(0), _issue_line(1)])
    monkeypatch.setattr(subprocess, "run", fake)
    # import reports 2 — matches export_n (2), so no hard abort
    monkeypatch.setattr(issue_cli, "import_bd_export_safe",
                        lambda kb, p, dry_run, project: {"issues_imported": 2,
                                                         "deps_imported": 0, "comments_imported": 0})
    monkeypatch.setattr(bd_import, "verify_fidelity", lambda kb, p: [])
    monkeypatch.setattr(bd_import, "_build_test_kb", lambda p: object())
    monkeypatch.chdir(tmp_path)
    rc = issue_cli.cmd_bead_migrate(_args(dry_run=True), kb=None)
    assert rc == 0                                  # completes — no abort
    assert not (tmp_path / ".kbt").exists()         # dry-run: no marker
    assert (tmp_path / ".beads").exists()           # dry-run: .beads untouched
    captured = capsys.readouterr()
    assert "WARNING" in captured.err               # advisory warning was printed
    assert "delta" in captured.err                 # named the count delta


def test_export_internal_mismatch_aborts(tmp_path, monkeypatch):
    """T1: imported != export_n is a hard abort (genuine INSERT collapse / exception).

    If the importer reports fewer issues than the export contained (e.g. id collision
    caused INSERT OR REPLACE to silently collapse records), the migration aborts.
    """
    monkeypatch.setattr(issue_cli.shutil, "which", lambda n: "/usr/bin/bd")
    (tmp_path / ".beads").mkdir()
    # export has 3 issue lines; live also says 3 (no benign delta)
    fake = _fake_run_factory([{"id": "x-0"}, {"id": "x-1"}, {"id": "x-2"}],
                             [_issue_line(0), _issue_line(1), _issue_line(2)])
    monkeypatch.setattr(subprocess, "run", fake)
    # importer only reports 2 — mismatch against export_n=3 → hard abort
    monkeypatch.setattr(issue_cli, "import_bd_export_safe",
                        lambda kb, p, dry_run, project: {"issues_imported": 2,
                                                         "deps_imported": 0, "comments_imported": 0})
    monkeypatch.setattr(bd_import, "verify_fidelity", lambda kb, p: [])
    monkeypatch.setattr(bd_import, "_build_test_kb", lambda p: object())
    monkeypatch.chdir(tmp_path)
    rc = issue_cli.cmd_bead_migrate(_args(dry_run=True), kb=None)
    assert rc == 1
    assert not (tmp_path / ".kbt").exists()         # no marker on abort
    assert (tmp_path / ".beads").exists()            # .beads untouched


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


# --------------------------------------------------------------------------
# T3: id-collision pre-flight
# --------------------------------------------------------------------------
def _make_kb(tmp_path, name="kb.db"):
    from kb.bd_import import _build_test_kb
    return _build_test_kb(tmp_path / name)


def _write_export(tmp_path, lines, name="export.ndjson"):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_id_collision_different_content_aborts(tmp_path):
    """Two exports sharing an id with DIFFERENT content must abort (T3)."""
    kb = _make_kb(tmp_path)
    issue_a = {"_type": "issue", "id": "proj-aaa", "status": "open",
                "issue_type": "task", "title": "original", "priority": 2}
    export_a = _write_export(tmp_path, [json.dumps(issue_a)], "a.ndjson")
    bd_import.import_bd_export(kb, export_a, dry_run=False)

    # Same id, different title
    issue_b = dict(issue_a, title="CHANGED")
    export_b = _write_export(tmp_path, [json.dumps(issue_b)], "b.ndjson")
    with pytest.raises(ValueError, match="id-collision"):
        bd_import.import_bd_export(kb, export_b, dry_run=False)


def test_id_collision_identical_reimport_is_idempotent(tmp_path):
    """Re-importing identical content for the same id must not abort (T3)."""
    kb = _make_kb(tmp_path, "kb2.db")
    issue = {"_type": "issue", "id": "proj-bbb", "status": "open",
             "issue_type": "task", "title": "stable", "priority": 2}
    export = _write_export(tmp_path, [json.dumps(issue)], "same.ndjson")
    bd_import.import_bd_export(kb, export, dry_run=False)
    # Second import of identical data — must not raise
    stats = bd_import.import_bd_export(kb, export, dry_run=False)
    assert stats["issues_imported"] == 1


# --------------------------------------------------------------------------
# T4: deep verify_fidelity
# --------------------------------------------------------------------------
def test_verify_fidelity_catches_wrong_title(tmp_path):
    """A corrupted import (wrong title) is caught by verify_fidelity (T4)."""
    kb = _make_kb(tmp_path, "kb3.db")
    issue = {"_type": "issue", "id": "proj-ccc", "status": "open",
             "issue_type": "task", "title": "correct title", "priority": 2}
    export = _write_export(tmp_path, [json.dumps(issue)], "t4a.ndjson")
    bd_import.import_bd_export(kb, export, dry_run=False)

    # Corrupt the title in the db directly
    kb.conn.execute("UPDATE issues SET title='corrupted' WHERE id='proj-ccc'")
    kb.conn.commit()

    discrepancies = bd_import.verify_fidelity(kb, export)
    assert len(discrepancies) == 1
    assert any("title" in d for d in discrepancies[0]["diffs"])


def test_verify_fidelity_catches_dropped_comment(tmp_path):
    """A dropped comment is detected by verify_fidelity (T4)."""
    kb = _make_kb(tmp_path, "kb4.db")
    issue = {"_type": "issue", "id": "proj-ddd", "status": "open",
             "issue_type": "task", "title": "has comment", "priority": 2,
             "comments": [{"id": "cmt-t4", "issue_id": "proj-ddd",
                           "text": "hello", "author": "alice",
                           "created_at": "2025-01-01"}]}
    export = _write_export(tmp_path, [json.dumps(issue)], "t4b.ndjson")
    bd_import.import_bd_export(kb, export, dry_run=False)

    # Delete the comment to simulate a dropped import
    kb.conn.execute("DELETE FROM issue_comments WHERE id='cmt-t4'")
    kb.conn.commit()

    discrepancies = bd_import.verify_fidelity(kb, export)
    assert len(discrepancies) == 1
    assert any("comment" in d for d in discrepancies[0]["diffs"])


def test_verify_fidelity_clean_import_zero_discrepancies(tmp_path):
    """A clean import verifies with 0 discrepancies (T4)."""
    kb = _make_kb(tmp_path, "kb5.db")
    issue = {"_type": "issue", "id": "proj-eee", "status": "closed",
             "issue_type": "epic", "title": "clean", "priority": 1,
             "description": "desc", "notes": "note",
             "assignee": "bob", "created_at": "2025-01-01", "closed_at": "2025-06-01",
             "comments": [{"id": "cmt-t4c", "issue_id": "proj-eee",
                           "text": "done", "author": "bob",
                           "created_at": "2025-06-01"}]}
    export = _write_export(tmp_path, [json.dumps(issue)], "t4c.ndjson")
    bd_import.import_bd_export(kb, export, dry_run=False)

    discrepancies = bd_import.verify_fidelity(kb, export)
    assert discrepancies == []


# --------------------------------------------------------------------------
# T6: rowcount stats don't overcount on second identical import
# --------------------------------------------------------------------------
def test_stats_no_overcount_on_reimport(tmp_path):
    """deps_imported and comments_imported do not overcount on idempotent re-import (T6b)."""
    kb = _make_kb(tmp_path, "kb6.db")
    parent = {"_type": "issue", "id": "proj-fff", "status": "open",
              "issue_type": "epic", "title": "parent", "priority": 2}
    child = {"_type": "issue", "id": "proj-fff.1", "status": "open",
             "issue_type": "task", "title": "child", "priority": 2,
             "dependencies": [{"issue_id": "proj-fff.1", "depends_on_id": "proj-fff",
                               "type": "parent-child", "created_at": "",
                               "created_by": None}],
             "comments": [{"id": "cmt-rc1", "issue_id": "proj-fff.1",
                           "text": "hi", "author": "alice",
                           "created_at": "2025-01-01"}]}
    export = _write_export(tmp_path, [json.dumps(parent), json.dumps(child)], "t6.ndjson")

    stats1 = bd_import.import_bd_export(kb, export, dry_run=False)
    assert stats1["deps_imported"] == 1
    assert stats1["comments_imported"] == 1

    # Second import: INSERT OR IGNORE fires but rowcount=0 for existing rows
    stats2 = bd_import.import_bd_export(kb, export, dry_run=False)
    assert stats2["deps_imported"] == 0
    assert stats2["comments_imported"] == 0
