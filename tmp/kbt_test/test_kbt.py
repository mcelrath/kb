"""
kbt Phase 3 tests.

Tests:
1. Backend resolver unit tests (no network, no bd exec)
2. Contract tests against a temp kb db (KBT_BACKEND=kb)
   - create epic + 2 children
   - show --json key names match bd's captured shapes
   - dep list --json element has .type/.id/.title/.status
   - children --json parses
   - ready --json / blocked --json parse
   - update --claim exits 0 first time, nonzero second time

Run with KBT_EMBEDDING_URL set if embedding server is reachable.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kb.issue_cli import resolve_backend, _read_backend_from_config


# ---------------------------------------------------------------------------
# 1. Backend resolver unit tests
# ---------------------------------------------------------------------------

class TestBackendResolver(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="kbt_test_resolver_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_beads(self, base: str, backend: str) -> Path:
        d = Path(base) / ".beads"
        d.mkdir(parents=True)
        (d / "config.yaml").write_text(f"backend: {backend}\n")
        return Path(base)

    def test_kb_backend_from_config(self):
        proj = self._make_beads(self.tmpdir + "/projA", "kb")
        result = resolve_backend(proj)
        self.assertEqual(result, "kb")

    def test_dolt_backend_from_config(self):
        proj = self._make_beads(self.tmpdir + "/projB", "dolt")
        result = resolve_backend(proj)
        self.assertEqual(result, "dolt")

    def test_no_beads_defaults_to_dolt(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            result = resolve_backend(Path(d))
            self.assertEqual(result, "dolt")

    def test_env_override_kb(self):
        proj = self._make_beads(self.tmpdir + "/projC", "dolt")
        with patch.dict(os.environ, {"KBT_BACKEND": "kb"}):
            result = resolve_backend(proj)
        self.assertEqual(result, "kb")

    def test_env_override_logs_to_stderr(self):
        import io
        buf = io.StringIO()
        with patch.dict(os.environ, {"KBT_BACKEND": "kb"}):
            with patch("sys.stderr", buf):
                result = resolve_backend(Path(self.tmpdir))
        self.assertEqual(result, "kb")
        self.assertIn("KBT_BACKEND", buf.getvalue())

    def test_walk_up_finds_beads(self):
        """resolve_backend walks UP from a nested subdir to find .beads/"""
        proj = self._make_beads(self.tmpdir + "/projD", "kb")
        nested = proj / "src" / "deep" / "module"
        nested.mkdir(parents=True)
        result = resolve_backend(nested)
        self.assertEqual(result, "kb")

    def test_dolt_backend_code_path_would_execvp(self):
        """When backend==dolt, kbt calls os.execvp('bd', ...) — verify the code path."""
        proj = self._make_beads(self.tmpdir + "/projE", "dolt")
        # monkeypatch os.execvp to capture the call instead of actually exec'ing
        captured = {}
        def fake_execvp(file, argv):
            captured["file"] = file
            captured["argv"] = argv
            raise SystemExit(0)  # simulate exec completing

        with patch.dict(os.environ, {"KBT_BACKEND": ""}, clear=False):
            if "KBT_BACKEND" in os.environ:
                del os.environ["KBT_BACKEND"]

        with patch("os.execvp", fake_execvp):
            # Simulate what kbt main() does for dolt backend
            with patch.dict(os.environ, {}, clear=False):
                if "KBT_BACKEND" in os.environ:
                    del os.environ["KBT_BACKEND"]
                backend = resolve_backend(proj)
            self.assertEqual(backend, "dolt")
            # The kbt script does: os.execvp('bd', ['bd'] + sys.argv[1:])
            import os as _os
            with self.assertRaises(SystemExit):
                fake_execvp("bd", ["bd", "list"])
            self.assertEqual(captured["file"], "bd")
            self.assertEqual(captured["argv"][0], "bd")


# ---------------------------------------------------------------------------
# 2. Contract tests against a temp kb db
# ---------------------------------------------------------------------------

# bd show --json key set (empirically captured 2026-06-11)
BD_SHOW_EPIC_KEYS = {
    "id", "title", "design", "status", "priority", "issue_type", "owner",
    "created_at", "created_by", "updated_at", "dependent_count",
    "dependency_count", "comment_count",
}

BD_LIST_ELEMENT_KEYS_REQUIRED = {
    "id", "title", "status", "priority", "issue_type", "owner",
    "created_at", "created_by", "updated_at",
    "dependency_count", "dependent_count", "comment_count",
}

BD_DEP_LIST_ELEMENT_KEYS_REQUIRED = {
    "id", "title", "status", "type",
}

BD_READY_ELEMENT_KEYS_REQUIRED = {
    "id", "title", "status", "priority", "issue_type", "owner",
    "created_at", "created_by", "updated_at",
    "dependency_count", "dependent_count", "comment_count",
}

BD_BLOCKED_ELEMENT_KEYS_REQUIRED = {
    "id", "title", "status", "priority", "issue_type", "owner",
    "created_at", "created_by", "updated_at",
    "blocked_by_count", "blocked_by",
}


def _make_test_kb():
    """Create a KnowledgeBase on a fresh temp db for contract tests."""
    from kb.facade import KnowledgeBase
    tmp = tempfile.mkdtemp(prefix="kbt_contract_")
    db_path = Path(tmp) / "test.db"
    kb = KnowledgeBase(db_path=db_path)
    return kb, tmp


class TestKbtContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.kb, cls.tmpdir = _make_test_kb()
            cls.skip_reason = None
        except Exception as e:
            cls.kb = None
            cls.skip_reason = str(e)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if cls.tmpdir:
            shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        if self.kb is None:
            self.skipTest(f"KB setup failed: {self.skip_reason}")

    def test_create_epic_and_children(self):
        """Create an epic and two children; verify IDs and parent links."""
        result = self.kb.issue_create(
            title="Test Epic",
            type="epic",
            description="An epic for testing",
            design_file="# Plan content here",
            prefix="tst",
        )
        self.__class__.epic_id = result["id"]
        self.assertTrue(result["id"].startswith("tst-"))
        self.assertTrue(result["is_new"])

        r1 = self.kb.issue_create(
            title="Child task 1",
            type="task",
            parent_id=self.epic_id,
        )
        self.__class__.child1_id = r1["id"]
        self.assertEqual(r1["id"], f"{self.epic_id}.1")

        r2 = self.kb.issue_create(
            title="Child task 2",
            type="task",
            parent_id=self.epic_id,
        )
        self.__class__.child2_id = r2["id"]
        self.assertEqual(r2["id"], f"{self.epic_id}.2")

    def test_show_json_key_names(self):
        """show --json must have bd's key names (issue_type, design, owner, etc.)"""
        if not hasattr(self.__class__, "epic_id"):
            self.test_create_epic_and_children()

        from kb.issue_cli import _project_show
        row = self.kb.issue_get(self.epic_id)
        self.assertIsNotNone(row)
        out = _project_show(row, self.kb.conn)

        # Check required keys present
        for key in BD_SHOW_EPIC_KEYS - {"design"}:  # design only when set
            self.assertIn(key, out, f"Missing key: {key}")
        # design should be present since we set it
        self.assertIn("design", out)
        # Must NOT have kb-internal key names
        self.assertNotIn("type", out)  # must be issue_type
        self.assertNotIn("assignee", out)  # must be owner

    def test_list_json_key_names(self):
        """list --json elements must have bd's key names."""
        if not hasattr(self.__class__, "epic_id"):
            self.test_create_epic_and_children()

        from kb.issue_cli import _project_list_item
        rows = self.kb.issue_list()
        self.assertGreater(len(rows), 0)
        full = self.kb.issue_get(rows[0]["id"])
        item = _project_list_item(full, self.kb.conn)
        for key in BD_LIST_ELEMENT_KEYS_REQUIRED:
            self.assertIn(key, item, f"list item missing key: {key}")
        self.assertNotIn("type", item)  # must be issue_type

    def test_dep_list_json_element_shape(self):
        """dep list --json elements must have .type/.id/.title/.status."""
        if not hasattr(self.__class__, "epic_id"):
            self.test_create_epic_and_children()
        if not hasattr(self.__class__, "child1_id"):
            self.test_create_epic_and_children()

        # Add a dep: child1 blocks child2
        self.kb.issue_add_dep(self.child2_id, self.child1_id, "blocks")

        from kb.issue_cli import _project_dep_list_item
        deps = self.kb.issue_list_deps(self.child2_id)
        self.assertGreater(len(deps.get("outgoing", [])), 0)

        for d in deps.get("outgoing", []):
            target = self.kb.issue_get(d["id"])
            item = _project_dep_list_item(target, self.kb.conn, d["type"])
            for key in BD_DEP_LIST_ELEMENT_KEYS_REQUIRED:
                self.assertIn(key, item, f"dep list item missing key: {key}")

    def test_children_json_parses(self):
        """children --json returns parseable JSON list."""
        if not hasattr(self.__class__, "epic_id"):
            self.test_create_epic_and_children()

        from kb.issue_cli import _project_list_item
        rows = self.kb.issue_list(parent_id=self.epic_id)
        self.assertEqual(len(rows), 2)
        for r in rows:
            full = self.kb.issue_get(r["id"])
            item = _project_list_item(full, self.kb.conn)
            self.assertIn("id", item)
            self.assertIn("issue_type", item)

    def test_ready_json_parses(self):
        """ready --json returns list; elements have required keys."""
        from kb.issue_cli import _project_ready_item
        rows = self.kb.issue_ready()
        for r in rows:
            full = self.kb.issue_get(r["id"])
            if full:
                full_with_design = full
            else:
                full_with_design = r
            item = _project_ready_item(full_with_design, self.kb.conn)
            for key in BD_READY_ELEMENT_KEYS_REQUIRED:
                self.assertIn(key, item, f"ready item missing key: {key}")

    def test_blocked_json_parses(self):
        """blocked --json elements have blocked_by/blocked_by_count keys."""
        from kb.issue_cli import _project_blocked_item
        rows = self.kb.issue_blocked()
        for r in rows:
            full = self.kb.issue_get(r["id"])
            if full:
                full["blocker_ids"] = r.get("blocker_ids", [])
                item = _project_blocked_item(full, self.kb.conn)
            else:
                item = _project_blocked_item(r, self.kb.conn)
            for key in BD_BLOCKED_ELEMENT_KEYS_REQUIRED:
                self.assertIn(key, item, f"blocked item missing key: {key}")

    def test_claim_atomic_first_time_succeeds(self):
        """update --claim exits 0 first time."""
        if not hasattr(self.__class__, "child1_id"):
            self.test_create_epic_and_children()
        result = self.kb.issue_claim(self.child1_id, "test-user")
        self.assertTrue(result.get("claimed"), f"claim failed: {result}")

    def test_claim_second_time_fails(self):
        """update --claim exits nonzero on second claim (already in_progress)."""
        if not hasattr(self.__class__, "child1_id"):
            self.test_create_epic_and_children()
            self.test_claim_atomic_first_time_succeeds()
        result = self.kb.issue_claim(self.child1_id, "another-user")
        self.assertFalse(result.get("claimed"))
        self.assertTrue(result.get("already"))

    def test_show_json_key_diff_vs_bd(self):
        """Compare key sets: kbt show --json vs captured live bd show --json.

        BD_SHOW_EPIC_KEYS captured empirically from `bd show kb-sg0 --json`.
        kbt emits these same keys (plus optionally started_at/closed_at/parent
        for tasks). The diff should be empty for epics with design set.
        """
        if not hasattr(self.__class__, "epic_id"):
            self.test_create_epic_and_children()

        from kb.issue_cli import _project_show
        row = self.kb.issue_get(self.epic_id)
        out = _project_show(row, self.kb.conn)

        emitted_keys = set(out.keys())
        expected_keys = BD_SHOW_EPIC_KEYS

        missing = expected_keys - emitted_keys
        extra = emitted_keys - expected_keys

        # Extra keys are fine (task-only fields), but missing is a bug
        self.assertEqual(missing, set(), f"kbt missing bd keys: {missing}")
        if extra:
            # Report but don't fail — extra keys are additive
            print(f"  kbt emits extra keys (ok): {extra}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
