"""
Phase 2 tests for IssuesRepository.

Uses a TEMP SQLite DB under ./tmp/issues_p2/ (NOT ~/.cache/kb).
No embedding calls needed: uses a stub EmbeddingService.
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the kb package is importable from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import sqlite_vec

from kb.core.schema import init_schema
from kb.core.embedding import EmbeddingService
from kb.entities.issues import IssuesRepository


class StubEmbeddingService(EmbeddingService):
    """Minimal stub: returns a fixed-length zero vector. No network calls."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> bytes:
        import struct
        floats = [0.0] * self.dim
        return struct.pack(f"{self.dim}f", *floats)


def make_repo(db_path: str) -> tuple[sqlite3.Connection, IssuesRepository]:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    emb = StubEmbeddingService(dim=128)
    init_schema(conn, embedding_dim=128)
    repo = IssuesRepository(conn, emb)
    return conn, repo


class TestPhase2(unittest.TestCase):

    def setUp(self):
        # Each test gets a fresh DB under tmp/issues_p2/
        db_dir = Path(__file__).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        fd, self.db_path = tempfile.mkstemp(suffix=".db", dir=str(db_dir))
        os.close(fd)
        self.conn, self.repo = make_repo(self.db_path)

    def tearDown(self):
        self.conn.close()
        Path(self.db_path).unlink(missing_ok=True)

    def _create_issue(self, title, type="task", **kw):
        return self.repo.create(title, type=type, **kw)

    # ------------------------------------------------------------------
    # Helper: build the canonical DAG used across multiple tests
    # epic E; tasks A, B, C children of E; A blocks B; C no blockers.
    # ------------------------------------------------------------------
    def _build_dag(self):
        e = self._create_issue("Epic E", type="epic", prefix="ep")
        eid = e["id"]
        a = self._create_issue("Task A", parent_id=eid)
        b = self._create_issue("Task B", parent_id=eid)
        c = self._create_issue("Task C", parent_id=eid)
        # B depends_on A with type 'blocks' (A blocks B)
        self.repo.add_dep(b["id"], a["id"], "blocks")
        return eid, a["id"], b["id"], c["id"]

    # ------------------------------------------------------------------
    # Test: add_dep / list_deps
    # ------------------------------------------------------------------
    def test_add_dep_basic(self):
        x = self._create_issue("X")["id"]
        y = self._create_issue("Y")["id"]
        r = self.repo.add_dep(x, y, "related")
        self.assertTrue(r["is_new"])
        self.assertEqual(r["type"], "related")

        # Idempotent: second call returns is_new=False
        r2 = self.repo.add_dep(x, y, "related")
        self.assertFalse(r2["is_new"])

    def test_add_dep_invalid_type(self):
        x = self._create_issue("X")["id"]
        y = self._create_issue("Y")["id"]
        with self.assertRaises(ValueError):
            self.repo.add_dep(x, y, "invalid-type")

    def test_list_deps_directions(self):
        x = self._create_issue("X")["id"]
        y = self._create_issue("Y")["id"]
        self.repo.add_dep(x, y, "blocks")  # X depends_on Y (Y blocks X)
        deps = self.repo.list_deps(x)
        self.assertEqual(len(deps["outgoing"]), 1)
        self.assertEqual(deps["outgoing"][0]["id"], y)
        self.assertEqual(deps["outgoing"][0]["type"], "blocks")
        self.assertEqual(len(deps["incoming"]), 0)

        deps_y = self.repo.list_deps(y)
        self.assertEqual(len(deps_y["incoming"]), 1)
        self.assertEqual(deps_y["incoming"][0]["id"], x)
        self.assertEqual(len(deps_y["outgoing"]), 0)

    # ------------------------------------------------------------------
    # Test: add_comment
    # ------------------------------------------------------------------
    def test_add_comment(self):
        iid = self._create_issue("Issue X")["id"]
        r = self.repo.add_comment(iid, "APPROVED: looks good", author="agent1")
        self.assertTrue(r["id"].startswith("cmt-"))
        issue = self.repo.get(iid)
        self.assertEqual(len(issue["comments"]), 1)
        self.assertEqual(issue["comments"][0]["body"], "APPROVED: looks good")

    # ------------------------------------------------------------------
    # Test: set_status
    # ------------------------------------------------------------------
    def test_set_status_transitions(self):
        iid = self._create_issue("Issue Y")["id"]
        r = self.repo.set_status(iid, "in_progress")
        self.assertEqual(r["status"], "in_progress")
        issue = self.repo.get(iid)
        self.assertIsNotNone(issue["started_at"])
        self.assertIsNone(issue["closed_at"])

        r2 = self.repo.set_status(iid, "closed", close_reason="done", closed_by_session="s1")
        self.assertEqual(r2["status"], "closed")
        issue2 = self.repo.get(iid)
        self.assertIsNotNone(issue2["closed_at"])
        self.assertEqual(issue2["close_reason"], "done")

    # ------------------------------------------------------------------
    # Test: ready() / blocked() with canonical DAG
    # ------------------------------------------------------------------
    def test_ready_basic(self):
        eid, aid, bid, cid = self._build_dag()
        ready_ids = {i["id"] for i in self.repo.ready()}
        # A and C are ready. B is blocked by A.
        # Epic E: open with open children but no direct 'blocks' dep on E itself
        # → E is ready per its OWN deps (it has none); child-inherited blocking
        # does NOT propagate upward to E, only downward. So E is ready.
        self.assertIn(aid, ready_ids, "A should be ready (no blockers)")
        self.assertIn(cid, ready_ids, "C should be ready (no blockers)")
        self.assertNotIn(bid, ready_ids, "B should NOT be ready (blocked by A)")

    def test_blocked_basic(self):
        eid, aid, bid, cid = self._build_dag()
        blocked = self.repo.blocked()
        blocked_ids = {i["id"] for i in blocked}
        self.assertIn(bid, blocked_ids, "B should be blocked")
        self.assertNotIn(aid, blocked_ids, "A should not be blocked")
        self.assertNotIn(cid, blocked_ids, "C should not be blocked")

        b_entry = next(i for i in blocked if i["id"] == bid)
        self.assertIn(aid, b_entry["blocker_ids"], "A should be listed as blocker for B")

    def test_close_blocker_unblocks_dependent(self):
        eid, aid, bid, cid = self._build_dag()

        # Before: B is blocked
        ready_ids_before = {i["id"] for i in self.repo.ready()}
        self.assertNotIn(bid, ready_ids_before)

        # Close A
        self.repo.set_status(aid, "closed", close_reason="done")

        # After: B should now be ready (A is closed, not open/in_progress)
        ready_ids_after = {i["id"] for i in self.repo.ready()}
        self.assertIn(bid, ready_ids_after, "B should be ready after A is closed")

        blocked_after = {i["id"] for i in self.repo.blocked()}
        self.assertNotIn(bid, blocked_after, "B should not be in blocked() after A is closed")

    # ------------------------------------------------------------------
    # Test: parent-blocked propagation
    # E has an open external blocker → E's children are NOT ready
    # (children inherit E's blocked state)
    # ------------------------------------------------------------------
    def test_parent_blocked_propagates_to_children(self):
        # Create a separate blocker issue
        blocker = self._create_issue("External Blocker")["id"]
        # Create epic E2 that is blocked by the external blocker
        e2 = self._create_issue("Epic E2", type="epic", prefix="ep2")["id"]
        self.repo.add_dep(e2, blocker, "blocks")  # E2 depends_on blocker
        # Create child task under E2
        child = self._create_issue("Child of E2", parent_id=e2)["id"]

        ready_ids = {i["id"] for i in self.repo.ready()}
        # E2 is blocked → child inherits the blocked state → child is NOT ready
        self.assertNotIn(e2, ready_ids, "E2 should not be ready (has open blocker)")
        self.assertNotIn(child, ready_ids, "child of blocked E2 should not be ready")

        blocked_ids = {i["id"] for i in self.repo.blocked()}
        self.assertIn(e2, blocked_ids, "E2 should be blocked")
        self.assertIn(child, blocked_ids, "child should be blocked (inherits E2's blocker)")

    # ------------------------------------------------------------------
    # Test: claim() atomic compare-and-swap
    # ------------------------------------------------------------------
    def test_claim_success(self):
        cid = self._create_issue("Task C claim test")["id"]
        r = self.repo.claim(cid, "agent1")
        self.assertTrue(r["claimed"])
        self.assertEqual(r["id"], cid)
        issue = self.repo.get(cid)
        self.assertEqual(issue["status"], "in_progress")
        self.assertEqual(issue["assignee"], "agent1")

    def test_claim_double_claim(self):
        """Second claim by a different agent must fail (already-claimed, not contended)."""
        cid = self._create_issue("Task for double-claim")["id"]
        r1 = self.repo.claim(cid, "agent1")
        self.assertTrue(r1["claimed"])

        r2 = self.repo.claim(cid, "agent2")
        self.assertFalse(r2["claimed"])
        self.assertTrue(r2.get("already"), "should be already=True, not contended")
        self.assertFalse(r2.get("contended", False))

        # Assignee must still be agent1
        issue = self.repo.get(cid)
        self.assertEqual(issue["assignee"], "agent1", "assignee must not change after failed claim")

    def test_claim_closed_issue_fails(self):
        cid = self._create_issue("Closed issue")["id"]
        self.repo.set_status(cid, "closed")
        r = self.repo.claim(cid, "agent1")
        self.assertFalse(r["claimed"])
        self.assertTrue(r.get("already"))

    # ------------------------------------------------------------------
    # Test: 'discovered-from' and 'related' do NOT affect readiness
    # ------------------------------------------------------------------
    def test_non_blocking_dep_types_do_not_block(self):
        x = self._create_issue("X")["id"]
        y = self._create_issue("Y")["id"]
        self.repo.add_dep(x, y, "discovered-from")
        self.repo.add_dep(x, y, "related")
        ready_ids = {i["id"] for i in self.repo.ready()}
        # Both X and Y should be ready; 'discovered-from'/'related' don't block
        self.assertIn(x, ready_ids, "X should be ready; discovered-from/related don't block")
        self.assertIn(y, ready_ids, "Y should be ready")

    # ------------------------------------------------------------------
    # Test: project filter works
    # ------------------------------------------------------------------
    def test_project_filter(self):
        self._create_issue("Issue P1", project="proj1")
        self._create_issue("Issue P2", project="proj2")
        ready_p1 = self.repo.ready(project="proj1")
        ids = {i["id"] for i in ready_p1}
        titles = {i["title"] for i in ready_p1}
        self.assertIn("Issue P1", titles)
        self.assertNotIn("Issue P2", titles)


if __name__ == "__main__":
    unittest.main(verbosity=2)
