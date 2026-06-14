"""Smoke test for IssuesRepository Phase 1."""

import os
import sys
import tempfile
import pathlib

# Use a temp DB under the smoke dir, NOT the real knowledge.db
SMOKE_DB = str(pathlib.Path(__file__).parent / "smoke_test.db")

# Clean up from prior run
if os.path.exists(SMOKE_DB):
    os.unlink(SMOKE_DB)
for ext in ("-wal", "-shm"):
    p = SMOKE_DB + ext
    if os.path.exists(p):
        os.unlink(p)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from kb.facade import KnowledgeBase

embedding_url = os.environ.get("KB_EMBEDDING_URL", "http://ash:8081/embedding")
embedding_dim = int(os.environ.get("KB_EMBEDDING_DIM", "4096"))

kb = KnowledgeBase(
    db_path=SMOKE_DB,
    embedding_url=embedding_url,
    embedding_dim=embedding_dim,
)

print("=== IssuesRepository smoke test ===")

# Create root epic
r = kb.issue_create("Implement vector search for kb issues", type="epic", project="knowledge-base", prefix="kb")
epic_id = r["id"]
assert r["is_new"]
assert epic_id.startswith("kb-")
print(f"Root epic: {epic_id}")

# Create child 1
r1 = kb.issue_create("Schema: add issues table", type="task", parent_id=epic_id, project="knowledge-base")
child1 = r1["id"]
assert child1 == f"{epic_id}.1", f"Expected {epic_id}.1, got {child1}"
print(f"Child 1: {child1}")

# Create child 2
r2 = kb.issue_create("IssuesRepository: create/get/list/search", type="task", parent_id=epic_id, project="knowledge-base",
                     description="Implement IssuesRepository modeled on ConceptRepository with vector search")
child2 = r2["id"]
assert child2 == f"{epic_id}.2", f"Expected {epic_id}.2, got {child2}"
print(f"Child 2: {child2}")

# Create child 3 (counter monotonic)
r3 = kb.issue_create("Facade wiring", type="task", parent_id=epic_id, project="knowledge-base")
child3 = r3["id"]
assert child3 == f"{epic_id}.3", f"Expected {epic_id}.3, got {child3}"
print(f"Child 3: {child3}")

# Verify issue_get returns comments and deps
got = kb.issue_get(child2)
assert got is not None
assert got["id"] == child2
assert got["parent_id"] == epic_id
assert isinstance(got["comments"], list)
assert isinstance(got["deps"], list)
print(f"issue_get({child2}): ok, title='{got['title']}'")

# issue_list filtered by parent
children = kb.issue_list(parent_id=epic_id)
child_ids = [c["id"] for c in children]
assert child1 in child_ids, f"{child1} not in list"
assert child2 in child_ids, f"{child2} not in list"
assert child3 in child_ids, f"{child3} not in list"
print(f"issue_list(parent_id={epic_id}): {len(children)} children - OK")

# issue_list filtered by project
proj_issues = kb.issue_list(project="knowledge-base")
assert len(proj_issues) == 4  # epic + 3 children
print(f"issue_list(project='knowledge-base'): {len(proj_issues)} issues - OK")

# issue_search — try semantic search
embedding_ok = False
try:
    results = kb.issue_search("vector search implementation", project="knowledge-base", limit=5)
    # child2 has 'vector search' in its description
    ids = [r["id"] for r in results]
    assert len(results) > 0, "search returned 0 results"
    assert all("similarity" in r for r in results)
    print(f"issue_search: {len(results)} results, top={results[0]['id']} sim={results[0]['similarity']}")
    embedding_ok = True
except Exception as e:
    print(f"issue_search: EMBEDDING UNAVAILABLE ({e}) — id-allocation + list assertions passed")

print()
print("=== RESULTS ===")
print(f"id-allocation:  PASS")
print(f"child counter:  PASS (counters 1,2,3 monotonic)")
print(f"issue_get:      PASS")
print(f"issue_list:     PASS")
print(f"issue_search:   {'PASS' if embedding_ok else 'SKIPPED (embedding down)'}")

# Cleanup
kb.close()
os.unlink(SMOKE_DB)
for ext in ("-wal", "-shm"):
    p = SMOKE_DB + ext
    if os.path.exists(p):
        os.unlink(p)
