import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kb.facade import KnowledgeBase

db = Path(__file__).resolve().parent / "verify.db"
if db.exists():
    db.unlink()
for ext in ("-wal", "-shm"):
    p = Path(str(db) + ext)
    if p.exists():
        p.unlink()

kb = KnowledgeBase(
    db_path=db,
    embedding_url=os.environ.get("KB_EMBEDDING_URL", "http://ash:8081/embedding"),
    embedding_dim=int(os.environ.get("KB_EMBEDDING_DIM", "4096")),
)

epic = kb.issue_create("Replace beads tracker with kb-native issues", type="epic", project="ptest")
root = epic["id"]
print("root:", root)

c1 = kb.issue_create("Implement recursive ready/blocked SQL traversal", type="task", parent_id=root, project="ptest",
                     description="compute ready set via dependency closure")
c2 = kb.issue_create("Write the migration importer from dolt export", type="task", parent_id=root, project="ptest",
                     description="parse dolt json and insert into issues table")
c3 = kb.issue_create("Surfacing hook for open issues in context", type="task", parent_id=root, project="ptest",
                     description="inject open and closed issues via vector search")
print("children:", c1["id"], c2["id"], c3["id"])
assert c1["id"] == f"{root}.1", c1["id"]
assert c2["id"] == f"{root}.2", c2["id"]
assert c3["id"] == f"{root}.3", c3["id"]

g = kb.issue_get(root)
assert g["type"] == "epic" and g["comments"] == [] and g["deps"] == []
kids = kb.issue_list(parent_id=root)
assert len(kids) == 3, len(kids)

hits = kb.issue_search("dependency graph ready computation", project="ptest", limit=3)
print("search top:", [(h["id"], h["similarity"]) for h in hits])
assert hits, "no search hits"
assert hits[0]["id"] == c1["id"], f"expected {c1['id']} top, got {hits[0]['id']}"
assert all(h["similarity"] is not None for h in hits)
assert hits[0]["similarity"] > hits[-1]["similarity"] or len(hits) == 1, "not ranked by distance"

other = kb.issue_search("export and import database rows", project="ptest", limit=3)
print("search2 top:", [(h["id"], h["similarity"]) for h in other])
assert other[0]["id"] == c2["id"], f"expected {c2['id']} top, got {other[0]['id']}"

print("ALL ASSERTIONS PASS")
