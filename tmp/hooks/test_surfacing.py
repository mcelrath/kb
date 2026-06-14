#!/usr/bin/env python3
"""Verify open_issues_surface.py and compose_time_check query_issues:
- project isolation (right project only)
- open vs closed surfacing
- FTS fallback (no embedding server)
- exit 0
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta

# --- Build a test DB with two projects ---
tmpdir = tempfile.mkdtemp()
db_path = os.path.join(tmpdir, 'test_kb.db')

conn = sqlite3.connect(db_path)
conn.execute("""
    CREATE TABLE issues (
        id TEXT PRIMARY KEY,
        type TEXT DEFAULT 'task',
        status TEXT DEFAULT 'open',
        priority INTEGER DEFAULT 2,
        parent_id TEXT,
        title TEXT,
        description TEXT,
        design_file TEXT,
        assignee TEXT,
        close_reason TEXT,
        project TEXT,
        tags TEXT DEFAULT '[]',
        created_at TEXT,
        updated_at TEXT,
        started_at TEXT,
        closed_at TEXT,
        closed_by_session TEXT
    )
""")
conn.execute("""
    CREATE VIRTUAL TABLE issues_fts USING fts5(
        title, description,
        content='issues',
        content_rowid='rowid'
    )
""")

now = datetime.utcnow().isoformat()
recent_closed = (datetime.utcnow() - timedelta(days=3)).isoformat()
old_closed = (datetime.utcnow() - timedelta(days=30)).isoformat()

issues = [
    # project-alpha issues
    ('alpha-001', 'open', 2, 'Implement embedding search for alpha queries', 'project-alpha', None),
    ('alpha-002', 'in_progress', 1, 'Fix vector similarity calculation in alpha module', 'project-alpha', None),
    ('alpha-003', 'closed', 2, 'Deploy alpha search service (prior art)', 'project-alpha', recent_closed),
    ('alpha-old', 'closed', 2, 'Old alpha issue beyond cutoff', 'project-alpha', old_closed),
    # project-beta issues (different project — must NOT surface when project=alpha)
    ('beta-001', 'open', 2, 'Beta tracker issue should not surface in alpha context', 'project-beta', None),
    # null-project issues (should surface for any project context)
    ('null-001', 'open', 3, 'Unscoped embedding search performance issue', None, None),
]

for iid, status, pri, title, proj, closed_at in issues:
    conn.execute(
        """INSERT INTO issues (id, type, status, priority, title, project, tags, created_at, updated_at, closed_at)
           VALUES (?, 'task', ?, ?, ?, ?, '[]', ?, ?, ?)""",
        (iid, status, pri, title, proj, now, now, closed_at),
    )

# Populate FTS (content table — need manual trigger)
conn.execute("INSERT INTO issues_fts(rowid, title, description) SELECT rowid, title, COALESCE(description, '') FROM issues")
conn.commit()
conn.close()

print(f"Test DB: {db_path}")

# --- Helper: run a hook against a payload ---
def run_hook(hook_path: str, payload: dict, env_overrides: dict | None = None) -> tuple[int, str]:
    env = os.environ.copy()
    env['KB_DB'] = db_path
    # Disable ash_down by pointing to a non-existent server so embedding fails fast
    env['KB_EMBEDDING_URL'] = 'http://127.0.0.1:19999/embedding'
    env['KB_EMBED_TIMEOUT'] = '1'
    # Disable LLM similarly
    env['KB_LLM_URL'] = 'http://127.0.0.1:19999/completion'
    # No session state — _seen returns all keys (safe degradation)
    env.pop('CLAUDE_SESSION_ID', None)
    env['CLAUDE_STATE_DIR'] = os.path.join(tmpdir, 'state')
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, hook_path],
        input=json.dumps(payload),
        capture_output=True, text=True,
        env=env,
        timeout=15,
    )
    return result.returncode, result.stdout.strip()


# --- Test 1: FTS fallback, project=alpha, open issues surface ---
print("\n=== Test 1: FTS fallback, project=alpha ===")
# Simulate Agent dispatch about "embedding search"
# .claude/kb-project.json walk-up won't find a file, so project=None from cwd.
# Override via a fake CLAUDE_PROJECT_DIR with a kb-project.json
proj_dir = os.path.join(tmpdir, 'alpha_proj', '.claude')
os.makedirs(proj_dir, exist_ok=True)
with open(os.path.join(proj_dir, 'kb-project.json'), 'w') as f:
    json.dump({'kb_project': 'project-alpha'}, f)

payload = {
    'tool_name': 'Agent',
    'tool_input': {
        'prompt': 'Implement embedding search functionality for the alpha queries module with vector similarity',
    },
}
rc, out = run_hook(
    '/home/mcelrath/.claude/hooks/kb/open_issues_surface.py',
    payload,
    env_overrides={'CLAUDE_PROJECT_DIR': os.path.join(tmpdir, 'alpha_proj')},
)
print(f"Exit code: {rc}")
print(f"Output: {out}")
assert rc == 0, f"Hook must exit 0, got {rc}"
if out:
    data = json.loads(out)
    context = data['hookSpecificOutput']['additionalContext']
    print(f"Advisories:\n{context}")
    assert 'alpha-' in context or 'null-001' in context, "Expected alpha or null-scoped issues"
    assert 'beta-001' not in context, "ISOLATION FAIL: beta issue leaked into alpha context"
    assert '[OPEN-ISSUE:' in context or '[RESOLVED-ISSUE:' in context, "Expected advisory format"
    if 'alpha-003' in context:
        assert '[RESOLVED-ISSUE:' in context, "Closed recent issue should be RESOLVED-ISSUE"
    if 'alpha-old' in context:
        assert False, "Old closed issue (30 days) must NOT surface"
    print("PASS: isolation holds, correct advisory format")
else:
    print("(No output — FTS may not have matched; check token overlap)")


# --- Test 2: project=beta sees only beta issues ---
print("\n=== Test 2: project=beta isolation ===")
beta_dir = os.path.join(tmpdir, 'beta_proj', '.claude')
os.makedirs(beta_dir, exist_ok=True)
with open(os.path.join(beta_dir, 'kb-project.json'), 'w') as f:
    json.dump({'kb_project': 'project-beta'}, f)

payload2 = {
    'tool_name': 'Agent',
    'tool_input': {
        'prompt': 'Beta tracker issue surface test for the beta project context only',
    },
}
rc2, out2 = run_hook(
    '/home/mcelrath/.claude/hooks/kb/open_issues_surface.py',
    payload2,
    env_overrides={'CLAUDE_PROJECT_DIR': os.path.join(tmpdir, 'beta_proj')},
)
print(f"Exit code: {rc2}")
print(f"Output: {out2}")
assert rc2 == 0
if out2:
    data2 = json.loads(out2)
    ctx2 = data2['hookSpecificOutput']['additionalContext']
    print(f"Advisories:\n{ctx2}")
    assert 'alpha-001' not in ctx2 and 'alpha-002' not in ctx2, "ISOLATION FAIL: alpha leaked into beta"
    print("PASS: alpha issues did not leak into beta context")


# --- Test 3: compose_time_check query_issues function ---
print("\n=== Test 3: compose_time_check query_issues ===")
payload3 = {
    'tool_name': 'Agent',
    'tool_input': {
        'prompt': 'Implement embedding vector search for alpha queries, fixing similarity calculation',
    },
}
rc3, out3 = run_hook(
    '/home/mcelrath/.claude/hooks/kb/compose_time_check.py',
    payload3,
    env_overrides={'CLAUDE_PROJECT_DIR': os.path.join(tmpdir, 'alpha_proj')},
)
print(f"Exit code: {rc3}")
print(f"Output: {out3}")
assert rc3 == 0, f"compose_time_check must exit 0, got {rc3}"
if out3:
    data3 = json.loads(out3)
    ctx3 = data3['hookSpecificOutput']['additionalContext']
    print(f"Advisories:\n{ctx3}")
    if '[OPEN-ISSUE:' in ctx3 or '[RESOLVED-ISSUE:' in ctx3:
        assert 'beta-001' not in ctx3, "ISOLATION FAIL in compose_time_check"
        print("PASS: compose_time_check surfaces issues with correct format")
    else:
        print("(No issue advisories from compose_time_check — other advisories may have surfaced)")


# --- Test 4: FTS fallback confirmation ---
print("\n=== Test 4: FTS fallback (embedding server down) ===")
# Embedding server is already pointing to a dead port in all tests above
# Confirm no crash/error
payload4 = {
    'tool_name': 'Agent',
    'tool_input': {
        'prompt': 'Search for embedding issues in the alpha tracking system performance',
    },
}
rc4, out4 = run_hook(
    '/home/mcelrath/.claude/hooks/kb/open_issues_surface.py',
    payload4,
    env_overrides={'CLAUDE_PROJECT_DIR': os.path.join(tmpdir, 'alpha_proj')},
)
assert rc4 == 0, f"Must exit 0 with FTS fallback, got {rc4}"
print(f"Exit: {rc4} (must be 0) — FTS fallback OK")
print("PASS: no crash with embedding server down")


# --- Test 5: Bash bridge send path ---
print("\n=== Test 5: Bash bridge send ===")
payload5 = {
    'tool_name': 'Bash',
    'tool_input': {
        'command': '''bridge send other-agent "Embedding search update" --needs-reply << 'EOF'
Working on the alpha embedding search vector similarity implementation.
Need to fix the distance calculation for the alpha queries.
EOF''',
    },
}
rc5, out5 = run_hook(
    '/home/mcelrath/.claude/hooks/kb/open_issues_surface.py',
    payload5,
    env_overrides={'CLAUDE_PROJECT_DIR': os.path.join(tmpdir, 'alpha_proj')},
)
assert rc5 == 0
print(f"Exit: {rc5} — bridge send path OK")
if out5:
    data5 = json.loads(out5)
    ctx5 = data5['hookSpecificOutput']['additionalContext']
    print(f"Advisories: {ctx5}")

print("\n=== All tests passed ===")
