#!/usr/bin/env python3
"""kb-2os FINAL SWAP: register the 7 rebuilt bridge+tracker hooks in the plugin
hooks.json, and deregister the 10 old copies from ~/.claude/settings.json.

Plugin hooks load at session STARTUP (no hot-reload); settings.json DOES hot-reload
— so the moment settings.json loses the old entries, this session (and the other
live sessions, on their next reload) lose the old bridge/tracker hooks until they
RESTART into the plugin versions. That dead window is the accepted cutover cost
('swap+restart last'). Idempotent; validates JSON; backs settings.json up first."""
import json, shutil, os, sys

PLUGIN = "/home/mcelrath/Projects/ai/kb/hooks/hooks.json"
SETTINGS = os.path.expanduser("~/.claude/settings.json")
PR = "${CLAUDE_PLUGIN_ROOT}/hooks/scripts"

def sh(name):  return {"type": "command", "command": f'bash "{PR}/{name}"', "timeout": 15}
def py(name, t=12): return {"type": "command", "command": f'"{PR}/{name}"', "timeout": t}

# ---- 1. Register in plugin hooks.json (additive, dedup by command substring) ----
pl = json.load(open(PLUGIN))
H = pl["hooks"]

def has(blocks, needle):
    return any(needle in h.get("command", "") for b in blocks for h in b.get("hooks", []))

def add_to_unmatched(event, entry, needle):
    """Append entry to the first block in `event` that has NO matcher."""
    H.setdefault(event, [])
    if has(H[event], needle): return f"  {event}: {needle} already present"
    for b in H[event]:
        if "matcher" not in b:
            b["hooks"].append(entry); return f"  {event}: +{needle}"
    H[event].append({"hooks": [entry]}); return f"  {event}: +{needle} (new block)"

def add_to_matched(event, matcher, entry, needle):
    H.setdefault(event, [])
    if has(H[event], needle): return f"  {event}[{matcher}]: {needle} already present"
    for b in H[event]:
        if b.get("matcher") == matcher:
            b["hooks"].append(entry); return f"  {event}[{matcher}]: +{needle}"
    H[event].append({"matcher": matcher, "hooks": [entry]}); return f"  {event}[{matcher}]: +{needle} (new block)"

print("PLUGIN hooks.json registrations:")
print(add_to_unmatched("SessionStart", sh("bridge-resume.sh"), "bridge-resume.sh"))
print(add_to_unmatched("SessionStart", sh("session-followups.sh"), "session-followups.sh"))
print(add_to_unmatched("UserPromptSubmit", sh("bridge-inject.sh"), "bridge-inject.sh"))
print(add_to_matched("PreToolUse", "Bash", sh("bridge-inject.sh"), "bridge-inject.sh"))
print(add_to_matched("PreToolUse", "Write|Edit", sh("block-followup-without-bd-id.sh"), "block-followup-without-bd-id.sh"))
print(add_to_matched("PostToolUse", "Bash", sh("kbt-lifecycle.sh"), "kbt-lifecycle.sh"))
print(add_to_unmatched("Stop", sh("block-stop-without-kb-watcher.sh"), "block-stop-without-kb-watcher.sh"))
print(add_to_unmatched("Stop", py("bridge-owed-reply-stop.py"), "bridge-owed-reply-stop.py"))

json.loads(json.dumps(pl))  # validate
json.dump(pl, open(PLUGIN, "w"), indent=2)
print(f"  -> wrote {PLUGIN}")

# ---- 2. Deregister from settings.json (remove by basename, prune empty) ----
REMOVE = ["bridge-inject.sh", "bridge-recv-prompt.sh", "block-stop-without-bridge-watcher.sh",
          "bridge-unread-stop.sh", "bridge-owed-reply-stop.py", "bridge-resume.sh",
          "bd-lifecycle.sh", "session-followups.sh", "block-followup-without-bd-id.sh",
          "block-local-dolt-server.sh"]
shutil.copy(SETTINGS, SETTINGS + ".kb2os-swap.bak")
st = json.load(open(SETTINGS))
removed = []
for event, blocks in list(st.get("hooks", {}).items()):
    newblocks = []
    for b in blocks:
        kept = []
        for h in b.get("hooks", []):
            cmd = h.get("command", "")
            hit = next((r for r in REMOVE if r in cmd), None)
            if hit: removed.append(f"{event}{('['+b['matcher']+']') if 'matcher' in b else ''}: -{hit}")
            else: kept.append(h)
        if kept:
            b["hooks"] = kept; newblocks.append(b)
    if newblocks: st["hooks"][event] = newblocks
    else: del st["hooks"][event]

json.loads(json.dumps(st))  # validate
json.dump(st, open(SETTINGS, "w"), indent=2)
print("\nSETTINGS.json deregistrations:")
for r in removed: print("  " + r)
print(f"  -> wrote {SETTINGS} (backup: {SETTINGS}.kb2os-swap.bak)")
print(f"\nTOTAL removed from settings.json: {len(removed)}")
