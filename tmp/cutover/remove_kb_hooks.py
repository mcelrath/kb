import json, shutil, sys

P = "/home/mcelrath/.claude/settings.json"
shutil.copy(P, P + ".bak2")

# The 7 SHIPPED kb findings hooks now provided by the plugin. Remove ONLY these
# from the personal settings.json. EXCLUDED (stay personal): compose_time_check.py,
# symbol_surface.py, open_issues_surface.py, bd-lifecycle.sh, git-commit-check.sh,
# and all bridge/guard hooks.
TARGETS = [
    "kb-context.sh", "kb-prompt-surface.py", "kb-search-track.sh",
    "kb-error-extract.sh", "kb-flush-pending.sh", "kb-precompact.sh",
    "dedupe-kb-get.sh",
]

d = json.load(open(P))
removed = []
for event, groups in d.get("hooks", {}).items():
    for g in groups:
        kept = []
        for h in g.get("hooks", []):
            cmd = h.get("command", "")
            if any(t in cmd for t in TARGETS):
                removed.append((event, cmd))
            else:
                kept.append(h)
        g["hooks"] = kept

with open(P, "w") as f:
    json.dump(d, f, indent=2)
    f.write("\n")

# Validate + report
json.load(open(P))
print("removed %d kb-hook entries:" % len(removed))
for e, c in removed:
    print("  [%s] %s" % (e, c.split("/")[-1]))

# Survival check: critical hooks must remain
survivors = []
for event, groups in json.load(open(P)).get("hooks", {}).items():
    for g in groups:
        for h in g.get("hooks", []):
            survivors.append(h.get("command", ""))
must_survive = ["block-stop-without-bridge-watcher.sh", "bridge-owed-reply-stop.py",
                "guard-destructive-git.sh", "block-markdown-files.sh",
                "compose_time_check.py", "symbol_surface.py", "open_issues_surface.py"]
print("\nsurvival check:")
for m in must_survive:
    print("  %s: %s" % (m, "PRESENT" if any(m in s for s in survivors) else "*** MISSING ***"))
print("\nany of the 7 kb hooks still present:",
      [t for t in TARGETS if any(t in s for s in survivors)])
print("enabledPlugins.kb:", [k for k in d.get("enabledPlugins", {}) if k.startswith("kb@")])
