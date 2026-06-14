import json, shutil

P = "/home/mcelrath/.claude/settings.json"
shutil.copy(P, P + ".bak3")

# kb-jij.1: retire the mid-work BRIDGE_WATCHER_DOWN nag (bridge-watcher-alive.sh on
# PreToolUse + UserPromptSubmit). The watcher is now needed only at Stop (idle),
# enforced by block-stop-without-bridge-watcher.sh, which stays.
TARGET = "bridge-watcher-alive.sh"

d = json.load(open(P))
removed = []
for event, groups in d.get("hooks", {}).items():
    for g in groups:
        kept = []
        for h in g.get("hooks", []):
            if TARGET in h.get("command", ""):
                removed.append(event)
            else:
                kept.append(h)
        g["hooks"] = kept

with open(P, "w") as f:
    json.dump(d, f, indent=2)
    f.write("\n")

json.load(open(P))  # validate
print("removed bridge-watcher-alive from:", removed)

# Survival check: the Stop-side enforcement + the injector must remain
survivors = [h.get("command", "") for ev, gs in json.load(open(P))["hooks"].items()
             for g in gs for h in g.get("hooks", [])]
for must in ["block-stop-without-bridge-watcher.sh", "bridge-watcher-check.sh",
             "bridge-recv-prompt.sh", "bridge-owed-reply-stop.py"]:
    print("  %s: %s" % (must, "PRESENT" if any(must in s for s in survivors) else "*** MISSING ***"))
print("  bridge-watcher-alive still present:", any(TARGET in s for s in survivors))
