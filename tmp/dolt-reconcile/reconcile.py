import subprocess, json, os, sys

PROJECTS = {
    "braidinfer": "/home/mcelrath/Projects/ai/braidinfer",
    "exterior_algebra": "/home/mcelrath/Projects/ai/exterior_algebra",
    "llamacpp": "/home/mcelrath/Projects/ai/llama.cpp",
}

def bd_all(host, cwd):
    env = dict(os.environ, BEADS_DOLT_SERVER_HOST=host)
    r = subprocess.run(["bd", "list", "--all", "--json"],
                       cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    out = r.stdout
    i = out.find("[")
    if i < 0:
        i = out.find("{")
    if i < 0:
        return None, (r.stdout + r.stderr)[:300]
    try:
        data = json.loads(out[i:])
    except Exception as e:
        return None, f"parse-fail: {e}"
    if isinstance(data, dict):
        data = data.get("issues", [])
    return data, None

def main():
    for db, cwd in PROJECTS.items():
        if not os.path.isdir(os.path.join(cwd, ".beads")):
            print(f"{db}: NO .beads at {cwd}")
            continue
        loc, lerr = bd_all("127.0.0.1", cwd)
        tar, terr = bd_all("tardis", cwd)
        if loc is None or tar is None:
            print(f"{db}: loc_err={lerr} tar_err={terr}")
            continue
        lids = {i["id"] for i in loc}
        tids = {i["id"] for i in tar}
        only_local = sorted(lids - tids)
        only_tardis = sorted(tids - lids)
        print(f"{db}: local={len(lids)} tardis={len(tids)} "
              f"local_only={len(only_local)} tardis_only={len(only_tardis)}")
        if only_local:
            print(f"  LOCAL-ONLY (would be lost): {only_local}")
        if only_tardis:
            print(f"  tardis-only (already safe): {only_tardis[:8]}{'...' if len(only_tardis)>8 else ''}")

main()
