"""CLI handler for `kb peers` — manage the federated-kb peer registry (epic kb-907fc8 P1).

Subcommands: list (default) | add | remove | health.
"""

import json
import time
import urllib.request

import kb.cli.output as output


def _probe(url: str, token: str | None) -> tuple[bool, str]:
    """Reachability probe of a peer kb-server. Returns (ok, info)."""
    try:
        req = urllib.request.Request(url.rstrip("/") + "/bridge/agents")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=4) as r:
            d = json.loads(r.read())
            return True, f"{len(d.get('agents', []))} agents"
    except Exception as e:  # noqa: BLE001 — reachability check, any failure = offline
        return False, str(e)[:50]


def _age(ts: float | None) -> str:
    if not ts:
        return ""
    sec = time.time() - ts
    if sec < 90:
        return "seen now"
    if sec < 5400:
        return f"seen {int(sec // 60)}m ago"
    if sec < 172800:
        return f"seen {int(sec // 3600)}h ago"
    return f"seen {sec / 86400:.1f}d ago"


def run_peers(kb, args) -> None:
    repo = kb._peers
    sub = getattr(args, "peers_cmd", None)

    if sub == "add":
        r = repo.add(
            args.url, label=args.label, model_id=args.model_id, dim=args.dim,
            quant=args.quant, instruction_prefix=args.instruction_prefix, token=args.token,
        )
        print(f"{'Added' if r['is_new'] else 'Updated'} peer: {args.url}")
        return

    if sub == "remove":
        ok = repo.remove(args.url)
        print(f"Removed: {args.url}" if ok else f"Not found: {args.url}")
        return

    if sub == "health":
        peers = repo.list()
        if not peers:
            print("No peers registered.")
            return
        for p in peers:
            ok, info = _probe(p["url"], p.get("token"))
            if ok:
                repo.set_last_seen(p["url"], time.time())
            status = output.c("online ", "green") if ok else output.c("offline", "red")
            print(output.fit_line(f"  {status}  {p['url']}  ({p.get('label') or '?'})  {info}"))
        return

    # default: list
    peers = repo.list()
    if not peers:
        print("No peers registered. Add one:  kb peers add <url> --label NAME --model-id M --dim N")
        return
    for p in peers:
        dis = "" if p.get("enabled") else output.c(" [disabled]", "dim")
        model = f"{p.get('model_id') or '?'}/{p.get('dim') or '?'}d"
        seen = _age(p.get("last_seen"))
        seen_s = f"  {output.c(seen, 'dim')}" if seen else ""
        print(output.fit_line(f"  {p['url']}  {output.c('(' + (p.get('label') or '?') + ')', 'dim')}  {model}{dis}{seen_s}"))
