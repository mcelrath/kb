"""`kb search --federated` client (epic kb-907fc8 P3).

Search locally AND fan out the query to every enabled peer's /federation/search,
then merge + rank ALL results by RAW cosine similarity (1 - d^2/2 — comparable only
under the company-standardized embedding model, epic P0). Offline/slow peers are
tolerated (per-peer timeout; their failure is reported, not fatal).
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import kb.cli.output as output

_PEER_TIMEOUT = 6


def _query_peer(peer: dict, query: str, k: int) -> dict:
    url = peer["url"].rstrip("/") + "/federation/search"
    body = json.dumps({"query": query, "k": k}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    if peer.get("token"):
        req.add_header("Authorization", f"Bearer {peer['token']}")
    with urllib.request.urlopen(req, timeout=_PEER_TIMEOUT) as r:
        return json.loads(r.read())


def run_federated_search(kb, args) -> None:
    query = args.query
    k = args.limit
    local_model = getattr(getattr(kb, "_embedding", None), "embedding_model", None)

    # Local results first (owner = '(local)').
    rows: list[dict] = []
    for r in kb.search(query, limit=k):
        rows.append({
            "id": r.get("id"), "type": r.get("type"),
            "summary": r.get("summary") or (r.get("content") or "").split("\n")[0][:100],
            "project": r.get("project"), "similarity": r.get("similarity"),
            "owner": "(local)", "model_id": local_model,
        })

    peers = kb._peers.list(enabled_only=True)
    errors: list[tuple[str, str]] = []
    if peers:
        with ThreadPoolExecutor(max_workers=min(8, len(peers))) as ex:
            futs = {ex.submit(_query_peer, p, query, k): p for p in peers}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    resp = fut.result()
                    rows.extend(resp.get("results", []))
                except Exception as e:  # noqa: BLE001 — offline peer is non-fatal
                    errors.append((p["url"], str(e)[:60]))

    # Merge + rank on RAW cosine (drop rows without a comparable similarity).
    rows = [r for r in rows if isinstance(r.get("similarity"), (int, float))]
    rows.sort(key=lambda r: r["similarity"], reverse=True)
    rows = rows[:k]

    if getattr(args, "json", False):
        print(json.dumps({"results": rows, "errors": errors}, indent=2, default=str))
        return

    if not rows:
        print("No federated results.")
    for r in rows:
        owner = r.get("owner") or "?"
        sim = r.get("similarity")
        sim_s = f"({sim:.2f})" if isinstance(sim, (int, float)) else ""
        # Flag results from a peer on a DIFFERENT embedding model (scores not comparable).
        warn = ""
        if r.get("model_id") and local_model and r["model_id"] != local_model:
            warn = output.c(f" ⟨model {r['model_id']}≠{local_model}⟩", "yellow")
        text = r.get("summary") or ""
        print(output.fit_line(
            f"  {output.c(f'[{owner}]', 'cyan')} {output.c(str(r.get('id') or ''), 'dim')} "
            f"{output.c(sim_s, output.sim_color(sim) if isinstance(sim,(int,float)) else None)}{warn}  {text}"))
    for url, err in errors:
        print(output.c(f"  (peer unreachable: {url} — {err})", "dim"))
