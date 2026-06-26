"""Federated-kb server endpoint (epic kb-907fc8 P2a).

POST /federation/search — a peer asks THIS node to search its own findings (and
kbt tasks — P2c) and return top-k. Auth-gated (request_authorized; default-deny for
non-loopback without a token).

Embedding model: under the company-standardized model (epic P0), the requester
sends the query TEXT and this node embeds it locally — same model => comparable
vectors, so the raw-cosine `similarity` we return is directly mergeable across
peers (P3). We also advertise this node's owner + model_id/dim so the requester can
verify comparability and fall back to text re-embedding on mismatch. A precomputed
`query_vector` fast-path (P2b, into hybrid.py) is a future optimization — for now we
always embed the text, which is correct under standardized embeddings.
"""

from __future__ import annotations

import os
import socket

from starlette.responses import JSONResponse

from .auth import request_authorized

PROTOCOL_VERSION = 1


def _node_owner() -> str:
    """Identity of THIS node in federated results (KB_NODE_NAME env, else hostname)."""
    return os.environ.get("KB_NODE_NAME", "").strip() or socket.gethostname()


def _node_model(kb) -> tuple[str | None, int | None]:
    emb = getattr(kb, "_embedding", None)
    return (getattr(emb, "embedding_model", None) or None,
            getattr(emb, "embedding_dim", None))


def make_federation_handlers(kb):
    """Return the federated-search route handler bound to `kb`."""

    async def federated_search(request):
        if not request_authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)

        query = str(body.get("query") or "").strip()
        if not query:
            return JSONResponse({"error": "query (text) required"}, status_code=400)
        try:
            k = int(body.get("k", 10))
        except (TypeError, ValueError):
            k = 10
        k = max(1, min(k, 50))

        owner = _node_owner()
        model_id, dim = _node_model(kb)

        try:
            results = kb.search(query, limit=k)
        except Exception as e:  # noqa: BLE001 — never 500 the whole fan-out for one peer
            return JSONResponse(
                {"protocol_version": PROTOCOL_VERSION, "owner": owner,
                 "model_id": model_id, "dim": dim, "results": [], "error": f"search failed: {e}"},
                status_code=200,
            )

        out = []
        for r in results:
            out.append({
                "id": r.get("id"),
                "type": r.get("type"),
                "summary": r.get("summary") or (r.get("content") or "").split("\n")[0][:160],
                "project": r.get("project"),
                # raw cosine (1 - d^2/2) — comparable across peers ONLY under a shared model.
                "similarity": r.get("similarity"),
                "owner": owner,
                "model_id": model_id,
            })
        return JSONResponse({
            "protocol_version": PROTOCOL_VERSION,
            "owner": owner,
            "model_id": model_id,
            "dim": dim,
            "results": out,
        })

    return federated_search
