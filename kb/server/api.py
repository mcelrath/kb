"""KB read API endpoints for the opencode UI panels.

GET /kb/search?q=<query>&limit=N    -> JSON array of findings (hybrid search)
GET /kb/recent?limit=N              -> JSON array of recent findings
GET /issues?project=&status=&parent= -> JSON array of issues
GET /issues/{id}                    -> single issue JSON

All handlers are returned by make_api_handlers(kb) which closes over the
KnowledgeBase instance, mirroring the make_web_handlers / make_live_handlers
factory pattern used in routes.py and live.py.
"""

import json
import os
import re
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse

# Per-recipient /moim delivery cursor. A STATELESS poller (goose's fixed-URL
# ContextProvider can't carry ?since=) relies on the SERVER to remember what it
# has already delivered, so it gets only-NEW bridge messages each call and
# NOTHING on first contact (start at tail) — the same lesson the SSE endpoint
# learned (bridge.py fresh-subscriber tail). Without it /moim re-dumps the last
# 50 messages every turn (the goose first-run "whole bridge history" bug).
_MOIM_CURSOR_PATH = Path.home() / ".cache" / "kb" / "moim-cursors.json"


def _moim_cursor_load(recipient: str) -> int | None:
    try:
        data = json.loads(_MOIM_CURSOR_PATH.read_text())
        return int(data[recipient])
    except Exception:
        return None


def _moim_cursor_save(recipient: str, last_id: int) -> None:
    try:
        _MOIM_CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(_MOIM_CURSOR_PATH.read_text())
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        data[recipient] = int(last_id)
        tmp = _MOIM_CURSOR_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, _MOIM_CURSOR_PATH)
    except Exception:
        pass


def _bridge_tail_id() -> int:
    """Current max bridge message id (mirrors the SSE fresh-subscriber tail)."""
    from .bridge import BRIDGE_MESSAGES_PATH
    maxid = 0
    try:
        with open(BRIDGE_MESSAGES_PATH) as f:
            for line in f:
                try:
                    mid = json.loads(line).get("id")
                    if mid is not None and int(mid) > maxid:
                        maxid = int(mid)
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return maxid


class _StrJSONResponse(JSONResponse):
    """JSONResponse that serializes with default=str so datetimes/Paths don't
    500 (mirrors the bridge endpoints' json.dumps(..., default=str); Starlette's
    JSONResponse.render has no default= hook)."""

    def render(self, content) -> bytes:
        return json.dumps(content, default=str).encode("utf-8")


def _json(data, status_code: int = 200) -> JSONResponse:
    return _StrJSONResponse(data, status_code=status_code)


def make_api_handlers(kb):
    """Return (kb_search, kb_recent, issues_list, issue_get) bound to kb."""

    async def kb_search(request: Request) -> JSONResponse:
        """GET /kb/search?q=<query>&limit=N&project=<tag>

        Returns a JSON array of findings from hybrid search.
        Default limit=20, max 500. Optional ?project=<tag> scopes results to one
        project (e.g. a per-agent ContextProvider scoping its injection surface).
        """
        query = request.query_params.get("q", "").strip()
        if not query:
            return JSONResponse({"error": "?q=<query> required"}, status_code=400)
        try:
            limit = int(request.query_params.get("limit", "20"))
        except ValueError:
            limit = 20
        limit = max(1, min(limit, 500))
        kw: dict = {"limit": limit}
        project = request.query_params.get("project", "").strip()
        if project:
            kw["project"] = project
        results = kb.search(query, **kw)
        return _json(results)

    async def kb_recent(request: Request) -> JSONResponse:
        """GET /kb/recent?limit=N

        Returns a JSON array of recent findings, newest first.
        Default limit=20, max 500.
        """
        try:
            limit = int(request.query_params.get("limit", "20"))
        except ValueError:
            limit = 20
        limit = max(1, min(limit, 500))
        results = kb.list_findings(limit=limit)
        return _json(results)

    async def finding_get(request: Request) -> JSONResponse:
        """GET /kb/finding/{id}

        Returns the FULL finding (incl. evidence) — the list/search endpoints
        omit evidence (it can be large); fetch the full record by id here.
        """
        finding_id = request.path_params["id"]
        result = kb.get(finding_id)
        if result is None:
            return JSONResponse({"error": f"Finding not found: {finding_id}"}, status_code=404)
        return _json(result)

    async def issues_list(request: Request) -> JSONResponse:
        """GET /issues?project=&status=&parent=&limit=N

        Returns a JSON array of issues. All query params are optional.
        Default limit=100, max 500.
        """
        project = request.query_params.get("project", "").strip() or None
        status = request.query_params.get("status", "").strip() or None
        parent = request.query_params.get("parent", "").strip() or None
        try:
            limit = int(request.query_params.get("limit", "100"))
        except ValueError:
            limit = 100
        limit = max(1, min(limit, 500))
        results = kb._issues.list(
            project=project,
            status=status,
            parent_id=parent,
            limit=limit,
        )
        return _json(results)

    async def issue_get(request: Request) -> JSONResponse:
        """GET /issues/{id}

        Returns the full issue JSON including comments and deps.
        """
        issue_id = request.path_params["id"]
        result = kb._issues.get(issue_id)
        if result is None:
            return JSONResponse({"error": f"Issue not found: {issue_id}"}, status_code=404)
        return _json(result)

    async def moim(request: Request):
        """GET /moim?query=<text>&recipient=<id>&session_id=<sid>&since=<cursor>&limit=N

        Returns plain-text context for injection into agent MOIM: unread bridge
        messages + relevant kb findings. The SOLE bridge-delivery path for goose
        (goose's native BridgeReader was retired in favor of this — kb-3fj).
        Designed as a ContextProvider target for goose.
        """
        from starlette.responses import PlainTextResponse
        from .bridge import _parse_bridge_messages

        recipient = request.query_params.get("recipient", "goose").strip()
        session_id = request.query_params.get("session_id", "").strip()
        query = request.query_params.get("query", "").strip()
        # bridge_only: suppress the (intentionally un-cursored) findings block so the
        # response is EMPTY when no new bridge message exists. A per-turn ContextProvider
        # wants findings re-surfaced every turn; a background asyncRewake poller (goose's
        # 5s loop) needs empty-when-idle or it wakes spuriously and never reaches readline.
        bridge_only = request.query_params.get("bridge_only", "").strip().lower() in ("1", "true", "yes")

        # Refresh the recipient's bridge LIVENESS mtime so `bridge agents` reports
        # it ONLINE while it actively pulls /moim every turn (goose's path). Without
        # this, goose reads 'offline:stale' despite fetching each turn — same root
        # cause the Claude bridge-inject hook touch fixes for Claude sessions.
        # recipient is a query param → charset-guard to block path traversal.
        if re.fullmatch(r"[A-Za-z0-9_-]+", recipient):
            try:
                _cur = os.path.expanduser(f"~/.agent-bridge/{recipient}.cursor")
                if os.path.exists(_cur):
                    os.utime(_cur, None)
                else:
                    open(_cur, "a").close()
            except OSError:
                pass
        raw_since = request.query_params.get("since", "").strip()
        try:
            since: int | None = int(raw_since) if raw_since else None
        except ValueError:
            since = None
        try:
            limit = int(request.query_params.get("limit", "5"))
        except ValueError:
            limit = 5
        limit = max(1, min(limit, 50))

        parts = []

        # Cursor key is SESSION-scoped when the caller passes ?session_id= (goose's
        # ContextProvider does), so two sessions of the same recipient don't consume
        # each other's messages; falls back to recipient-only otherwise. An explicit
        # ?since= wins; else the stored cursor; else the TAIL on first contact so a
        # stateless poller never gets the backlog dumped. Advance to the newest
        # delivered id so each message injects exactly once.
        cursor_key = f"{recipient}#{session_id}" if session_id else recipient
        if since is not None:
            eff_since: int | None = since
        else:
            stored = _moim_cursor_load(cursor_key)
            eff_since = stored if stored is not None else _bridge_tail_id()
        msgs = _parse_bridge_messages(recipient, limit=50, last_event_id=eff_since)
        new_cursor = eff_since or 0
        for _m in msgs:
            try:
                new_cursor = max(new_cursor, int(_m["id"]))
            except (TypeError, ValueError, KeyError):
                pass
        # NOTE: the cursor is advanced AFTER the response body is built (below),
        # not here — so a formatting error can't advance past undelivered messages.
        if msgs:
            lines = []
            for m in msgs:
                reply = f" (reply to #{m['reply_to']})" if m.get("reply_to") else ""
                lines.append(
                    f"[bridge #{m['id']}{reply}] from {m['sender']} at {m.get('ts','')}: "
                    f"{m['subject']}\n{m['body']}"
                )
            parts.append("Unread peer messages via agent-bridge:\n" + "\n\n".join(lines))

        findings = []
        if not bridge_only:
            if query:
                findings = kb.search(query, limit=limit)
            else:
                findings = kb.list_findings(limit=limit)
        if findings:
            lines = []
            for f in findings:
                proj = f" [{f['project']}]" if f.get("project") else ""
                lines.append(f"[kb {f['id']}{proj}] {f.get('summary', '')}\n{f.get('content', '')}")
            parts.append("Relevant knowledge base findings:\n" + "\n\n".join(lines))

        # Advance the cursor only now that the bridge messages are captured in the
        # response body — if formatting above had thrown, the cursor stays put and
        # the messages redeliver on the next poll (no silent loss).
        _moim_cursor_save(cursor_key, new_cursor)
        if not parts:
            return PlainTextResponse("")
        return PlainTextResponse("\n\n".join(parts))

    return kb_search, kb_recent, finding_get, issues_list, issue_get, moim
