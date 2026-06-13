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

from starlette.requests import Request
from starlette.responses import JSONResponse


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
        """GET /kb/search?q=<query>&limit=N

        Returns a JSON array of findings from hybrid search.
        Default limit=20, max 500.
        """
        query = request.query_params.get("q", "").strip()
        if not query:
            return JSONResponse({"error": "?q=<query> required"}, status_code=400)
        try:
            limit = int(request.query_params.get("limit", "20"))
        except ValueError:
            limit = 20
        limit = max(1, min(limit, 500))
        results = kb.search(query, limit=limit)
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
        """GET /moim?query=<text>&recipient=<id>&since=<cursor>&limit=N

        Returns plain-text context for injection into agent MOIM.
        Combines unread bridge messages and kb findings.
        Designed as a ContextProvider target for goose.
        """
        from starlette.responses import PlainTextResponse
        from .bridge import _parse_bridge_messages

        recipient = request.query_params.get("recipient", "goose").strip()
        query = request.query_params.get("query", "").strip()
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

        msgs = _parse_bridge_messages(recipient, limit=50, last_event_id=since)
        if msgs:
            lines = []
            for m in msgs:
                reply = f" (reply to #{m['reply_to']})" if m.get("reply_to") else ""
                lines.append(
                    f"[bridge #{m['id']}{reply}] from {m['sender']} at {m.get('ts','')}: "
                    f"{m['subject']}\n{m['body']}"
                )
            parts.append("Unread peer messages via agent-bridge:\n" + "\n\n".join(lines))

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

        if not parts:
            return PlainTextResponse("")
        return PlainTextResponse("\n\n".join(parts))

    return kb_search, kb_recent, finding_get, issues_list, issue_get, moim
