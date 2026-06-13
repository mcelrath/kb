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

    return kb_search, kb_recent, finding_get, issues_list, issue_get
