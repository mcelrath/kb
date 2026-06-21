"""Bridge endpoints: /bridge/messages, /bridge/agents, /bridge/watch (SSE).

All three handlers are extracted verbatim from kb.py:1849-1975.
_parse_bridge_messages and the path constants are redefined here (not imported
from kb.py) so this module has no dependency on the top-level CLI script — the
copies are byte-identical to kb.py:66-67,81-112.
"""

import asyncio
import json
import os
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

# Module-level bridge paths (mirrors kb.py:66-67)
BRIDGE_MESSAGES_PATH = Path.home() / ".agent-bridge" / "messages.jsonl"
BRIDGE_AGENTS_PATH = Path.home() / ".agent-bridge" / "agents.json"
BRIDGE_BIN = Path.home() / ".agent-bridge" / "bridge"


async def bridge_send(request: Request) -> JSONResponse:
    """POST /bridge/send — send a bridge message AS the requesting agent (kb-jij.6).

    JSON body: {"from": <sender-id>, "to": <id|comma-list|[ids]>, "subject": str,
                "body": str, "reply_to"?: int, "needs_reply"?: bool,
                "supersedes"?: int, "verified_by"?: str, "unverified"?: str}

    Delegates to `~/.agent-bridge/bridge send` (AGENT_ID=<from>) so the jsonl
    format, id assignment, and cursor stay byte-identical to the CLI sender —
    the kb-server is the API; the proven binary does the write.
    """
    import subprocess
    from starlette.concurrency import run_in_threadpool

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    sender = (data.get("from") or data.get("sender") or "").strip()
    to = data.get("to")
    subject = data.get("subject", "")
    body = data.get("body", "")
    if not sender or not to or not body:
        return JSONResponse({"error": "from, to, and body are required"}, status_code=400)
    to_arg = to if isinstance(to, str) else ",".join(str(t) for t in to)
    if not BRIDGE_BIN.exists():
        return JSONResponse({"error": "bridge binary not found"}, status_code=500)

    argv = [str(BRIDGE_BIN), "send", to_arg, str(subject)]
    if data.get("reply_to"):
        argv += ["--reply", str(data["reply_to"])]
    if data.get("supersedes"):
        # Retract/obsolete an earlier message (e.g. clear your own outbound
        # needs-reply). The read-side hooks already honor `supersedes`; this
        # stops /bridge/send from silently dropping it on write (kb-2os bug,
        # reported by mes-researcher #5695).
        argv += ["--supersedes", str(data["supersedes"])]
    if data.get("needs_reply"):
        argv += ["--needs-reply"]
    if data.get("verified_by"):
        argv += ["--verified-by", str(data["verified_by"])]
    if data.get("unverified"):
        argv += ["--unverified", str(data["unverified"])]

    env = dict(os.environ)
    env["AGENT_ID"] = sender
    try:
        proc = await run_in_threadpool(
            lambda: subprocess.run(
                argv, input=body, capture_output=True, text=True, timeout=15, env=env
            )
        )
    except Exception as e:
        return JSONResponse({"error": f"send failed: {e}"}, status_code=500)
    if proc.returncode != 0:
        return JSONResponse(
            {"error": proc.stderr.strip() or "send failed", "stdout": proc.stdout.strip()},
            status_code=500,
        )
    sent_id = None
    for tok in proc.stdout.split():
        if tok.startswith("id="):
            try:
                sent_id = int(tok[3:])
            except ValueError:
                pass
            break
    return JSONResponse({"ok": True, "id": sent_id, "stdout": proc.stdout.strip()})


def _msg_to_list(msg: dict) -> list:
    to = msg.get("to", [])
    if isinstance(to, str):
        to = [t.strip() for t in to.split(",")]
    if not isinstance(to, list):
        to = [str(to)]
    return to


def _bridge_msg_for_recipient(msg: dict, recipient: str) -> bool:
    """Return True if msg is addressed to recipient or is a broadcast to 'all'."""
    to = _msg_to_list(msg)
    return recipient in to or "all" in to


def _bridge_msg_directed(msg: dict, recipient: str) -> bool:
    """True only if msg is EXPLICITLY addressed to recipient (NOT an 'all' broadcast).

    The directed/broadcast distinction is what lets WAKE fire only on messages
    addressed to you and keeps broadcast volume from evicting directed messages.
    """
    return recipient in _msg_to_list(msg)


def _parse_bridge_messages(
    recipient: str | None, limit: int, last_event_id: int | None = None,
    mode: str = "any",
) -> list[dict]:
    """Read messages.jsonl, filter by recipient, return newest-last.

    mode:
      'any'        — directed-to-recipient OR 'all' broadcast (default; recv/pull)
      'directed'   — EXPLICITLY addressed to recipient only (wake + no-evict path)
      'broadcast'  — 'all' broadcasts NOT also explicitly addressed to recipient
    If last_event_id is given, only return messages with numeric id > last_event_id.
    """
    if not BRIDGE_MESSAGES_PATH.exists():
        return []
    msgs = []
    try:
        with open(BRIDGE_MESSAGES_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if last_event_id is not None:
                    msg_id = msg.get("id")
                    if msg_id is not None:
                        try:
                            if int(msg_id) <= int(last_event_id):
                                continue
                        except (TypeError, ValueError):
                            pass
                if recipient is None:
                    keep = True
                elif mode == "directed":
                    keep = _bridge_msg_directed(msg, recipient)
                elif mode == "broadcast":
                    keep = (_bridge_msg_for_recipient(msg, recipient)
                            and not _bridge_msg_directed(msg, recipient))
                else:
                    keep = _bridge_msg_for_recipient(msg, recipient)
                if keep:
                    msgs.append(msg)
    except OSError:
        return []
    return msgs[-limit:] if limit > 0 else msgs


async def bridge_messages(request: Request) -> JSONResponse:
    """GET /bridge/messages?recipient=<id>&limit=N&since=<cursor>

    Returns a JSON array of bridge messages addressed to <recipient>
    (or 'all' broadcasts), newest-last. Default limit=50.
    Optional ?since=N returns only messages with id > N.
    """
    recipient = request.query_params.get("recipient", "").strip() or None
    try:
        limit = int(request.query_params.get("limit", "50"))
    except ValueError:
        limit = 50
    limit = max(1, min(limit, 500))
    raw_since = request.query_params.get("since", "").strip()
    try:
        since: int | None = int(raw_since) if raw_since else None
    except ValueError:
        since = None
    mode = request.query_params.get("mode", "any").strip() or "any"
    if mode not in ("any", "directed", "broadcast"):
        mode = "any"
    msgs = _parse_bridge_messages(recipient, limit, last_event_id=since, mode=mode)
    return JSONResponse(msgs)


async def bridge_agents(request: Request) -> JSONResponse:
    """GET /bridge/agents

    Returns the agent registry from ~/.agent-bridge/agents.json.
    Fields: id, role, cwd, description, session_id, joined_at.
    """
    if not BRIDGE_AGENTS_PATH.exists():
        return JSONResponse({"agents": []})
    try:
        data = json.loads(BRIDGE_AGENTS_PATH.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse(data)


async def bridge_watch(request: Request) -> StreamingResponse | JSONResponse:
    """GET /bridge/watch?id=<agent-id>[&since=N]

    SSE stream of bridge messages for the given agent id (plus 'all'
    broadcasts).  Honors the standard Last-Event-ID request header for
    reconnect/resume — only messages with numeric id > that value are sent.

    Resume precedence: Last-Event-ID header > ?since=N query param > TAIL.
    The ?since=N param exists for clients that CANNOT set Last-Event-ID on the
    INITIAL connect (e.g. browser EventSource only auto-sends it on reconnect):
    pass the last id you saw to replay missed backlog on the first connect.

    Frame format:
        id: <msg-id>\\ndata: <json>\\n\\n

    Heartbeat every ~10s:
        : ping\\n\\n
    """
    agent_id = request.query_params.get("id", "").strip()
    if not agent_id:
        return JSONResponse({"error": "?id=<agent-id> required"}, status_code=400)

    # Parse Last-Event-ID header for resume (SSE-standard reconnect).
    raw_lei = request.headers.get("last-event-id", "").strip()
    try:
        last_id: int | None = int(raw_lei) if raw_lei else None
    except ValueError:
        last_id = None

    # No header -> honor an explicit ?since=N replay floor (initial-connect backlog).
    if last_id is None:
        raw_since = request.query_params.get("since", "").strip()
        if raw_since:
            try:
                last_id = int(raw_since)
            except ValueError:
                last_id = None

    # Fresh subscriber (no Last-Event-ID, no ?since): start at the CURRENT TAIL —
    # deliver only NEW messages, never replay history. Otherwise a
    # freshly-launched SSE client is flooded with every past 'all'
    # broadcast on connect (caught while exercising kb-jij.4). UIs that
    # want backfill use GET /bridge/messages?limit=N separately.
    if last_id is None:
        _maxid = 0
        try:
            _mp = os.path.expanduser("~/.agent-bridge/messages.jsonl")
            with open(_mp) as _f:
                for _line in _f:
                    try:
                        _mid = json.loads(_line).get("id")
                        if _mid is not None and int(_mid) > _maxid:
                            _maxid = int(_mid)
                    except Exception:
                        pass
        except FileNotFoundError:
            pass
        last_id = _maxid

    async def event_generator():
        nonlocal last_id
        last_heartbeat = asyncio.get_event_loop().time()

        # On connect: replay any DIRECTED messages past last_id. Broadcasts NEVER
        # wake (they would flood the watcher with cross-project 'all' traffic and
        # let broadcast volume cursor-leap a directed message); they remain
        # pull-only via GET /bridge/messages. Wake fires only for messages
        # EXPLICITLY addressed to this agent.
        catchup = _parse_bridge_messages(agent_id, limit=200, last_event_id=last_id,
                                         mode="directed")
        for msg in catchup:
            msg_id = msg.get("id")
            data = json.dumps(msg, default=str)
            frame = f"id: {msg_id}\ndata: {data}\n\n"
            yield frame.encode()
            if msg_id is not None:
                try:
                    last_id = int(msg_id)
                except (TypeError, ValueError):
                    pass

        # Tail: poll for new messages every 0.75s
        while True:
            now = asyncio.get_event_loop().time()
            # Heartbeat every 10s
            if now - last_heartbeat >= 10.0:
                yield b": ping\n\n"
                last_heartbeat = now

            new_msgs = _parse_bridge_messages(agent_id, limit=50, last_event_id=last_id,
                                              mode="directed")
            for msg in new_msgs:
                msg_id = msg.get("id")
                data = json.dumps(msg, default=str)
                frame = f"id: {msg_id}\ndata: {data}\n\n"
                yield frame.encode()
                if msg_id is not None:
                    try:
                        last_id = int(msg_id)
                    except (TypeError, ValueError):
                        pass
                last_heartbeat = asyncio.get_event_loop().time()

            await asyncio.sleep(0.75)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
