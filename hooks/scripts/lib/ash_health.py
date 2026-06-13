"""Shared health gate for kb-infra servers that feed information to agents:
  - ash:8081      embedding server (kb semantic search + surfacing)
  - tardis:9510   local LLM (precompact summary, kb digest/rerank)

When either is down the corresponding surfacing goes BLIND and SILENT; agents
then forge ahead / re-derive. ash_down() returns True if EITHER is unreachable
and sets module global STOP_LINE naming which. Cached 60s in /tmp.
"""
import os, re, time, urllib.request

_CACHE = "/tmp/.kbinfra_health_cache"
_TTL = 60

def _base(url: str) -> str:
    m = re.match(r"(https?://[^/]+)", url)
    return (m.group(1) if m else url).rstrip("/") + "/"

# Endpoints derive from the configured service URLs; defaults preserve the
# original ash/tardis hosts.
_EMB = _base(os.environ.get("KB_EMBEDDING_URL", "http://ash:8081/embedding"))
_LLM = _base(os.environ.get("KB_LLM_URL", "http://tardis:9510/completion"))
_ENDPOINTS = {f"{_EMB.split('://',1)[-1].rstrip('/')} (embeddings)": _EMB,
              f"{_LLM.split('://',1)[-1].rstrip('/')} (LLM)": _LLM}

STOP_LINE = ""

def _probe():
    down = []
    for name, url in _ENDPOINTS.items():
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status != 200:
                    down.append(name)
        except Exception:
            down.append(name)
    return down

def ash_down() -> bool:
    global STOP_LINE
    down = None
    try:
        st = os.stat(_CACHE)
        if time.time() - st.st_mtime < _TTL:
            c = open(_CACHE).read().strip()
            down = [] if c == "UP" else c.split("|")
    except OSError:
        pass
    if down is None:
        down = _probe()
        try:
            open(_CACHE, "w").write("UP" if not down else "|".join(down))
        except OSError:
            pass
    if down:
        STOP_LINE = (
            f"[🛑 KB-INFRA DOWN ({', '.join(down)}) — the information this feeds agents "
            "(kb-search, structural-fact/codified surfacing, precompact summary, kb digest) "
            "is BLIND/SILENT. Do NOT dispatch compute/derivation or re-derive: prior-art "
            "retrieval is non-functional and you would forge ahead blind. STOP, tell the user "
            "'kb-infra down — holding', do mechanical-only work (commit/move/build) until it "
            "recovers. An empty surface now means BLIND, not 'nothing relevant'.]")
        return True
    STOP_LINE = ""
    return False
