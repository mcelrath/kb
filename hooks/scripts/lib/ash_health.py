"""Shared health gate for kb-infra servers that feed information to agents:
  - ash:8081      embedding server (kb semantic search + surfacing)
  - tardis:9510   local LLM (precompact summary, kb digest/rerank)

These fail INDEPENDENTLY and have DIFFERENT consequences:
  - embedding down  -> semantic search + surfacing go BLIND/SILENT. An empty
                       surface means "blind", not "nothing relevant" -> agents
                       must STOP compute/derivation. embedding_down() sets the
                       module global STOP_LINE accordingly.
  - LLM down        -> only summaries/digest/rerank degrade; retrieval is FINE.
                       This must NOT trigger a compute STOP. llm_down() is
                       advisory and never sets STOP_LINE.

Probed independently, cached 60s in /tmp. (The earlier combined ash_down() made
an LLM-only outage falsely STOP all compute even though search still worked.)
"""
import os, re, time, urllib.request, urllib.error

_CACHE = "/tmp/.kbinfra_health_cache"
_TTL = 60

def _base(url: str) -> str:
    m = re.match(r"(https?://[^/]+)", url)
    return (m.group(1) if m else url).rstrip("/") + "/"

# Endpoints derive from the configured service URLs; defaults preserve the
# original ash/tardis hosts.
_EMB = _base(os.environ.get("KB_EMBEDDING_URL", "http://ash:8081/embedding"))
_LLM = _base(os.environ.get("KB_LLM_URL", "http://tardis:9510/completion"))

STOP_LINE = ""


def _probe_one(url: str) -> bool:
    """True if the endpoint is DOWN. 'Down' means UNREACHABLE (connection refused /
    timeout / DNS) — NOT a non-200 status. An HTTP error response (e.g. 404 on the
    root path, which llama-server and the router both return) proves the server is
    up and answering, so it counts as UP. Probing for a specific 200 path would
    false-positive 'down' on healthy servers that don't serve that path."""
    try:
        urllib.request.urlopen(url, timeout=2)
        return False
    except urllib.error.HTTPError:
        return False  # got an HTTP response → server is reachable → UP
    except Exception:
        return True   # connection refused / timeout / DNS failure → DOWN


def _status() -> dict[str, bool]:
    """Return {'emb': down?, 'llm': down?}, cached 60s as 'emb=0,llm=1'."""
    try:
        st = os.stat(_CACHE)
        if time.time() - st.st_mtime < _TTL:
            c = dict(kv.split("=") for kv in open(_CACHE).read().strip().split(","))
            return {"emb": c.get("emb") == "1", "llm": c.get("llm") == "1"}
    except Exception:
        pass
    s = {"emb": _probe_one(_EMB), "llm": _probe_one(_LLM)}
    try:
        open(_CACHE, "w").write(f"emb={int(s['emb'])},llm={int(s['llm'])}")
    except OSError:
        pass
    return s


def embedding_down() -> bool:
    """True if the embedding server is unreachable -> semantic retrieval is BLIND.
    Sets STOP_LINE so callers can inject a hard compute-STOP advisory."""
    global STOP_LINE
    if _status()["emb"]:
        STOP_LINE = (
            f"[🛑 KB-EMBEDDING DOWN ({_EMB}) — semantic search + structural-fact/"
            "codified surfacing are BLIND/SILENT. Do NOT dispatch compute/derivation "
            "or re-derive: prior-art retrieval is non-functional and you would forge "
            "ahead blind. STOP, tell the user 'kb embedding down — holding', do "
            "mechanical-only work (commit/move/build) until it recovers. An empty "
            "surface now means BLIND, not 'nothing relevant'.]")
        return True
    STOP_LINE = ""
    return False


def llm_down() -> bool:
    """True if the local LLM is unreachable. Advisory only: summaries/digest/rerank
    degrade, but semantic retrieval is unaffected — never a compute-STOP."""
    return _status()["llm"]
