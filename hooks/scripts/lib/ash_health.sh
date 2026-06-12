# Shared health gate for kb-infra servers feeding agents (cached 60s):
#   ash:8081 (embeddings), tardis:9510 (LLM). Sets ASH_STOP_LINE naming which.
# Usage: source; `if ash_down; then echo "$ASH_STOP_LINE" >&2; fi`
_KBINFRA_CACHE=/tmp/.kbinfra_health_cache_sh
ash_down() {
  local down=""
  if [ -f "$_KBINFRA_CACHE" ] && [ $(( $(date +%s) - $(stat -c %Y "$_KBINFRA_CACHE" 2>/dev/null || echo 0) )) -lt 60 ]; then
    down="$(cat "$_KBINFRA_CACHE")"
  else
    # Derive the base (scheme://host:port) of each service from its configured
    # URL; defaults preserve the original ash/tardis endpoints.
    local _emb _llm _emb_base _llm_base
    _emb="${KB_EMBEDDING_URL:-http://ash:8081/embedding}"
    _llm="${KB_LLM_URL:-http://tardis:9510/completion}"
    _emb_base=$(printf '%s' "$_emb" | sed -E 's|(https?://[^/]+).*|\1|')
    _llm_base=$(printf '%s' "$_llm" | sed -E 's|(https?://[^/]+).*|\1|')
    # Probe /health, NOT / : llama.cpp returns 404 for / (the old probe
    # false-fired "down" every time) and 200 for /health once the model is
    # loaded. /health stays 200 during transient slot-busy 503s on /embedding,
    # so we announce DOWN only when the server is genuinely unreachable/loading
    # (retrieval still works — slot-wait or FTS fallback — when /health is 200).
    curl -s -m2 -o /dev/null -w '%{http_code}' "$_emb_base/health" 2>/dev/null | grep -q 200 || down="${_emb_base#*://}(embeddings) "
    curl -s -m2 -o /dev/null -w '%{http_code}' "$_llm_base/health" 2>/dev/null | grep -q 200 || down="${down}${_llm_base#*://}(LLM)"
    echo "${down:-UP}" > "$_KBINFRA_CACHE"
  fi
  [ "$down" = "UP" ] && return 1
  [ -z "$down" ] && return 1
  ASH_STOP_LINE="🛑 KB-INFRA DOWN ($down) — kb-search + surfacing + summaries are BLIND/SILENT. STOP retrieval-dependent compute/derivation; an empty kb-search now means BLIND not 'nothing found'. Tell the user 'kb-infra down — holding'; mechanical-only until recovered."
  return 0
}
