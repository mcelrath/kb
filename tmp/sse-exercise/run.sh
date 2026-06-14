#!/bin/bash
# Exercise the kb-server bridge SSE endpoints end-to-end. Self-terminating
# (timeout on the server, --max-time on the SSE curl) — no `kill` (deny-listed).
cd /home/mcelrath/Projects/ai/kb
PORT=8772
OUT=/tmp/claude/sse-out.txt
mkdir -p /tmp/claude
: > "$OUT"

timeout 30 .venv/bin/python kb.py serve --port "$PORT" >/tmp/claude/kbserve.log 2>&1 &

# Wait for the server to accept connections
up=0
for i in $(seq 1 30); do
  if curl -s -m 2 -o /dev/null "http://localhost:$PORT/bridge/agents"; then up=1; break; fi
  sleep 0.5
done
if [ "$up" != 1 ]; then
  echo "SERVER_DID_NOT_START"
  cat /tmp/claude/kbserve.log | tail -15
  exit 0
fi

echo "[1] GET /bridge/agents"
curl -s -m 5 "http://localhost:$PORT/bridge/agents" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); a=d.get('agents',d) if isinstance(d,dict) else d; print('   HTTP ok — agents:', len(a))"

echo "[2] GET /bridge/messages?recipient=claude-config-dev&limit=3"
curl -s -m 5 "http://localhost:$PORT/bridge/messages?recipient=claude-config-dev&limit=3" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('   HTTP ok — msgs:', len(d)); [print('   ', m.get('id'), '|', str(m.get('subject',''))[:48]) for m in d]"

echo "[3] SSE live-push: subscribe /bridge/watch?id=sse-exercise, then send a message to it"
curl -sN --max-time 12 "http://localhost:$PORT/bridge/watch?id=sse-exercise" > "$OUT" 2>&1 &
sleep 2
/home/mcelrath/.agent-bridge/bridge send sse-exercise "SSE live-push exercise" <<< "proactive SSE delivery test" >/dev/null 2>&1
echo "   message sent to sse-exercise; waiting for the stream to receive it..."
sleep 5

echo "[4] SSE stream captured:"
sed 's/^/   /' "$OUT" | head -25
echo "[5] verdict:"
if grep -q "SSE live-push exercise\|proactive SSE delivery" "$OUT"; then
  echo "   PUSH CONFIRMED — the message arrived over the SSE stream"
else
  echo "   PUSH NOT SEEN — stream output above (heartbeats only?)"
fi
