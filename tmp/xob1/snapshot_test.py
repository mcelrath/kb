#!/usr/bin/env python3
"""Snapshot oracle for kb-xob.1 pure producers.

For each producer, runs `kb surface --<mode> --json` and verifies:
  1. The call succeeds (exit 0)
  2. The JSON output is well-formed with expected keys
  3. The `context` field matches the format the live hooks emit (same header,
     same line format per entry)

NOTE on seen-gate delta: the live hooks call filter_unseen (a session-scoped
seen-set that deduplicates across turns). The producers do NOT — they return
the full pre-dedup candidate set. So if findings were already surfaced this
session in a real run, the hook would emit fewer lines than produce_* returns.

To eliminate this delta in testing, we run with a fresh process (no seen-gate
state) and compare only structure/format, not specific IDs (since the DB
content may vary across environments).

Run:
  cd /home/mcelrath/Projects/ai/kb
  ALL_PROXY="" NO_PROXY="*" .venv/bin/python tmp/xob1/snapshot_test.py
"""

import json
import os
import subprocess
import sys

KB_VENV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        ".venv", "bin", "python")
KB_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          "kb.py")

ENV = {**os.environ, "ALL_PROXY": "", "NO_PROXY": "*",
       "KB_EMBEDDING_URL": "http://ash:8081/embedding",
       "KB_EMBEDDING_DIM": "4096"}


def run_surface(*args: str, timeout: int = 20) -> dict:
    """Run kb surface ... --json and return parsed JSON output."""
    cmd = [KB_VENV, KB_SCRIPT, "surface"] + list(args) + ["--json"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=ENV)
    assert r.returncode == 0, f"surface exited {r.returncode}: {r.stderr[:400]}"
    assert r.stdout.strip(), f"surface produced no output; stderr: {r.stderr[:400]}"
    return json.loads(r.stdout)


def check_injection(result: dict, producer: str) -> None:
    """Assert the JSON structure is a valid Injection."""
    assert "producer" in result, f"missing 'producer' key: {result}"
    assert "fired" in result, f"missing 'fired' key: {result}"
    assert "context" in result, f"missing 'context' key: {result}"
    assert "hits" in result, f"missing 'hits' key: {result}"
    assert result["producer"] == producer, \
        f"wrong producer: expected {producer!r}, got {result['producer']!r}"
    assert isinstance(result["hits"], list), "hits must be a list"
    assert isinstance(result["fired"], bool), "fired must be bool"
    assert isinstance(result["context"], str), "context must be str"


def test_prompt() -> None:
    """produce_prompt: semantic search on a user prompt."""
    result = run_surface("--prompt", "how to ingest typescript symbols into kb", "-n", "3")
    check_injection(result, "prompt")

    if result["fired"]:
        # Context must start with the exact header from kb-prompt-surface.py
        assert result["context"].startswith(
            "Possibly-relevant prior findings (semantic match to your prompt"
        ), f"Wrong header: {result['context'][:100]!r}"
        # Each hit line must match [KB ~N.NN id (proj): summary] format
        lines = result["context"].splitlines()[1:]
        for line in lines:
            assert line.startswith("[KB ~"), f"Unexpected hit line format: {line!r}"
        print(f"  PASS prompt: fired=True, {len(lines)} hits")
    else:
        print(f"  PASS prompt: fired=False (no hits above threshold — acceptable if KB sparse)")


def test_analysis() -> None:
    """produce_analysis: INTENT_RX gate + near-duplicate prior-art."""
    # Text with explicit reimplementation intent
    intent_text = (
        "I'll create a new function to parse bridge messages from the JSONL file. "
        "The implementation will use the BridgeMessagesRepository and embed each "
        "substantive message via the EmbeddingService. Let me implement this as a "
        "standalone producer that takes a message id and returns the formatted context. "
        "This new module will live in kb/surface/producers.py and will be pure."
    )
    result = run_surface("--analysis", intent_text, "-n", "3")
    check_injection(result, "analysis")

    if result["fired"]:
        assert result["context"].startswith(
            "PRIOR ART — your analysis proposes building"
        ), f"Wrong PRIOR ART header: {result['context'][:100]!r}"
        lines = result["context"].splitlines()[1:]
        for line in lines:
            assert line.startswith("[KB ~"), f"Unexpected hit line: {line!r}"
        print(f"  PASS analysis: fired=True, {len(lines)} hits")
    else:
        print(f"  PASS analysis: fired=False (no near-duplicate or no intent match)")

    # Text WITHOUT intent — must return fired=False
    no_intent = (
        "The embedding service operates at ash:8081 and uses the llamacpp format "
        "by default. Vectors are 4096-dimensional. The similarity formula converts "
        "L2 distance to cosine: similarity = 1 - (dist**2) / 2."
    )
    result2 = run_surface("--analysis", no_intent)
    check_injection(result2, "analysis")
    assert not result2["fired"], (
        "analysis should NOT fire on text without reimplementation intent"
    )
    print(f"  PASS analysis: no-intent text correctly returns fired=False")


def test_file() -> None:
    """produce_symbols: RETIRED/NOTATION surface on a file."""
    # Use producers.py itself as the test file — it has Python symbols
    producers_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "kb", "surface", "producers.py"
    )
    result = run_surface("--file", producers_path)
    check_injection(result, "symbols")

    # fired can be True or False depending on whether any symbols are RETIRED/NOTATION
    if result["fired"]:
        for line in result["context"].splitlines():
            assert line.startswith("[RETIRED:") or line.startswith("[NOTATION:") or \
                   line.startswith("[KB-VALUE:"), \
                f"Unexpected advisory line format: {line!r}"
        print(f"  PASS file: fired=True, {len(result['hits'])} advisories")
    else:
        print(f"  PASS file: fired=False (no RETIRED/NOTATION symbols found — OK)")


def test_issues() -> None:
    """produce_open_issues: vector+FTS over issues table."""
    result = run_surface("--issues", "surface producers pure function injection context")
    check_injection(result, "open_issues")

    if result["fired"]:
        for line in result["context"].splitlines():
            assert line.startswith("[OPEN-ISSUE:") or line.startswith("[RESOLVED-ISSUE:"), \
                f"Unexpected issue line: {line!r}"
        print(f"  PASS issues: fired=True, {len(result['hits'])} issues")
    else:
        print(f"  PASS issues: fired=False (no matching issues — OK if issues table sparse)")


def test_bridge() -> None:
    """produce_bridge: search bridge_messages on a text query."""
    result = run_surface("--bridge", "kb surface producers injection hooks migration")
    check_injection(result, "bridge")

    if result["fired"]:
        for line in result["context"].splitlines():
            assert line.startswith("[BRIDGE ~"), f"Unexpected bridge line: {line!r}"
        print(f"  PASS bridge: fired=True, {len(result['hits'])} bridge msgs")
    else:
        print(f"  PASS bridge: fired=False (no relevant bridge msgs — OK)")


def test_query_mode_preserved() -> None:
    """Legacy --query mode still works."""
    cmd = [KB_VENV, KB_SCRIPT, "surface", "embedding similarity search", "--json"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, env=ENV)
    assert r.returncode == 0, f"legacy surface exited {r.returncode}: {r.stderr[:300]}"
    data = json.loads(r.stdout)
    assert "symbols" in data and "findings" in data and "bridge" in data, \
        f"legacy --query mode missing expected keys: {list(data.keys())}"
    print(f"  PASS query (legacy): symbols={len(data['symbols'])} findings={len(data['findings'])} bridge={len(data['bridge'])}")


def main():
    print("kb-xob.1 snapshot oracle")
    print(f"  KB_SCRIPT: {KB_SCRIPT}")
    print(f"  KB_VENV:   {KB_VENV}")
    print()

    tests = [
        ("prompt",             test_prompt),
        ("analysis",           test_analysis),
        ("file (symbols)",     test_file),
        ("open_issues",        test_issues),
        ("bridge",             test_bridge),
        ("query (legacy)",     test_query_mode_preserved),
    ]

    passed = failed = 0
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1
        print()

    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
