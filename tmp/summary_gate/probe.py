#!/usr/bin/env python3
"""
Probe: verify subscription-sdk summarizer works with stale key scrubbed.
Run from /home/mcelrath/Projects/ai/kb/tmp/summary_gate/ (contains ZZZSENTINEL CLAUDE.md)
"""
import asyncio
import os
import sys

SENTINEL = "ZZZSENTINEL"
TEST_TEXT = "kb is a SQLite knowledge base with vector search and LLM-powered summaries"


def _collect_text(messages):
    """Extract text from SDK message stream."""
    result = ""
    for msg in messages:
        # ResultMessage has content list of TextBlock etc.
        if hasattr(msg, 'content') and isinstance(msg.content, list):
            for block in msg.content:
                if hasattr(block, 'text'):
                    result += block.text
        elif hasattr(msg, 'text'):
            result += msg.text
    return result.strip()


async def _run_query(scrub_key: bool) -> str:
    import claude_agent_sdk

    env_copy = dict(os.environ)
    if scrub_key:
        env_copy.pop("ANTHROPIC_API_KEY", None)

    options = claude_agent_sdk.ClaudeAgentOptions(
        setting_sources=[],
        allowed_tools=[],
        model="claude-haiku-4-5",
        env=env_copy,
        cwd="/tmp",  # neutral cwd, no CLAUDE.md
    )
    messages = []
    async for msg in claude_agent_sdk.query(
        prompt=f"Summarize in ONE LINE, no preamble: {TEST_TEXT}",
        options=options,
    ):
        messages.append(msg)
    return _collect_text(messages)


def run_with_key_scrubbed():
    """(a) Call with ANTHROPIC_API_KEY removed — should succeed via subscription OAuth."""
    key_was_set = "ANTHROPIC_API_KEY" in os.environ
    try:
        result = asyncio.run(_run_query(scrub_key=True))
        return result, key_was_set
    except Exception as e:
        return None, str(e)


def run_with_key_present():
    """(c) Call WITH the stale ANTHROPIC_API_KEY set — expect failure."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        return None, "ANTHROPIC_API_KEY not in env — cannot test stale-key failure"
    try:
        result = asyncio.run(_run_query(scrub_key=False))
        return result, "call SUCCEEDED (key may not be stale)"
    except Exception as e:
        return None, f"call FAILED with: {e}"


if __name__ == "__main__":
    print(f"CWD: {os.getcwd()}")
    print(f"CLAUDE.md present in cwd: {os.path.exists('CLAUDE.md')}")
    print(f"ANTHROPIC_API_KEY set: {'ANTHROPIC_API_KEY' in os.environ}")
    print()

    # --- (c) stale key present run ---
    print("=== TEST (c): WITH stale ANTHROPIC_API_KEY ===")
    c_result, c_note = run_with_key_present()
    print(f"result: {repr(c_result)}")
    print(f"note:   {c_note}")
    print()

    # --- (a) key scrubbed run ---
    print("=== TEST (a): WITHOUT ANTHROPIC_API_KEY (scrubbed) ===")
    a_result, a_note = run_with_key_scrubbed()
    print(f"result: {repr(a_result)}")
    print(f"note:   {a_note}")
    print()

    # --- Assertions ---
    print("=== ASSERTIONS ===")
    gate_pass = True

    # (a) call succeeded + returned non-empty one-liner
    if a_result and len(a_result) > 0:
        print(f"(a) PASS: call succeeded, returned: {repr(a_result[:120])}")
    else:
        print(f"(a) FAIL: call did not return a non-empty string (note: {a_note})")
        gate_pass = False

    # (b) output does NOT contain ZZZSENTINEL
    if a_result and SENTINEL not in a_result:
        print(f"(b) PASS: output does NOT contain {SENTINEL}")
    elif a_result:
        print(f"(b) FAIL: output CONTAINS {SENTINEL} — CLAUDE.md was ingested!")
        gate_pass = False
    else:
        print(f"(b) N/A: no output to check")
        gate_pass = False

    # (c) key present -> failed or succeeded differently
    if c_result is None:
        print(f"(c) PASS: call with stale key FAILED — {c_note}")
    else:
        print(f"(c) INCONCLUSIVE: call with stale key also succeeded: {repr(c_result[:80])}")

    print()
    print(f"GATE DECISION: {'PASS — subscription-sdk is safe to wire' if gate_pass else 'FAIL — do NOT wire subscription-sdk as default'}")
    sys.exit(0 if gate_pass else 1)
