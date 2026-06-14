#!/usr/bin/env python3
"""Debug: with stale key (do not scrub)."""
import asyncio
import os
import claude_agent_sdk

async def test():
    # Do NOT scrub key
    options = claude_agent_sdk.ClaudeAgentOptions(
        setting_sources=[],
        allowed_tools=[],
        model='claude-haiku-4-5',
        cwd='/tmp',
        debug_stderr=None,
    )
    try:
        async for msg in claude_agent_sdk.query(
            prompt='Say: hello world',
            options=options,
        ):
            print(f"MSG TYPE: {type(msg).__name__}")
            for attr in ('content', 'result', 'subtype', 'is_error'):
                if hasattr(msg, attr):
                    val = getattr(msg, attr)
                    print(f"  .{attr} = {repr(val)[:300]}")
            print()
    except Exception as e:
        print(f'EXCEPTION: {type(e).__name__}: {e}')

asyncio.run(test())
