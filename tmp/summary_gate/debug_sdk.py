#!/usr/bin/env python3
"""Debug: inspect what messages the SDK stream emits."""
import asyncio
import os
import claude_agent_sdk

async def test():
    env_copy = dict(os.environ)
    env_copy.pop('ANTHROPIC_API_KEY', None)
    options = claude_agent_sdk.ClaudeAgentOptions(
        setting_sources=[],
        allowed_tools=[],
        model='claude-haiku-4-5',
        env=env_copy,
        cwd='/tmp',
        debug_stderr=None,  # suppress internal noise
    )
    try:
        async for msg in claude_agent_sdk.query(
            prompt='Say: hello world',
            options=options,
        ):
            print(f"MSG TYPE: {type(msg).__name__}")
            print(f"  attrs: {[a for a in dir(msg) if not a.startswith('_')]}")
            # Print key attributes
            for attr in ('content', 'text', 'type', 'result', 'subtype', 'is_error'):
                if hasattr(msg, attr):
                    val = getattr(msg, attr)
                    print(f"  .{attr} = {repr(val)[:300]}")
            print()
    except Exception as e:
        print(f'EXCEPTION: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(test())
