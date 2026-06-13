"""Surface producers: pure functions that compute injection context strings."""
from .producers import (
    Injection,
    produce_prompt,
    produce_analysis,
    produce_symbols,
    produce_open_issues,
    produce_bridge,
)

__all__ = [
    "Injection",
    "produce_prompt",
    "produce_analysis",
    "produce_symbols",
    "produce_open_issues",
    "produce_bridge",
]
