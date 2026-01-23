"""
Content Validation

Functions for validating finding content and tags.
"""

import math
import re
import struct
from .constants import CONTENT_WARNINGS


def validate_finding_content(content: str, tags: list[str] | None = None) -> list[dict[str, str]]:
    """Validate finding content for anti-patterns.

    Returns list of warnings, each with 'type' and 'message' keys.
    Empty list means no issues found.
    """
    warnings: list[dict[str, str]] = []

    for warn_type, config in CONTENT_WARNINGS.items():
        for pattern in config["patterns"]:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                warnings.append({
                    "type": warn_type,
                    "message": str(config["message"]),
                })
                break  # One warning per type is enough

    # Check for "index" or "entry-point" tags
    if tags:
        if "index" in tags or "entry-point" in tags:
            warnings.append({
                "type": "index_tag",
                "message": "INDEX/entry-point tagged findings get stale - avoid creating them",
            })

    return warnings


def validate_tags(tags: list[object] | None) -> list[str]:
    """Validate, normalize, and sanitize a list of tags.

    Normalization:
    - Convert to lowercase
    - Replace spaces with hyphens
    - Strip trailing punctuation (., ?, ', ")

    Removes invalid tags:
    - Too short (< 2 chars) or too long (> 50 chars)
    - Contains garbage characters (multiple special chars, or invalid combos)
    - Unclosed parentheses like 'SO(3' or orphaned closing parens like 'n)'
    - Only punctuation
    - Starts with 'source:' (file references, not semantic tags)

    Returns:
        List of valid, normalized tags (may be empty if all invalid)
    """
    if not tags:
        return []

    valid: list[str] = []
    for t in tags:
        if not isinstance(t, str):
            continue
        t = t.strip()

        # Skip source: file references
        if t.startswith('source:'):
            continue

        # Normalize: lowercase, spaces to hyphens, strip trailing punctuation
        t = t.lower()
        t = t.replace(' ', '-')
        t = t.rstrip('.?\'"')

        # Length check
        if len(t) < 2 or len(t) > 50:
            continue
        # Garbage: contains multiple special chars (not just hyphens, underscores, parens)
        special_count = len(re.findall(r'[<>&#@%;"\'/\\]', t))
        if special_count >= 2:
            continue
        # Unclosed parentheses: starts with letters, has open paren, ends with alphanumeric
        if re.match(r'^[A-Za-z]+\([0-9a-z]+$', t):
            continue
        # Orphaned closing paren: just alphanumeric followed by )
        if re.match(r'^[0-9a-z]+\)$', t):
            continue
        # Only punctuation
        if re.match(r'^[?.!,;:\'"()-]+$', t):
            continue
        valid.append(t)

    # Deduplicate while preserving order
    return list(dict.fromkeys(valid))


def serialize_f32(vector: list[float]) -> bytes:
    """Serialize a float32 vector for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def deserialize_f32(blob: bytes) -> list[float]:
    """Deserialize a float32 vector from sqlite-vec."""
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def l2_normalize(vector: list[float]) -> list[float]:
    """L2 normalize a vector for cosine similarity via L2 distance.

    For L2-normalized vectors: L2_distance² = 2(1 - cosine_similarity)
    So we can compute cosine_similarity = 1 - L2_distance²/2
    """
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]
