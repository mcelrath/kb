"""
Knowledge Base Package

A SQLite + sqlite-vec powered findings database with:
- Vector similarity search for semantic retrieval
- Supersession chains for correcting outdated findings
- Full-text search fallback
- Project/sprint tagging
"""

from .constants import (
    FINDING_TYPES,
    NOTATION_DOMAINS,
    DEFAULT_DB_PATH,
    DEFAULT_EMBEDDING_URL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_LLM_URL,
)
from .validation import (
    validate_finding_content,
    validate_tags,
    serialize_f32,
    deserialize_f32,
    l2_normalize,
)
from .facade import KnowledgeBase

__all__ = [
    "KnowledgeBase",
    "FINDING_TYPES",
    "NOTATION_DOMAINS",
    "DEFAULT_DB_PATH",
    "DEFAULT_EMBEDDING_URL",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_LLM_URL",
    "validate_finding_content",
    "validate_tags",
    "serialize_f32",
    "deserialize_f32",
    "l2_normalize",
]
