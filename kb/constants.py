"""
Knowledge Base Constants

Centralized definitions for finding types, notation domains, and content validation patterns.
"""

from pathlib import Path

# Default paths and embedding/LLM configuration sourced from the single config
# resolver (kb.config). Using force_reload=True so that when this module is
# reloaded (e.g. in tests that patch os.environ), the resolver re-reads env vars
# rather than returning a stale singleton.
from .config import load_config as _load_config

_cfg = _load_config(force_reload=True)

# Keep the same names so all existing importers continue to work unchanged.
DEFAULT_DB_PATH: Path = _cfg.db_path
DEFAULT_EMBEDDING_URL: str = _cfg.embedding_url
DEFAULT_EMBEDDING_DIM: int = _cfg.embedding_dim
DEFAULT_EMBEDDING_FORMAT: str = _cfg.embedding_format
DEFAULT_EMBEDDING_MODEL: str = _cfg.embedding_model
DEFAULT_EMBEDDING_KEY: str = _cfg.embedding_key
DEFAULT_LLM_URL: str = _cfg.llm_url

# Finding types
FINDING_TYPES = ["success", "failure", "experiment", "discovery", "correction"]

# Domain types for notation
NOTATION_DOMAINS = ["physics", "math", "cs", "general"]

# Content validation patterns (anti-patterns to warn about)
CONTENT_WARNINGS = {
    "paper_update": {
        "patterns": [
            r"\d+→\d+\s*pages",  # "20→17 pages"
            r"paper\s+(compiles|updated|condensed)",
            r"section\s+\d+\s+updated",
            r"compiles\s+to\s+\d+\s*pages",
            r"\.tex\s+(cleaned|updated|condensed)",
        ],
        "message": "Looks like a paper update log - these are transient and shouldn't be in KB",
    },
    "absolute_path": {
        "patterns": [
            r"/home/\w+/",  # /home/user/ - absolute paths are fragile
            r"Source:\s*\S+\.(?:py|sage|md)$",  # "Source: FILE.md" as only content
        ],
        "message": "Contains absolute paths which are environment-specific",
    },
    "index_entry": {
        "patterns": [
            r"^INDEX:",  # INDEX: prefix
            r"^GOTCHAS:",  # GOTCHAS: prefix (auto-generated)
        ],
        "message": "INDEX/GOTCHAS entries get stale - use kb_search() instead",
    },
    "nested_reference": {
        "patterns": [
            r"kb-\d{8}-\d{6}-[a-f0-9]{6}",  # kb-YYYYMMDD-HHMMSS-XXXXXX
        ],
        "message": "Contains KB finding references - each finding should be standalone",
    },
    "specific_count": {
        "patterns": [
            r"\b\d+\s+(?:states|fermions|bosons|generators|dimensions)\b",  # "56 states"
            r"\b\d+\s+total\b",  # "64 total"
        ],
        "message": "Contains specific counts that may become stale - describe structure instead",
    },
}

# Greek letter Unicode code points — used ONLY to identify which characters are Greek.
# NEVER use this as a meanings source; meanings come from the notations DB table.
# Meanings here are generic-physics fallbacks that are WRONG for this project.
# (e.g. η = K_48 grading here, NOT metric tensor; α = alpha_triality, NOT fine structure)
GREEK_MEANINGS: dict[str, str] = {
    letter: '' for letter in
    'αβγδεζηθικλμνξοπρστυφχψω'
    'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
}

# Unicode to ASCII mappings for summary generation
UNICODE_TO_ASCII = {
    '⊂': ' subset ', '⊃': ' supset ', '⊆': '<=', '⊇': '>=',
    '∈': ' in ', '∉': ' notin ', '×': 'x', '→': '->', '←': '<-',
    '≈': '~', '≠': '!=', '≤': '<=', '≥': '>=', '∞': 'inf',
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    '⁰': '^0', '¹': '^1', '²': '^2', '³': '^3', '⁴': '^4',
    '′': "'", '″': '"', '‴': "'''",
}

# Allowed Unicode characters in summaries
ALLOWED_UNICODE = set(
    'αβγδεζηθικλμνξοπρστυφχψω'
    + 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
    + '∈∉⊂⊃⊆⊇∩∪∅∞∂∇∫∑∏√'
    + '≈≠≤≥≡≢±×÷'
    + '→←↔⇒⇐⇔'
    + '₀₁₂₃₄₅₆₇₈₉'
    + '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻'
    + '′″‴'
)
