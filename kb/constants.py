"""
Knowledge Base Constants

Centralized definitions for finding types, notation domains, and content validation patterns.
"""

import os
from pathlib import Path

# Default paths
DEFAULT_DB_PATH = Path.home() / ".cache" / "kb" / "knowledge.db"

# Embedding configuration (REQUIRED - no local fallback)
DEFAULT_EMBEDDING_URL = os.environ.get("KB_EMBEDDING_URL", "")
DEFAULT_EMBEDDING_DIM = int(os.environ.get("KB_EMBEDDING_DIM", "4096"))

# LLM configuration for query expansion
DEFAULT_LLM_URL = os.environ.get("KB_LLM_URL", "http://tardis:9510/completion")

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

# Common Greek letter meanings for notation detection
GREEK_MEANINGS = {
    'α': 'fine structure constant / angle',
    'β': 'velocity ratio v/c / angle',
    'γ': 'Lorentz factor / gamma matrix',
    'δ': 'Dirac delta / variation',
    'ε': 'small parameter / Levi-Civita symbol',
    'ζ': 'Riemann zeta',
    'η': 'metric tensor',
    'θ': 'angle / Heaviside step function',
    'ι': 'inclusion map',
    'κ': 'curvature / coupling',
    'λ': 'eigenvalue / wavelength',
    'μ': 'spacetime index / mass scale',
    'ν': 'spacetime index / frequency',
    'ξ': 'gauge parameter / coordinate',
    'ο': 'omicron',
    'π': 'pi / projection',
    'ρ': 'density / representation',
    'σ': 'Pauli matrix / cross section',
    'τ': 'proper time / triality / tau lepton',
    'υ': 'upsilon',
    'φ': 'scalar field / angle',
    'χ': 'susceptibility / character',
    'ψ': 'spinor field / wavefunction',
    'ω': 'angular frequency / cube root of unity',
    'Γ': 'Christoffel symbol / gamma function',
    'Δ': 'Laplacian / gap parameter',
    'Θ': 'Heaviside function',
    'Λ': 'cosmological constant / cutoff',
    'Σ': 'sum / self-energy / bivector',
    'Φ': 'flux / scalar field',
    'Ψ': 'wavefunction',
    'Ω': 'solid angle / density parameter',
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
