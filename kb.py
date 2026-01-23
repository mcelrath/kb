#!/usr/bin/env python3
"""
Knowledge Base - SQLite + sqlite-vec powered findings database.

Records successes, failures, experiments, and discoveries with:
- Vector similarity search for semantic retrieval
- Supersession chains for correcting outdated findings
- Full-text search fallback
- Project/sprint tagging
"""

import argparse
import functools
import hashlib
import json
import os
import re
import sqlite3
import struct
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError
import html

import sqlite_vec

# Optional: rich for terminal markdown rendering
try:
    from rich.console import Console
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Optional: starlette/uvicorn for web server
try:
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocket
    import asyncio
    import uvicorn
    SERVE_AVAILABLE = True
except ImportError:
    SERVE_AVAILABLE = False

# Default paths
DEFAULT_DB_PATH = Path.home() / ".cache" / "kb" / "knowledge.db"

# Embedding configuration (REQUIRED - no local fallback)
DEFAULT_EMBEDDING_URL = os.environ.get("KB_EMBEDDING_URL", "")  # e.g., "http://ash:8080/embedding"
DEFAULT_EMBEDDING_DIM = int(os.environ.get("KB_EMBEDDING_DIM", "4096"))

# LLM configuration for query expansion
DEFAULT_LLM_URL = os.environ.get("KB_LLM_URL", "http://tardis:9510/completion")

# Query expansion cache (module-level for persistence across calls)
_expansion_cache: dict[str, str] = {}

# Embedding cache with LRU eviction (module-level, bounded to 500 entries ~200MB for 4096-dim)
_embedding_cache: dict[str, list[float]] = {}
_embedding_cache_order: list[str] = []
_EMBEDDING_CACHE_MAX = 500


def _cache_embedding(text_hash: str, embedding: list[float]) -> None:
    """Add embedding to cache with LRU eviction."""
    if text_hash in _embedding_cache:
        _embedding_cache_order.remove(text_hash)
        _embedding_cache_order.append(text_hash)
        return
    if len(_embedding_cache) >= _EMBEDDING_CACHE_MAX:
        oldest = _embedding_cache_order.pop(0)
        del _embedding_cache[oldest]
    _embedding_cache[text_hash] = embedding
    _embedding_cache_order.append(text_hash)


def _get_cached_embedding(text_hash: str) -> list[float] | None:
    """Get embedding from cache, updating LRU order."""
    if text_hash in _embedding_cache:
        _embedding_cache_order.remove(text_hash)
        _embedding_cache_order.append(text_hash)
        return _embedding_cache[text_hash]
    return None


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


def validate_finding_content(content: str, tags: list[str] | None = None) -> list[dict]:
    """Validate finding content for anti-patterns.

    Returns list of warnings, each with 'type' and 'message' keys.
    Empty list means no issues found.
    """
    warnings = []

    for warn_type, config in CONTENT_WARNINGS.items():
        for pattern in config["patterns"]:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                warnings.append({
                    "type": warn_type,
                    "message": config["message"],
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
    import math
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


class KnowledgeBase:
    """SQLite + sqlite-vec knowledge base for findings."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        embedding_url: str = DEFAULT_EMBEDDING_URL,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_url = embedding_url
        self.embedding_dim = embedding_dim

        self.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 30000")

        self._init_schema()

    def _embed_remote(self, text: str, max_retries: int = 5, base_delay: float = 2.0) -> list[float]:
        """Get embedding from remote endpoint (llama.cpp style).

        llama.cpp returns per-token embeddings. We use mean pooling to get
        a single embedding for the entire text.

        Retries with exponential backoff + jitter on failure. No fallback to local
        model to prevent dimension mismatch errors.

        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts (default 5 for overloaded servers)
            base_delay: Base delay in seconds (doubles each retry with jitter)

        Raises:
            RuntimeError: If all retries fail
        """
        import random
        last_error = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, delay * 0.25)
                delay += jitter
                print(f"Embedding retry {attempt}/{max_retries} after {delay:.1f}s...", file=sys.stderr)
                time.sleep(delay)

            req = Request(
                self.embedding_url,
                data=json.dumps({"content": text}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    # llama.cpp format: [{"index": 0, "embedding": [[tok1], [tok2], ...]}]
                    # Mean pool across all token embeddings
                    token_embeddings = data[0]["embedding"]
                    if len(token_embeddings) == 1:
                        return token_embeddings[0]
                    # Mean pooling
                    dim = len(token_embeddings[0])
                    pooled = [0.0] * dim
                    for tok_emb in token_embeddings:
                        for i, v in enumerate(tok_emb):
                            pooled[i] += v
                    n = len(token_embeddings)
                    return [v / n for v in pooled]
            except (URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError,
                    ConnectionError, OSError) as e:
                last_error = e
                continue

        raise RuntimeError(
            f"Remote embedding failed after {max_retries} retries: {last_error}. "
            f"Check that embedding server at {self.embedding_url} is running."
        )

    def expand_query(
        self,
        query: str,
        project: str | None = None,
        verbose: bool = False,
    ) -> str:
        """Expand a search query using a local LLM for better recall.

        Uses few-shot prompting optimized for technical/scientific content:
        - Expands acronyms (FMHA → Flash Multi-Head Attention)
        - Preserves compound terms (vector similarity → "vector similarity")
        - Adds domain-specific related terms based on project context
        - Generates synonyms and alternative phrasings

        Args:
            query: The original search query
            project: Optional project name for domain context
            verbose: If True, print the expanded query to stderr

        Returns:
            Expanded query string combining original + generated terms
        """
        # Check cache first (keyed by query + project)
        cache_key = f"{query}|{project or ''}"
        if cache_key in _expansion_cache:
            expanded = _expansion_cache[cache_key]
            if verbose:
                print(f"[cached] Expanded: {expanded}", file=sys.stderr)
            return expanded

        # Determine LLM URL: explicit env var > derived from embedding URL
        llm_url = DEFAULT_LLM_URL
        if not llm_url and self.embedding_url:
            base = self.embedding_url.rsplit("/", 1)[0]
            llm_url = f"{base}/completion"

        if not llm_url:
            if verbose:
                print("Warning: No LLM available for query expansion (set KB_LLM_URL)", file=sys.stderr)
            return query

        # Build domain context
        domain_hint = ""
        if project:
            domain_hint = f"\nDomain context: {project} project (use domain-specific terminology)"

        # Technical few-shot prompt
        prompt = f"""You are a search query expansion assistant for a technical knowledge base.
Given a search query, output ONLY additional search terms on a single line.

Rules:
1. Expand acronyms: FMHA → "Flash Multi-Head Attention" FMHA
2. Keep multi-word concepts quoted: "attention mechanism" not attention mechanism
3. Add closely related technical terms and synonyms
4. Include both formal and informal variations
5. Output terms separated by spaces, NO explanations
{domain_hint}
Examples:
Query: GEMM performance
Output: "General Matrix Multiply" GEMM matmul "matrix multiplication" BLAS cuBLAS rocBLAS throughput latency

Query: CUDA kernel launch
Output: "kernel launch" GPU "thread block" grid warp __global__ hipLaunchKernelGGL

Query: quaternion rotation
Output: "quaternion rotation" "unit quaternion" SO(3) "rotation matrix" "Euler angles" versor

Query: {query}
Output:"""

        req = Request(
            llm_url,
            data=json.dumps({
                "prompt": prompt,
                "n_predict": 150,
                "temperature": 0.2,
                "stop": ["\n\n", "\nQuery:", "\n\n"],
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                expansion = data.get("content", "").strip()
                # Handle JSON-wrapped responses from LLM
                expansion = self._extract_text_from_json(expansion, keys=["expansion", "terms", "output", "result"])
                # Clean up: remove any newlines, extra whitespace
                expansion = " ".join(expansion.split())
                if expansion:
                    expanded = f"{query} {expansion}"
                    _expansion_cache[cache_key] = expanded
                    if verbose:
                        print(f"Expanded: {expanded}", file=sys.stderr)
                    return expanded
        except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            if verbose:
                print(f"Warning: Query expansion failed ({e})", file=sys.stderr)

        # Cache and return original on failure
        _expansion_cache[cache_key] = query
        return query

    def _llm_complete(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.3,
        stop: list[str] | None = None,
        timeout: int = 30,
        use_chat: bool = True,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str | None:
        """Generic LLM completion helper.

        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences (only for raw completion mode)
            timeout: Request timeout in seconds
            use_chat: Use chat completion API (recommended for better format adherence)
            system_prompt: System prompt for chat mode
            json_mode: If True, request JSON output format (llama.cpp response_format)

        Returns the completion text, or None on failure.
        """
        llm_url = DEFAULT_LLM_URL
        if not llm_url:
            return None

        try:
            if use_chat:
                # Use chat completion API for better format adherence
                chat_url = llm_url.replace("/completion", "/v1/chat/completions")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                request_body = {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }

                # Enable JSON mode if requested (llama.cpp supports this)
                if json_mode:
                    request_body["response_format"] = {"type": "json_object"}

                req = Request(
                    chat_url,
                    data=json.dumps(request_body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data["choices"][0]["message"]["content"].strip()
                    return self._strip_thinking(content)
            else:
                # Raw completion API
                if stop is None:
                    stop = ["\n\n"]
                req = Request(
                    llm_url,
                    data=json.dumps({
                        "prompt": prompt,
                        "n_predict": max_tokens,
                        "temperature": temperature,
                        "stop": stop,
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data.get("content", "").strip()
                    return self._strip_thinking(content)
        except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            return None

    def _strip_thinking(self, text: str) -> str:
        """Remove <think>...</think> blocks from LLM output."""
        if not text:
            return text
        # Remove thinking blocks (handles multiline)
        result = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        return result.strip()

    def _extract_text_from_json(self, text: str, keys: list[str] | None = None) -> str:
        """Extract text content from JSON-wrapped LLM responses.

        The LLM is configured for JSON-only output, so responses often come wrapped
        in JSON even when we want plain text. This helper extracts the actual content.

        Args:
            text: The raw LLM response (may be JSON or plain text)
            keys: Preferred keys to look for (defaults to common ones)

        Returns:
            Extracted text content, or empty string if error/invalid
        """
        if not text:
            return ""
        text = text.strip()
        if not text.startswith("{"):
            return text

        if keys is None:
            keys = ["text", "result", "output", "answer", "response", "content",
                    "summary", "tags", "type", "signature", "expansion", "terms"]

        # Try parsing twice: normal, then with escaped backslashes for LaTeX
        for json_str in [text, text.replace("\\", "\\\\")]:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    # Check for error responses
                    if parsed.get("error") is True:
                        return ""
                    if "error" in str(parsed.get("message", "")).lower():
                        return ""
                    # Try preferred keys in order
                    for key in keys:
                        if key in parsed:
                            val = parsed[key]
                            if isinstance(val, str) and val.strip():
                                return val.strip()
                            if isinstance(val, list):
                                # For lists (like tags), join with commas
                                return ", ".join(str(v) for v in val if v)
                            if isinstance(val, dict):
                                # For nested dicts, look for description/text fields
                                for nested_key in ["description", "text", "content", "summary", "analysis"]:
                                    if nested_key in val:
                                        nested_val = val[nested_key]
                                        if isinstance(nested_val, str):
                                            return nested_val.strip()
                                        if isinstance(nested_val, list):
                                            return " ".join(str(v) for v in nested_val if v)
                    # Fall back to first non-trivial content (any key)
                    for val in parsed.values():
                        if isinstance(val, str) and len(val.strip()) > 2:
                            return val.strip()
                        # Top-level list of strings or dicts
                        if isinstance(val, list) and val:
                            # Check if it's a list of strings
                            if all(isinstance(v, str) for v in val):
                                return " ".join(str(v) for v in val if v)
                            # Check for list of dicts with text fields
                            for item in val:
                                if isinstance(item, dict):
                                    for nested_key in ["description", "text", "content", "summary", "analysis", "reasoning"]:
                                        if nested_key in item:
                                            nested_val = item[nested_key]
                                            if isinstance(nested_val, str) and nested_val.strip():
                                                return nested_val.strip()
                        # Nested dict with any text content
                        if isinstance(val, dict):
                            for nested_val in val.values():
                                if isinstance(nested_val, str) and len(nested_val.strip()) > 2:
                                    return nested_val.strip()
                                if isinstance(nested_val, list) and nested_val:
                                    if all(isinstance(v, str) for v in nested_val):
                                        return " ".join(str(v) for v in nested_val if v)
                    # Last resort: stringify the first non-trivial value
                    for val in parsed.values():
                        if isinstance(val, (dict, list)) and val:
                            try:
                                return json.dumps(val, ensure_ascii=False, indent=2)[:500]
                            except (TypeError, ValueError):
                                pass
                return ""  # Parsed but no usable content
            except json.JSONDecodeError:
                continue
        return text  # Not valid JSON, return as-is

    def _generate_summary(self, content: str, evidence: str | None = None) -> str | None:
        """Generate a one-line summary for a finding.

        Args:
            content: The finding content to summarize
            evidence: Optional evidence to consider

        Returns:
            A short summary (max ~100 chars) or None on failure
        """
        system_prompt = "You write concise one-line summaries. Output ONLY the summary, no intro phrases like 'This finding shows' or 'Summary:'. Max 80 chars."

        # Normalize Unicode math symbols to ASCII to avoid confusing the LLM
        text = content[:500]
        # Strip emotional/error-like prefixes that confuse the JSON-only LLM
        # (it interprets "FATAL", "ERROR", etc. as error indicators)
        text = re.sub(r'^(?:CRITICAL\s+)?(?:CORRECTION|ERROR|FATAL\s+FLAW|WARNING|NOTE|IMPORTANT):\s*', '', text, flags=re.IGNORECASE)
        unicode_to_ascii = {
            '⊂': ' subset ', '⊃': ' supset ', '⊆': '<=', '⊇': '>=',
            '∈': ' in ', '∉': ' notin ', '×': 'x', '→': '->', '←': '<-',
            '≈': '~', '≠': '!=', '≤': '<=', '≥': '>=', '∞': 'inf',
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
            '⁰': '^0', '¹': '^1', '²': '^2', '³': '^3', '⁴': '^4',
            '′': "'", '″': '"', '‴': "'''",
        }
        for uc, asc in unicode_to_ascii.items():
            text = text.replace(uc, asc)
        # Strip remaining non-ASCII
        text = text.encode('ascii', 'ignore').decode('ascii')

        if evidence:
            ev = evidence[:200].encode('ascii', 'ignore').decode('ascii')
            text += f"\nEvidence: {ev}"

        prompt = f"Summarize in ONE technical line (max 80 chars):\n{text}"

        result = self._llm_complete(
            prompt,
            max_tokens=150,  # Needs room for JSON wrapper + actual summary content
            temperature=0.2,
            system_prompt=system_prompt,
            timeout=30,
        )

        if result:
            # Extract from JSON wrapper using consolidated helper
            result = self._extract_text_from_json(result, keys=["summary", "result", "text", "output"])

            if result:
                # Clean: strip quotes, remove garbage
                summary = result.strip().strip('"').strip("'")
                # Remove literal Unicode escapes that weren't decoded
                summary = re.sub(r'\\u[0-9a-fA-F]{4}', '', summary)
                # Remove control chars
                summary = re.sub(r'[\x00-\x1f\x7f]', '', summary)
                # Keep only ASCII + specific common math Unicode chars
                allowed_unicode = set(
                    'αβγδεζηθικλμνξοπρστυφχψω'  # Greek lowercase
                    'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'  # Greek uppercase
                    '∈∉⊂⊃⊆⊇∩∪∅∞∂∇∫∑∏√'  # Set/calculus
                    '≈≠≤≥≡≢±×÷'  # Relations/operators
                    '→←↔⇒⇐⇔'  # Arrows
                    '₀₁₂₃₄₅₆₇₈₉'  # Subscripts
                    '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻'  # Superscripts
                    '′″‴'  # Primes
                )
                summary = ''.join(c for c in summary if ord(c) < 128 or c in allowed_unicode)
                # Clean up multiple spaces
                summary = re.sub(r'  +', ' ', summary).strip()
                # Check for garbage: need real words and reasonable letter ratio
                words = re.findall(r'[a-zA-Z]{3,}', summary)
                letter_ratio = sum(1 for c in summary if c.isalpha()) / max(len(summary), 1)
                if (summary and not summary.startswith("{") and len(summary) > 10
                        and len(words) >= 3 and letter_ratio > 0.5):
                    return summary[:100]

        # Fallback: first sentence or truncated content
        first_sentence = content.split('.')[0]
        if len(first_sentence) <= 100:
            return first_sentence
        return content[:97] + "..."

    def suggest_tags(self, content: str, project: str | None = None) -> list[str]:
        """Suggest tags for a finding based on its content using LLM."""
        # Get existing tags from the project for context
        existing_tags: set[str] = set()
        if project:
            rows = self.conn.execute(
                "SELECT DISTINCT tags FROM findings WHERE project = ? AND tags IS NOT NULL",
                (project,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT tags FROM findings WHERE tags IS NOT NULL"
            ).fetchall()

        for row in rows:
            if row[0]:
                try:
                    existing_tags.update(json.loads(row[0]))
                except json.JSONDecodeError:
                    pass

        existing_list = ", ".join(sorted(existing_tags)[:30]) if existing_tags else "none yet"

        system_prompt = "You suggest tags for knowledge base findings. Return JSON with a 'tags' array."

        prompt = f"""Suggest 2-5 tags for this finding. Return JSON: {{"tags": ["tag1", "tag2", ...]}}

Existing tags to prefer: {existing_list}
Status tags: proven, heuristic, open-problem
Importance tags: core-result, technique, detail
Dimension tags: dim-2, dim-4, dim-8

Content: {content[:500]}"""

        result = self._llm_complete(
            prompt,
            max_tokens=100,
            temperature=0.2,
            system_prompt=system_prompt,
            timeout=60,
            json_mode=True,
        )
        if not result:
            return []

        # Extract from JSON if LLM returned JSON-wrapped response
        result = self._extract_text_from_json(result, keys=["tags"])

        # Parse comma-separated tags, handle potential formatting issues
        # Remove any markdown formatting or extra text
        result = result.strip().strip('`').strip()
        if result.startswith("Tags:"):
            result = result[5:]

        # Clean and validate tags
        import re
        tags = []
        for t in result.split(","):
            t = t.strip().lower().replace(" ", "-")
            # Remove quotes and normalize unicode dashes
            t = t.strip('"\'').replace("‑", "-").replace("–", "-").replace("—", "-")
            # Only keep valid tag format: alphanumeric and hyphens
            if t and len(t) < 30 and re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$', t):
                tags.append(t)
        return tags[:5]  # Max 5 tags

    def detect_duplicates(
        self, content: str, project: str | None = None, threshold: float = 0.85
    ) -> list[dict]:
        """Check if similar findings already exist before adding.

        Returns list of potentially duplicate findings with similarity scores.
        """
        # First do vector similarity search
        similar = self.search(content, limit=5, project=project)

        if not similar:
            return []

        # Filter by threshold
        candidates = [s for s in similar if s.get("similarity", 0) >= threshold]

        if not candidates:
            return []

        # Use LLM to confirm semantic duplicates
        duplicates = []
        for candidate in candidates[:3]:  # Check top 3
            prompt = f"""Are these two findings saying essentially the same thing? Return JSON: {{"answer": true}} or {{"answer": false}}

Finding 1: {content[:300]}

Finding 2: {candidate['content'][:300]}"""

            result = self._llm_complete(prompt, max_tokens=100, temperature=0.1, json_mode=True)
            if result:
                # Parse JSON response - handle both boolean and string
                try:
                    data = json.loads(result)
                    answer = data.get("answer", False)
                    is_duplicate = answer is True or str(answer).upper() in ("YES", "TRUE")
                except json.JSONDecodeError:
                    # Fallback: check for YES in text
                    is_duplicate = "YES" in result.upper()
                if is_duplicate:
                    duplicates.append(candidate)

        return duplicates

    def classify_finding_type(self, content: str) -> str:
        """Suggest finding type based on content."""
        system_prompt = "You classify findings. Return JSON with 'type' field."

        prompt = f"""Classify this finding. Return JSON: {{"type": "<type>"}}

Types:
- success: Verified working approach
- failure: Something that doesn't work
- discovery: New understanding or insight
- experiment: Inconclusive, needs more work

Content: {content[:400]}"""

        result = self._llm_complete(
            prompt,
            max_tokens=50,
            temperature=0.1,
            system_prompt=system_prompt,
            timeout=30,
            json_mode=True,
        )
        if result:
            # Extract from JSON
            result = self._extract_text_from_json(result, keys=["type", "classification", "result"])
            result = result.lower().strip().split()[0]  # Take first word only
            if result in ("success", "failure", "discovery", "experiment"):
                return result

        return "discovery"  # Default

    def normalize_error_signature(self, error_text: str) -> str:
        """Normalize an error message to a canonical signature for matching."""
        system_prompt = "You extract error signatures. Return JSON with 'signature' field."

        prompt = f"""Extract a canonical error signature. Return JSON: {{"signature": "<normalized error>"}}

Rules:
- Remove paths, line numbers, memory addresses
- Keep error type and core message
- Use placeholders: <N> for numbers, <PATH> for paths, <ADDR> for addresses

Error: {error_text[:500]}"""

        result = self._llm_complete(
            prompt,
            max_tokens=150,
            temperature=0.1,
            system_prompt=system_prompt,
            timeout=30,
            json_mode=True,
        )
        if result:
            # Extract from JSON
            result = self._extract_text_from_json(result, keys=["signature", "error", "result"])
            if result:
                # Take first line only
                return result.strip().split('\n')[0].strip()

        # Fallback: basic normalization
        import re
        sig = re.sub(r'/[\w/.-]+', '<PATH>', error_text)
        sig = re.sub(r':\d+', ':<N>', sig)
        sig = re.sub(r'0x[0-9a-fA-F]+', '<ADDR>', sig)
        sig = re.sub(r'\b\d+\b', '<N>', sig)
        return sig[:200]

    def validate_finding_llm(self, content: str, tags: list[str] | None = None) -> dict:
        """LLM-based semantic validation of finding content.

        Catches anti-patterns that regex misses, like:
        - Transient information disguised as findings
        - Incomplete findings that need more context
        - Findings that should be split or merged

        Returns dict with 'is_valid', 'issues', 'quality_score', 'suggestions'.
        """
        system_prompt = "You validate knowledge base findings. Return JSON with validation results."

        prompt = f"""Evaluate this knowledge base finding. Return JSON:
{{
  "is_valid": true/false,
  "quality_score": 1-5,
  "issues": ["issue1", ...] or [],
  "suggestions": ["fix1", ...] or []
}}

Quality criteria:
- 5: Excellent standalone insight
- 3: Acceptable finding
- 1: Transient/should delete

Bad patterns: paper edit logs, index entries, transient counts, absolute paths (/home/...)

Content: {content[:800]}
Tags: {', '.join(tags) if tags else 'none'}"""

        result = self._llm_complete(
            prompt,
            max_tokens=300,
            temperature=0.2,
            system_prompt=system_prompt,
            timeout=60,
            json_mode=True,
        )

        # Parse response
        validation = {
            "is_valid": True,
            "quality_score": 3,
            "issues": [],
            "suggestions": [],
        }

        if result:
            result_stripped = result.strip()
            # Handle JSON-wrapped responses from LLM
            if result_stripped.startswith("{"):
                try:
                    data = json.loads(result_stripped)
                    if "is_valid" in data or "valid" in data:
                        val = data.get("is_valid", data.get("valid"))
                        validation["is_valid"] = val is True or str(val).upper() == "YES"
                    if "quality" in data or "quality_score" in data:
                        score = data.get("quality_score", data.get("quality", 3))
                        if isinstance(score, int):
                            validation["quality_score"] = max(1, min(5, score))
                    if "issues" in data and isinstance(data["issues"], list):
                        validation["issues"] = [str(i) for i in data["issues"] if i]
                    if "suggestions" in data and isinstance(data["suggestions"], list):
                        validation["suggestions"] = [str(s) for s in data["suggestions"] if s]
                    elif "fix" in data and data["fix"]:
                        validation["suggestions"] = [str(data["fix"])]
                    return validation
                except json.JSONDecodeError:
                    pass

            # Fall back to line-based parsing
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("IS_VALID:"):
                    validation["is_valid"] = "YES" in line.upper()
                elif line.startswith("QUALITY:"):
                    try:
                        score = int(''.join(c for c in line if c.isdigit())[:1])
                        validation["quality_score"] = max(1, min(5, score))
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("ISSUES:"):
                    issues_text = line.split(":", 1)[1].strip()
                    if issues_text.lower() not in ("none", "n/a", ""):
                        validation["issues"].append(issues_text)
                elif line.startswith("FIX:"):
                    fix_text = line.split(":", 1)[1].strip()
                    if fix_text.lower() not in ("none", "n/a", ""):
                        validation["suggestions"].append(fix_text)

        return validation

    def suggest_finding_fix(self, content: str, issues: list[str]) -> str | None:
        """Generate corrected content for a finding with issues.

        Args:
            content: Original finding content
            issues: List of identified issues to fix

        Returns:
            Suggested corrected content, or None if no fix possible.
        """
        if not issues:
            return None

        system_prompt = """You fix knowledge base findings. Output ONLY the corrected content.
Rules:
- KEEP relative file paths to scripts/code (lib/foo.py, calculus_4d/bar.sage) - these are valuable references
- REMOVE absolute paths (/home/user/...) - replace with relative paths
- Remove specific counts that may change (e.g., "56 states" -> "the states at N=3")
- Remove paper edit details (page counts, section numbers), keep only substantive insights
- Make findings standalone (no references to other finding IDs like kb-XXXXX)
- Keep technical accuracy and key details"""

        prompt = f"""Fix this finding:

Original: {content[:1000]}

Issues to fix:
{chr(10).join(f'- {issue}' for issue in issues)}

Corrected content (output ONLY the fixed text):"""

        result = self._llm_complete(
            prompt,
            max_tokens=500,
            temperature=0.3,
            system_prompt=system_prompt,
            timeout=60,
        )

        if result:
            # Extract from JSON if LLM returned JSON-wrapped response
            result = self._extract_text_from_json(result, keys=["content", "corrected", "fix", "text"])
            if len(result.strip()) > 20:
                return result.strip()
        return None

    def suggest_cross_references(
        self, finding_id: str, content: str, project: str | None = None
    ) -> dict:
        """Suggest related findings, scripts, and docs to link."""
        suggestions: dict = {"findings": [], "scripts": [], "docs": []}

        # Find related findings
        related = self.search(content, limit=5, project=project)
        for r in related:
            if r["id"] != finding_id and r.get("similarity", 0) > 0.6:
                suggestions["findings"].append({
                    "id": r["id"],
                    "content": r["content"][:100],
                    "similarity": r.get("similarity", 0)
                })

        # Find related scripts
        scripts = self.script_search(content, project=project, limit=3)
        for s in scripts:
            if s.get("similarity", 0) > 0.5:
                suggestions["scripts"].append({
                    "id": s["id"],
                    "filename": s["filename"],
                    "purpose": s.get("purpose", "")[:100],
                    "similarity": s.get("similarity", 0)
                })

        # Find related docs
        docs = self.doc_search(content, project=project)
        for d in docs[:3]:
            suggestions["docs"].append({
                "id": d["id"],
                "title": d["title"],
            })

        return suggestions

    def summarize_evidence(self, evidence: str, max_length: int = 200) -> str:
        """Summarize long evidence text."""
        if len(evidence) <= max_length:
            return evidence

        prompt = f"""Summarize this evidence/output concisely, preserving key technical details.

Evidence:
{evidence[:1500]}

Summary (max {max_length} chars):"""

        result = self._llm_complete(prompt, max_tokens=100, temperature=0.2, stop=["\n\n"])
        if result:
            # Extract from JSON if LLM returned JSON-wrapped response
            result = self._extract_text_from_json(result, keys=["summary", "text", "result"])
            return result[:max_length]

        # Fallback: truncate
        return evidence[:max_length-3] + "..."

    def detect_notations(self, content: str, project: str | None = None) -> list[dict]:
        """Detect mathematical/physics notations in content that should be tracked.

        Uses hybrid approach: regex extraction + hardcoded meanings for reliability.
        """
        # Common Greek letter meanings in physics/math
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

        # Extract Greek letters using regex (reliable)
        greek_pattern = r'[α-ωΑ-Ω]'
        found_greek = sorted(set(re.findall(greek_pattern, content)))

        parsed = []
        for letter in found_greek:
            meaning = GREEK_MEANINGS.get(letter, '')
            parsed.append({"symbol": letter, "meaning": meaning})

        if not parsed:
            return []

        # Batch lookup: get all existing notations for this project in one query
        symbols = [p["symbol"] for p in parsed]
        placeholders = ",".join("?" * len(symbols))
        sql = f"SELECT current_symbol FROM notations WHERE current_symbol IN ({placeholders})"
        params = symbols
        if project:
            sql += " AND project = ?"
            params = symbols + [project]
        existing_symbols = {row[0] for row in self.conn.execute(sql, params).fetchall()}

        # Build results with exists flag
        return [
            {"symbol": p["symbol"], "meaning": p["meaning"], "exists": p["symbol"] in existing_symbols}
            for p in parsed
        ]

    def extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from text for reconciliation."""
        system_prompt = "You extract factual claims from text. Return JSON with 'claims' array."

        prompt = f"""Extract distinct factual claims from this text. Return JSON:
{{"claims": ["claim 1", "claim 2", ...]}}

Rules:
1. Each claim should be a single verifiable statement
2. Include mathematical results, theorems, properties
3. Skip vague or context-dependent statements
4. Max 10 claims

Text: {text[:1500]}"""

        result = self._llm_complete(prompt, max_tokens=500, temperature=0.3, system_prompt=system_prompt, json_mode=True)
        if not result:
            return []

        claims = []
        result_stripped = result.strip()

        # Handle JSON-wrapped responses from LLM
        if result_stripped.startswith("{"):
            try:
                data = json.loads(result_stripped)
                if isinstance(data.get("claims"), list):
                    for claim in data["claims"]:
                        if isinstance(claim, str) and len(claim) > 20:
                            claims.append(claim)
                    return claims[:10]
            except json.JSONDecodeError:
                pass

        # Fall back to line-based parsing
        for line in result.split("\n"):
            line = line.strip()
            # Remove numbering
            if line and len(line) > 20:
                import re
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line:
                    claims.append(line)

        return claims[:10]

    def suggest_consolidation(self, project: str | None = None, limit: int = 50) -> list[dict]:
        """Find clusters of related findings that might be consolidated."""
        # Get recent findings
        findings = self.list_findings(project=project, limit=limit)
        if len(findings) < 3:
            return []

        # Group by similarity
        clusters: list[dict] = []
        used_ids: set[str] = set()

        for f in findings:
            if f["id"] in used_ids:
                continue

            # Find similar findings
            similar = self.search(f["content"], limit=5, project=project)
            cluster_members = [f]

            for s in similar:
                if s["id"] != f["id"] and s["id"] not in used_ids:
                    if s.get("similarity", 0) > 0.7:
                        cluster_members.append(s)
                        used_ids.add(s["id"])

            if len(cluster_members) >= 2:
                used_ids.add(f["id"])
                # Use LLM to suggest consolidation
                contents = "\n---\n".join([m["content"][:200] for m in cluster_members[:4]])
                system_prompt = "You analyze related findings for consolidation. Return JSON with 'analysis' field."
                prompt = f"""Analyze these related findings. Return JSON: {{"analysis": "<your analysis>"}}

Should they be consolidated? If yes, suggest a combined summary. If no, explain why distinct.

Findings:
{contents}"""

                result = self._llm_complete(prompt, max_tokens=400, temperature=0.3, system_prompt=system_prompt, json_mode=True)
                analysis = None
                if result:
                    # Extract from JSON response - LLM uses varied keys
                    analysis = self._extract_text_from_json(result, keys=[
                        "analysis", "summary", "result", "text", "response",
                        "consolidated_summary", "combined_summary", "recommendation", "reasoning"
                    ])
                clusters.append({
                    "members": [{"id": m["id"], "content": m["content"][:100]} for m in cluster_members],
                    "analysis": analysis or "Analysis unavailable",
                })

        return clusters

    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL CHECK(type IN ('success', 'failure', 'experiment', 'discovery', 'correction')),
                status TEXT DEFAULT 'current' CHECK(status IN ('current', 'superseded')),
                supersedes_id TEXT REFERENCES findings(id),
                project TEXT,
                sprint TEXT,
                tags TEXT,  -- JSON array
                content TEXT NOT NULL,
                summary TEXT,  -- LLM-generated one-line summary for search results
                evidence TEXT,  -- Supporting evidence (log snippets, test output)
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_findings_status ON findings(status);
            CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(type);
            CREATE INDEX IF NOT EXISTS idx_findings_project ON findings(project);
            CREATE INDEX IF NOT EXISTS idx_findings_supersedes ON findings(supersedes_id);
            CREATE INDEX IF NOT EXISTS idx_findings_created_at ON findings(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_findings_project_status ON findings(project, status);

            CREATE VIRTUAL TABLE IF NOT EXISTS findings_fts USING fts5(
                content, evidence, tags,
                content='findings',
                content_rowid='rowid'
            );

            -- Notation tracking tables
            CREATE TABLE IF NOT EXISTS notations (
                id TEXT PRIMARY KEY,
                current_symbol TEXT NOT NULL,
                meaning TEXT NOT NULL,
                project TEXT,
                domain TEXT CHECK(domain IN ('physics', 'math', 'cs', 'general')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS notation_history (
                id TEXT PRIMARY KEY,
                notation_id TEXT NOT NULL REFERENCES notations(id),
                old_symbol TEXT NOT NULL,
                new_symbol TEXT NOT NULL,
                reason TEXT,
                changed_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_notations_project ON notations(project);
            CREATE INDEX IF NOT EXISTS idx_notations_symbol ON notations(current_symbol);
            CREATE INDEX IF NOT EXISTS idx_notations_project_symbol ON notations(project, current_symbol);
            CREATE INDEX IF NOT EXISTS idx_notation_history_notation ON notation_history(notation_id);

            -- Error tracking and solution linking
            CREATE TABLE IF NOT EXISTS errors (
                id TEXT PRIMARY KEY,
                signature TEXT NOT NULL,  -- Error message or pattern
                error_type TEXT,  -- build, runtime, test, etc.
                project TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                occurrence_count INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS error_solutions (
                error_id TEXT NOT NULL REFERENCES errors(id),
                finding_id TEXT NOT NULL REFERENCES findings(id),
                linked_at TEXT NOT NULL,
                verified INTEGER DEFAULT 0,  -- 1 if solution was confirmed to work
                PRIMARY KEY (error_id, finding_id)
            );

            -- Authoritative documents (specs, papers, standards)
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                url TEXT,  -- URL or file path
                doc_type TEXT NOT NULL CHECK(doc_type IN ('spec', 'paper', 'standard', 'internal', 'reference')),
                project TEXT,
                status TEXT DEFAULT 'active' CHECK(status IN ('active', 'superseded', 'deprecated')),
                summary TEXT,  -- Brief description of the document
                created_at TEXT NOT NULL,
                superseded_by TEXT REFERENCES documents(id)
            );

            -- Links between findings and documents they cite
            CREATE TABLE IF NOT EXISTS document_citations (
                finding_id TEXT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                citation_type TEXT DEFAULT 'references' CHECK(citation_type IN ('references', 'implements', 'contradicts', 'extends')),
                notes TEXT,
                cited_at TEXT NOT NULL,
                PRIMARY KEY (finding_id, document_id)
            );

            CREATE INDEX IF NOT EXISTS idx_errors_project ON errors(project);
            CREATE INDEX IF NOT EXISTS idx_errors_signature ON errors(signature);
            CREATE INDEX IF NOT EXISTS idx_error_solutions_error ON error_solutions(error_id);
            CREATE INDEX IF NOT EXISTS idx_error_solutions_finding ON error_solutions(finding_id);
            CREATE INDEX IF NOT EXISTS idx_error_solutions_verified ON error_solutions(verified);
            CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project);
            CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(doc_type);
            CREATE INDEX IF NOT EXISTS idx_document_citations_doc ON document_citations(document_id);
            CREATE INDEX IF NOT EXISTS idx_document_citations_finding ON document_citations(finding_id);

            -- Script registry for tracking hypothesis-testing scripts
            CREATE TABLE IF NOT EXISTS scripts (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,  -- Original file path
                filename TEXT NOT NULL,  -- Just the filename
                content_hash TEXT NOT NULL,  -- SHA256 of content for deduplication
                content TEXT,  -- Full script content (optional, for small scripts)
                purpose TEXT NOT NULL,  -- What hypothesis/question this script tests
                project TEXT,
                language TEXT DEFAULT 'python' CHECK(language IN ('python', 'sage', 'bash', 'other')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- Links between findings and scripts that generated them
            CREATE TABLE IF NOT EXISTS finding_scripts (
                finding_id TEXT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
                script_id TEXT NOT NULL REFERENCES scripts(id) ON DELETE CASCADE,
                relationship TEXT DEFAULT 'generated_by' CHECK(relationship IN ('generated_by', 'validates', 'contradicts')),
                linked_at TEXT NOT NULL,
                PRIMARY KEY (finding_id, script_id)
            );

            CREATE INDEX IF NOT EXISTS idx_scripts_project ON scripts(project);
            CREATE INDEX IF NOT EXISTS idx_scripts_hash ON scripts(content_hash);
            CREATE INDEX IF NOT EXISTS idx_scripts_filename ON scripts(filename);
            CREATE INDEX IF NOT EXISTS idx_finding_scripts_script ON finding_scripts(script_id);
            CREATE INDEX IF NOT EXISTS idx_finding_scripts_finding ON finding_scripts(finding_id);

            CREATE TRIGGER IF NOT EXISTS findings_ai AFTER INSERT ON findings BEGIN
                INSERT INTO findings_fts(rowid, content, evidence, tags)
                VALUES (new.rowid, new.content, new.evidence, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS findings_ad AFTER DELETE ON findings BEGIN
                INSERT INTO findings_fts(findings_fts, rowid, content, evidence, tags)
                VALUES ('delete', old.rowid, old.content, old.evidence, old.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS findings_au AFTER UPDATE ON findings BEGIN
                INSERT INTO findings_fts(findings_fts, rowid, content, evidence, tags)
                VALUES ('delete', old.rowid, old.content, old.evidence, old.tags);
                INSERT INTO findings_fts(rowid, content, evidence, tags)
                VALUES (new.rowid, new.content, new.evidence, new.tags);
            END;
        """)

        # Create vector table for embeddings
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS findings_vec USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{self.embedding_dim}]
            )
        """)

        # Create vector table for script purpose embeddings
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS scripts_vec USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{self.embedding_dim}]
            )
        """)

        # Schema migration: add summary column if not exists
        try:
            self.conn.execute("SELECT summary FROM findings LIMIT 1")
        except sqlite3.OperationalError:
            self.conn.execute("ALTER TABLE findings ADD COLUMN summary TEXT")

        self.conn.commit()

    def _validate_tags(self, tags: list[str] | None) -> list[str]:
        """Validate and sanitize a list of tags.

        Removes invalid tags:
        - Too short (< 2 chars) or too long (> 50 chars)
        - Contains garbage characters (multiple special chars, or invalid combos)
        - Unclosed parentheses like 'SO(3' or orphaned closing parens like 'n)'
        - Only punctuation

        Returns:
            List of valid tags (may be empty if all invalid)
        """
        if not tags:
            return []

        valid = []
        for t in tags:
            if not isinstance(t, str):
                continue
            t = t.strip()
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
        return valid

    def _embed(self, text: str) -> bytes:
        """Generate embedding for text using remote endpoint.

        Embeddings are L2-normalized so L2 distance can be used for cosine similarity.
        For normalized vectors: cosine_similarity = 1 - L2_distance²/2

        Results are cached (LRU, max 500 entries) to avoid redundant API calls.

        Raises:
            RuntimeError: If KB_EMBEDDING_URL is not configured
        """
        if not self.embedding_url:
            raise RuntimeError(
                "KB_EMBEDDING_URL not configured. Set this environment variable to your embedding endpoint."
            )

        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cached = _get_cached_embedding(text_hash)
        if cached is not None:
            return serialize_f32(cached)

        embedding = self._embed_remote(text)
        embedding = l2_normalize(embedding)
        _cache_embedding(text_hash, embedding)
        return serialize_f32(embedding)

    def check_duplicate(
        self,
        content: str,
        evidence: str | None = None,
        threshold: float = 0.85,
    ) -> tuple[bool, dict | None, bytes]:
        """Check if a similar finding already exists.

        Returns (is_duplicate, existing_finding, embedding) tuple.
        The embedding is returned so it can be reused by add() to avoid double embedding.
        """
        text = content + " " + (evidence or "")
        embedding = self._embed(text)

        # Search for similar findings
        rows = self.conn.execute("""
            SELECT f.*, v.distance
            FROM findings f
            JOIN findings_vec v ON f.id = v.id
            WHERE v.embedding MATCH ?
            AND k = 3
            AND f.status = 'current'
        """, (embedding,)).fetchall()

        for row in rows:
            similarity = 1 - (row["distance"] ** 2) / 2
            if similarity >= threshold:
                return True, {
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"],
                    "similarity": similarity,
                }, embedding
        return False, None, embedding

    def add(
        self,
        content: str,
        finding_type: str | None = None,
        project: str | None = None,
        sprint: str | None = None,
        tags: list[str] | None = None,
        evidence: str | None = None,
        check_duplicate: bool = True,
        duplicate_threshold: float = 0.85,
        check_contradictions: bool = True,
        auto_tag: bool = True,
        auto_classify: bool = True,
        auto_summarize_evidence: bool = True,
        max_evidence_length: int = 500,
    ) -> dict:
        """Add a new finding with automatic LLM enhancements.

        Args:
            content: The finding content
            finding_type: Type (auto-classified if None): success, failure, discovery, experiment
            project: Project name
            sprint: Sprint name
            tags: Tags (auto-suggested if None/empty and auto_tag=True)
            evidence: Evidence text (auto-summarized if long and auto_summarize_evidence=True)
            check_duplicate: If True, warns if a similar finding exists
            duplicate_threshold: Similarity threshold for duplicate detection (0.0-1.0)
            check_contradictions: If True, checks for contradicting findings (warns but doesn't block)
            auto_tag: Auto-suggest tags if none provided
            auto_classify: Auto-classify finding type if not specified
            auto_summarize_evidence: Auto-summarize evidence if > max_evidence_length

        Returns:
            dict with 'id', 'tags_suggested', 'type_suggested', 'evidence_summarized',
            'cross_refs', 'notations_detected', 'content_warnings', 'contradictions'

        Raises:
            ValueError: If a near-duplicate exists and check_duplicate is True
        """
        result = {
            "id": None,
            "tags_suggested": False,
            "tags_missing_warning": False,
            "type_suggested": False,
            "type_mismatch_warning": None,
            "evidence_summarized": False,
            "cross_refs": None,
            "notations_detected": None,
            "content_warnings": [],
            "contradictions": [],
        }

        # Track if tags were originally missing
        original_tags_missing = not tags

        # Auto-classify finding type if not specified
        if finding_type is None:
            if auto_classify:
                finding_type = self.classify_finding_type(content)
                result["type_suggested"] = True
            else:
                finding_type = "discovery"
        elif auto_classify:
            # Type was provided - still classify and warn if mismatch
            suggested_type = self.classify_finding_type(content)
            if suggested_type != finding_type:
                result["type_mismatch_warning"] = (
                    f"Provided type '{finding_type}' differs from suggested '{suggested_type}'"
                )

        if finding_type not in FINDING_TYPES:
            raise ValueError(f"Invalid type: {finding_type}. Must be one of {FINDING_TYPES}")

        # Warn if failure without evidence (failures should explain WHY it failed)
        result["evidence_missing_warning"] = False
        if finding_type == "failure" and not evidence:
            result["evidence_missing_warning"] = True

        # Warn if no tags provided
        if original_tags_missing:
            result["tags_missing_warning"] = True

        # Auto-suggest tags if none provided
        if not tags and auto_tag:
            suggested = self.suggest_tags(content, project=project)
            if suggested:
                tags = suggested
                result["tags_suggested"] = True

        # Validate content for anti-patterns (warn but don't block)
        result["content_warnings"] = validate_finding_content(content, tags)

        # Auto-summarize long evidence
        original_evidence = evidence
        if evidence and auto_summarize_evidence and len(evidence) > max_evidence_length:
            evidence = self.summarize_evidence(evidence, max_length=max_evidence_length)
            result["evidence_summarized"] = True

        # Check for duplicates (also generates embedding to reuse)
        embedding = None
        if check_duplicate:
            is_dup, existing, embedding = self.check_duplicate(content, original_evidence or evidence, duplicate_threshold)
            if is_dup and existing:
                raise ValueError(
                    f"Similar finding already exists (similarity: {existing['similarity']:.2f}):\n"
                    f"  ID: {existing['id']}\n"
                    f"  Content: {existing['content'][:100]}...\n"
                    f"Use check_duplicate=False to add anyway, or kb_correct to update."
                )

        # Check for contradictions (warns but doesn't block)
        if check_contradictions:
            contradictions = self.check_contradictions(content, project=project)
            if contradictions:
                result["contradictions"] = contradictions

        finding_id = f"kb-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = datetime.now().isoformat()
        # Validate tags to prevent garbage from LLM or bad input
        tags = self._validate_tags(tags)
        tags_json = json.dumps(tags)

        # Generate summary
        summary = self._generate_summary(content, evidence)

        # Generate embedding BEFORE transaction to avoid holding lock on failure
        if embedding is None:
            embedding = self._embed(content + " " + (evidence or ""))

        # Now do the database inserts (transaction starts here)
        try:
            self.conn.execute("""
                INSERT INTO findings (id, type, status, project, sprint, tags, content, summary, evidence, created_at, updated_at)
                VALUES (?, ?, 'current', ?, ?, ?, ?, ?, ?, ?, ?)
            """, (finding_id, finding_type, project, sprint, tags_json, content, summary, evidence, now, now))

            self.conn.execute(
                "INSERT INTO findings_vec (id, embedding) VALUES (?, ?)",
                (finding_id, embedding)
            )

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        result["id"] = finding_id

        # Auto-detect cross-references (non-blocking)
        result["cross_refs"] = self.suggest_cross_references(finding_id, content, project=project)

        # Auto-detect notations (non-blocking)
        result["notations_detected"] = self.detect_notations(content, project=project)

        return result

    def correct(
        self,
        supersedes_id: str,
        content: str,
        reason: Optional[str] = None,
        evidence: Optional[str] = None,
    ) -> dict:
        """Correct an existing finding by superseding it.

        Returns:
            dict with 'id' (new finding ID) and 'impacted_findings' (findings that cite the superseded one)
        """
        # Verify the old finding exists
        old = self.conn.execute(
            "SELECT id, project, sprint, tags FROM findings WHERE id = ?",
            (supersedes_id,)
        ).fetchone()

        if not old:
            raise ValueError(f"Finding not found: {supersedes_id}")

        # Find findings that might be impacted (cite the superseded finding)
        impacted = self.find_citing_findings(supersedes_id)

        # Create correction finding
        finding_id = f"kb-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = datetime.now().isoformat()

        full_content = content
        if reason:
            full_content = f"[CORRECTION: {reason}] {content}"

        # Generate summary for the correction
        summary = self._generate_summary(content, evidence)

        # Generate embedding BEFORE transaction to avoid holding lock on failure
        embedding = self._embed(full_content + " " + (evidence or ""))

        # Now do the database operations (transaction starts here)
        try:
            # Mark old finding as superseded
            self.conn.execute(
                "UPDATE findings SET status = 'superseded', updated_at = ? WHERE id = ?",
                (now, supersedes_id)
            )

            self.conn.execute("""
                INSERT INTO findings (id, type, status, supersedes_id, project, sprint, tags, content, summary, evidence, created_at, updated_at)
                VALUES (?, 'correction', 'current', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (finding_id, supersedes_id, old["project"], old["sprint"], old["tags"], full_content, summary, evidence, now, now))

            self.conn.execute(
                "INSERT INTO findings_vec (id, embedding) VALUES (?, ?)",
                (finding_id, embedding)
            )

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        return {"id": finding_id, "impacted_findings": impacted}

    def bulk_add_tags(self, finding_ids: list[str], tags: list[str]) -> dict:
        """Add tags to multiple findings.

        Args:
            finding_ids: List of finding IDs to update
            tags: Tags to add (merged with existing tags)

        Returns:
            dict with 'updated' count and 'skipped' (not found) count
        """
        # Validate input tags
        tags = self._validate_tags(tags)
        if not tags:
            return {"updated": 0, "skipped": len(finding_ids), "error": "No valid tags provided"}

        updated = 0
        skipped = 0
        now = datetime.now().isoformat()

        for fid in finding_ids:
            row = self.conn.execute(
                "SELECT tags FROM findings WHERE id = ?", (fid,)
            ).fetchone()

            if not row:
                skipped += 1
                continue

            existing = json.loads(row["tags"]) if row["tags"] else []
            merged = list(set(existing + tags))

            self.conn.execute(
                "UPDATE findings SET tags = ?, updated_at = ? WHERE id = ?",
                (json.dumps(merged), now, fid)
            )
            updated += 1

        self.conn.commit()
        return {"updated": updated, "skipped": skipped}

    def consolidate_cluster(
        self,
        finding_ids: list[str],
        summary: str,
        reason: str,
        finding_type: str = "discovery",
        tags: list[str] | None = None,
        evidence: str | None = None,
    ) -> dict:
        """Supersede multiple findings with a single consolidated finding.

        Args:
            finding_ids: List of finding IDs to supersede
            summary: Content of the new consolidated finding
            reason: Why these findings are being merged
            finding_type: Type for the new finding (default: discovery)
            tags: Tags for new finding (if None, merges tags from all superseded findings)
            evidence: Evidence for new finding

        Returns:
            dict with 'new_id', 'superseded_count', 'skipped' (not found) count
        """
        if not finding_ids:
            raise ValueError("No finding IDs provided")

        superseded = 0
        skipped = 0
        merged_tags: set[str] = set()
        project = None
        sprint = None
        now = datetime.now().isoformat()

        # First pass: validate and collect metadata
        for fid in finding_ids:
            row = self.conn.execute(
                "SELECT id, project, sprint, tags, status FROM findings WHERE id = ?",
                (fid,)
            ).fetchone()

            if not row:
                skipped += 1
                continue

            if row["status"] == "superseded":
                skipped += 1  # Already superseded
                continue

            # Collect metadata from first valid finding
            if project is None:
                project = row["project"]
                sprint = row["sprint"]

            if row["tags"]:
                merged_tags.update(json.loads(row["tags"]))

        # Use provided tags or merged tags
        final_tags = tags if tags is not None else list(merged_tags)

        # Second pass: mark as superseded
        new_id = f"kb-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        for fid in finding_ids:
            result = self.conn.execute(
                "UPDATE findings SET status = 'superseded', updated_at = ? WHERE id = ? AND status = 'current'",
                (now, fid)
            )
            if result.rowcount > 0:
                superseded += 1

        if superseded == 0:
            raise ValueError("No valid findings to consolidate (all not found or already superseded)")

        # Create consolidated finding
        full_content = f"[CONSOLIDATION: {reason}] {summary}"

        self.conn.execute("""
            INSERT INTO findings (id, type, status, project, sprint, tags, content, evidence, created_at, updated_at)
            VALUES (?, ?, 'current', ?, ?, ?, ?, ?, ?, ?)
        """, (new_id, finding_type, project, sprint, json.dumps(final_tags), full_content, evidence, now, now))

        # Add embedding
        embedding = self._embed(full_content + " " + (evidence or ""))
        self.conn.execute(
            "INSERT INTO findings_vec (id, embedding) VALUES (?, ?)",
            (new_id, embedding)
        )

        self.conn.commit()
        return {"new_id": new_id, "superseded_count": superseded, "skipped": skipped}

    def search(
        self,
        query: str,
        limit: int = 10,
        include_superseded: bool = False,
        project: str | None = None,
        finding_type: str | None = None,
        tags: list[str] | None = None,
        after: str | None = None,
        before: str | None = None,
        hybrid: bool = True,
        expand: bool = False,
        verbose: bool = False,
        deprioritize_index: bool = True,
        exclude_corrections: bool = True,
        recency_weight: float = 0.1,
    ) -> list[dict]:
        """Search findings using hybrid vector + keyword search.

        Args:
            query: Search query string
            limit: Maximum results to return
            include_superseded: Include superseded findings
            project: Filter by project
            finding_type: Filter by type (success/failure/discovery/experiment)
            tags: Filter by tags (all must match)
            after: Filter to findings after this date (ISO format: YYYY-MM-DD)
            before: Filter to findings before this date (ISO format: YYYY-MM-DD)
            hybrid: Combine vector and keyword search (default True)
            expand: Use LLM to expand query with synonyms/related terms
            verbose: Show expanded query and other debug info
            deprioritize_index: Demote INDEX/entry-point findings in ranking
            exclude_corrections: Exclude [CORRECTION:...] entries
            recency_weight: Weight for recency boost (0-1, higher = more recency bias)
        """
        # Optionally expand query using LLM
        search_query = self.expand_query(query, project=project, verbose=verbose) if expand else query

        vector_results = {}
        fts_results = {}

        # Vector similarity search
        query_embedding = self._embed(search_query)
        sql = """
            SELECT f.*, v.distance
            FROM findings f
            JOIN findings_vec v ON f.id = v.id
            WHERE v.embedding MATCH ?
            AND k = ?
        """
        params = [query_embedding, limit * 3]
        rows = self.conn.execute(sql, params).fetchall()

        for rank, row in enumerate(rows, 1):
            vector_results[row["id"]] = {
                "row": row,
                "rank": rank,
                "similarity": 1 - (row["distance"] ** 2) / 2,
            }

        # Full-text search (for hybrid)
        if hybrid:
            # Escape special FTS5 characters and handle queries
            fts_query = search_query.replace('"', '""')
            try:
                sql = """
                    SELECT f.*, fts.rank
                    FROM findings f
                    JOIN findings_fts fts ON f.rowid = fts.rowid
                    WHERE findings_fts MATCH ?
                    ORDER BY fts.rank
                    LIMIT ?
                """
                fts_rows = self.conn.execute(sql, [fts_query, limit * 3]).fetchall()

                for rank, row in enumerate(fts_rows, 1):
                    fts_results[row["id"]] = {
                        "row": row,
                        "rank": rank,
                        "relevance": -row["rank"],
                    }
            except sqlite3.OperationalError:
                # FTS query failed (e.g., syntax error), skip keyword results
                if verbose:
                    print(f"FTS query failed for: {fts_query}")

        # Merge results using Reciprocal Rank Fusion (RRF)
        # RRF score = sum(1 / (k + rank)) for each retriever
        k = 60  # RRF constant
        all_ids = set(vector_results.keys()) | set(fts_results.keys())
        merged = {}

        for finding_id in all_ids:
            rrf_score = 0
            row = None

            if finding_id in vector_results:
                rrf_score += 1 / (k + vector_results[finding_id]["rank"])
                row = vector_results[finding_id]["row"]

            if finding_id in fts_results:
                rrf_score += 1 / (k + fts_results[finding_id]["rank"])
                if row is None:
                    row = fts_results[finding_id]["row"]

            merged[finding_id] = {
                "row": row,
                "rrf_score": rrf_score,
                "vector_sim": vector_results.get(finding_id, {}).get("similarity", 0),
                "fts_relevance": fts_results.get(finding_id, {}).get("relevance", 0),
            }

        # Convert to result list and apply filters
        results = []
        now = datetime.now()

        for finding_id, data in merged.items():
            row = data["row"]

            # Apply filters
            if not include_superseded and row["status"] == "superseded":
                continue
            if project and row["project"] != project:
                continue
            if finding_type and row["type"] != finding_type:
                continue

            # Tag filtering (all specified tags must match)
            if tags:
                finding_tags = json.loads(row["tags"] or "[]")
                if not all(t in finding_tags for t in tags):
                    continue

            # Date filtering
            if after and row["created_at"] < after:
                continue
            if before and row["created_at"] > before:
                continue

            # Calculate recency boost
            try:
                created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                age_days = (now - created.replace(tzinfo=None)).days
                # Decay factor: e^(-age_days / 180) gives ~0.5 at 6 months
                recency_factor = 1 + recency_weight * (2.718 ** (-age_days / 180))
            except (ValueError, TypeError):
                recency_factor = 1.0

            # Combined score
            base_score = data["rrf_score"] if hybrid else data["vector_sim"]
            final_score = base_score * recency_factor

            results.append({
                "id": row["id"],
                "type": row["type"],
                "status": row["status"],
                "supersedes_id": row["supersedes_id"],
                "project": row["project"],
                "sprint": row["sprint"],
                "tags": json.loads(row["tags"] or "[]"),
                "summary": row["summary"],
                "content": row["content"],  # Kept for filtering/deprioritization logic
                "similarity": data["vector_sim"],
                "score": final_score,
            })

        # Post-process results: filter and re-rank
        if exclude_corrections:
            results = [r for r in results if not r["content"].startswith("[CORRECTION:")]

        if deprioritize_index:
            for r in results:
                content_lower = r["content"].lower()
                result_tags = r.get("tags", [])
                is_index = (
                    content_lower.startswith("index:") or
                    "entry-point" in result_tags or
                    "index" in result_tags
                )
                if is_index:
                    r["score"] *= 0.7  # 30% penalty for INDEX entries

        # Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]

    def related(
        self,
        finding_id: str,
        limit: int = 5,
        include_superseded: bool = False,
    ) -> list[dict]:
        """Find findings related to a given finding by embedding similarity.

        Args:
            finding_id: ID of the finding to find related content for
            limit: Maximum number of related findings to return
            include_superseded: Include superseded findings

        Returns:
            List of related findings sorted by similarity
        """
        # Get the finding's embedding
        row = self.conn.execute(
            "SELECT embedding FROM findings_vec WHERE id = ?",
            [finding_id]
        ).fetchone()

        if not row:
            return []

        # Search for similar embeddings
        sql = """
            SELECT f.*, v.distance
            FROM findings f
            JOIN findings_vec v ON f.id = v.id
            WHERE v.embedding MATCH ?
            AND k = ?
            AND f.id != ?
        """
        params = [row["embedding"], limit + 10, finding_id]
        rows = self.conn.execute(sql, params).fetchall()

        results = []
        for r in rows:
            if not include_superseded and r["status"] == "superseded":
                continue

            results.append({
                "id": r["id"],
                "type": r["type"],
                "status": r["status"],
                "project": r["project"],
                "tags": json.loads(r["tags"] or "[]"),
                "content": r["content"],
                "created_at": r["created_at"],
                "similarity": 1 - (r["distance"] ** 2) / 2,
            })

            if len(results) >= limit:
                break

        return results

    def ask(
        self,
        question: str,
        project: str | None = None,
        limit: int = 10,
        verbose: bool = False,
    ) -> dict:
        """Answer a natural language question using KB findings.

        Searches for relevant findings and uses LLM to synthesize an answer.

        Args:
            question: Natural language question
            project: Filter to specific project
            limit: Max findings to consider
            verbose: Include search results in response

        Returns:
            dict with 'answer', 'sources', and optionally 'search_results'
        """
        # Search for relevant findings
        results = self.search(
            query=question,
            project=project,
            limit=limit,
            expand=True,  # Use LLM query expansion for better recall
            deprioritize_index=True,
            exclude_corrections=True,
        )

        if not results:
            return {
                "answer": "No relevant findings found in the knowledge base.",
                "sources": [],
                "search_results": [] if verbose else None,
            }

        # Format findings for context
        context_parts = []
        sources = []
        for i, r in enumerate(results, 1):
            sim = r.get("similarity", r.get("relevance", 0))
            finding_text = f"[{i}] ({r['type']}, {r['project'] or 'no project'}, sim={sim:.2f})\n{r['content']}"
            if r.get("evidence"):
                finding_text += f"\nEvidence: {r['evidence'][:200]}"
            context_parts.append(finding_text)
            sources.append({
                "id": r["id"],
                "type": r["type"],
                "project": r["project"],
                "similarity": sim,
                "content": r["content"][:100] + "..." if len(r["content"]) > 100 else r["content"],
            })

        context = "\n\n".join(context_parts)

        # Generate answer using LLM
        system_prompt = """You are a knowledge base assistant. Answer questions based ONLY on the provided findings.
- Cite findings by their number [1], [2], etc.
- If findings conflict, explain the discrepancy
- If findings don't fully answer the question, say what's missing
- Be concise but thorough"""

        prompt = f"""QUESTION: {question}

RELEVANT FINDINGS:
{context}

Answer the question based on these findings. Cite sources by number."""

        answer = self._llm_complete(
            prompt,
            max_tokens=500,
            temperature=0.3,
            system_prompt=system_prompt,
            timeout=60,
        )

        if answer:
            # Extract from JSON if LLM returned JSON-wrapped response
            answer = self._extract_text_from_json(answer, keys=["answer", "response", "text"])

        if not answer:
            # Fallback: just return the top findings
            answer = "LLM unavailable. Top findings:\n\n" + "\n\n".join(
                f"- {r['content'][:200]}" for r in results[:3]
            )

        result = {
            "answer": answer,
            "sources": sources,
        }
        if verbose:
            result["search_results"] = results

        return result

    def get(self, finding_id: str) -> Optional[dict]:
        """Get a finding by ID."""
        row = self.conn.execute(
            "SELECT * FROM findings WHERE id = ?",
            (finding_id,)
        ).fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "type": row["type"],
            "status": row["status"],
            "supersedes_id": row["supersedes_id"],
            "project": row["project"],
            "sprint": row["sprint"],
            "tags": json.loads(row["tags"] or "[]"),
            "content": row["content"],
            "summary": row["summary"],
            "evidence": row["evidence"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_supersession_chain(self, finding_id: str) -> list[dict]:
        """Get the chain of findings that supersede each other."""
        chain = []
        current_id = finding_id

        while current_id:
            finding = self.get(current_id)
            if not finding:
                break
            chain.append(finding)

            # Find what supersedes this finding
            row = self.conn.execute(
                "SELECT id FROM findings WHERE supersedes_id = ?",
                (current_id,)
            ).fetchone()
            current_id = row["id"] if row else None

        return chain

    def list_findings(
        self,
        project: Optional[str] = None,
        sprint: Optional[str] = None,
        finding_type: Optional[str] = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        """List findings with optional filters."""
        sql = "SELECT * FROM findings WHERE 1=1"
        params = []

        if not include_superseded:
            sql += " AND status = 'current'"
        if project:
            sql += " AND project = ?"
            params.append(project)
        if sprint:
            sql += " AND sprint = ?"
            params.append(sprint)
        if finding_type:
            sql += " AND type = ?"
            params.append(finding_type)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "type": row["type"],
                "status": row["status"],
                "supersedes_id": row["supersedes_id"],
                "project": row["project"],
                "sprint": row["sprint"],
                "tags": json.loads(row["tags"] or "[]"),
                "summary": row["summary"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete(self, finding_id: str) -> bool:
        """Delete a finding (and its embedding)."""
        # Check if any finding supersedes this one
        superseding = self.conn.execute(
            "SELECT id FROM findings WHERE supersedes_id = ?",
            (finding_id,)
        ).fetchone()

        if superseding:
            raise ValueError(
                f"Cannot delete {finding_id}: it is superseded by {superseding['id']}. "
                "Delete the superseding finding first."
            )

        # Delete from vector table
        self.conn.execute("DELETE FROM findings_vec WHERE id = ?", (finding_id,))

        # Delete from main table (triggers handle FTS)
        cursor = self.conn.execute("DELETE FROM findings WHERE id = ?", (finding_id,))

        self.conn.commit()
        return cursor.rowcount > 0

    def stats(self) -> dict:
        """Get database statistics."""
        total = self.conn.execute("SELECT COUNT(*) as c FROM findings").fetchone()["c"]
        current = self.conn.execute("SELECT COUNT(*) as c FROM findings WHERE status = 'current'").fetchone()["c"]
        superseded = self.conn.execute("SELECT COUNT(*) as c FROM findings WHERE status = 'superseded'").fetchone()["c"]

        by_type = {}
        for row in self.conn.execute("SELECT type, COUNT(*) as c FROM findings WHERE status = 'current' GROUP BY type"):
            by_type[row["type"]] = row["c"]

        by_project = {}
        for row in self.conn.execute("SELECT project, COUNT(*) as c FROM findings WHERE status = 'current' AND project IS NOT NULL GROUP BY project"):
            by_project[row["project"]] = row["c"]

        notations = self.conn.execute("SELECT COUNT(*) as c FROM notations").fetchone()["c"]

        return {
            "total": total,
            "current": current,
            "superseded": superseded,
            "by_type": by_type,
            "by_project": by_project,
            "notations": notations,
            "db_path": str(self.db_path),
        }

    def get_all_tags(self) -> list[str]:
        """Get all unique tags from current findings."""
        tags = set()
        for row in self.conn.execute(
            "SELECT DISTINCT tags FROM findings WHERE status = 'current' AND tags IS NOT NULL"
        ):
            try:
                tags.update(json.loads(row[0]))
            except (json.JSONDecodeError, TypeError):
                pass
        return sorted(tags)

    def get_latest_update(self) -> tuple[int, str]:
        """Get count and latest timestamp for change detection."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt, MAX(updated_at) as latest FROM findings"
        ).fetchone()
        return (row["cnt"] or 0, row["latest"] or "")

    def reembed_all(self) -> dict:
        """Re-generate embeddings for all findings.

        Use this after fixing the embedding model or algorithm.
        Returns stats on what was re-embedded.
        """
        findings = self.conn.execute(
            "SELECT id, content, evidence FROM findings"
        ).fetchall()

        updated = 0
        failed = 0

        for row in findings:
            try:
                text = row["content"] + " " + (row["evidence"] or "")
                embedding = self._embed(text)
                self.conn.execute(
                    "UPDATE findings_vec SET embedding = ? WHERE id = ?",
                    (embedding, row["id"])
                )
                updated += 1
            except Exception as e:
                print(f"Failed to re-embed {row['id']}: {e}", file=sys.stderr)
                failed += 1

        self.conn.commit()
        return {"updated": updated, "failed": failed, "total": len(findings)}

    def backfill_summaries(
        self, project: str | None = None, batch_size: int = 20
    ) -> dict:
        """Generate summaries for findings that don't have one.

        Args:
            project: Optional project filter
            batch_size: How many to process in one batch

        Returns:
            Dict with updated/failed/total counts
        """
        query = "SELECT id, content, evidence FROM findings WHERE summary IS NULL"
        params = []
        if project:
            query += " AND project = ?"
            params.append(project)
        query += f" LIMIT {batch_size}"

        findings = self.conn.execute(query, params).fetchall()

        updated = 0
        failed = 0

        for row in findings:
            try:
                summary = self._generate_summary(row["content"], row["evidence"])
                if summary:
                    self.conn.execute(
                        "UPDATE findings SET summary = ? WHERE id = ?",
                        (summary, row["id"])
                    )
                    updated += 1
                    print(f"  {row['id']}: {summary}")
                else:
                    failed += 1
            except Exception as e:
                print(f"Failed to generate summary for {row['id']}: {e}", file=sys.stderr)
                failed += 1

        self.conn.commit()

        # Count remaining
        remaining_query = "SELECT COUNT(*) FROM findings WHERE summary IS NULL"
        remaining_params = []
        if project:
            remaining_query += " AND project = ?"
            remaining_params.append(project)
        remaining = self.conn.execute(remaining_query, remaining_params).fetchone()[0]

        return {
            "updated": updated,
            "failed": failed,
            "processed": len(findings),
            "remaining": remaining,
        }

    # =========================================================================
    # Notation Tracking Methods
    # =========================================================================

    def notation_add(
        self,
        symbol: str,
        meaning: str,
        project: str | None = None,
        domain: str = "general",
    ) -> str:
        """Add a new notation to track.

        Args:
            symbol: The notation symbol (e.g., "H_D", "\\slashed{k}")
            meaning: What the notation represents
            project: Project this notation belongs to
            domain: Domain type (physics, math, cs, general)
        """
        if domain not in NOTATION_DOMAINS:
            raise ValueError(f"Invalid domain: {domain}. Must be one of {NOTATION_DOMAINS}")

        # Check for existing notation with same symbol in project
        existing = self.conn.execute(
            "SELECT id FROM notations WHERE current_symbol = ? AND (project = ? OR (project IS NULL AND ? IS NULL))",
            (symbol, project, project)
        ).fetchone()

        if existing:
            raise ValueError(f"Notation '{symbol}' already exists for project '{project}'. Use notation_update to change it.")

        notation_id = f"notation-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = datetime.now().isoformat()

        self.conn.execute("""
            INSERT INTO notations (id, current_symbol, meaning, project, domain, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (notation_id, symbol, meaning, project, domain, now, now))

        self.conn.commit()
        return notation_id

    def notation_update(
        self,
        new_symbol: str,
        old_symbol: str | None = None,
        notation_id: str | None = None,
        meaning: str | None = None,
        reason: str | None = None,
        project: str | None = None,
    ) -> str:
        """Update a notation symbol, recording the change in history.

        Must provide either old_symbol or notation_id to identify the notation.

        Args:
            new_symbol: The new symbol to use
            old_symbol: The current symbol to replace (will search by symbol)
            notation_id: The ID of the notation to update (alternative to old_symbol)
            meaning: Optional new meaning (keeps old if not provided)
            reason: Reason for the change
            project: Project filter when searching by old_symbol
        """
        # Find the notation
        if notation_id:
            row = self.conn.execute(
                "SELECT * FROM notations WHERE id = ?",
                (notation_id,)
            ).fetchone()
        elif old_symbol:
            sql = "SELECT * FROM notations WHERE current_symbol = ?"
            params = [old_symbol]
            if project:
                sql += " AND project = ?"
                params.append(project)
            row = self.conn.execute(sql, params).fetchone()
        else:
            raise ValueError("Must provide either old_symbol or notation_id")

        if not row:
            raise ValueError(f"Notation not found: {old_symbol or notation_id}")

        found_id: str = row["id"]
        old_symbol_actual = row["current_symbol"]
        now = datetime.now().isoformat()

        # Record history
        history_id = f"notation-hist-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.conn.execute("""
            INSERT INTO notation_history (id, notation_id, old_symbol, new_symbol, reason, changed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (history_id, found_id, old_symbol_actual, new_symbol, reason, now))

        # Update notation
        if meaning:
            self.conn.execute("""
                UPDATE notations SET current_symbol = ?, meaning = ?, updated_at = ? WHERE id = ?
            """, (new_symbol, meaning, now, found_id))
        else:
            self.conn.execute("""
                UPDATE notations SET current_symbol = ?, updated_at = ? WHERE id = ?
            """, (new_symbol, now, found_id))

        self.conn.commit()
        return found_id

    def notation_get(self, notation_id: str) -> dict | None:
        """Get a notation by ID, including its history."""
        row = self.conn.execute(
            "SELECT * FROM notations WHERE id = ?",
            (notation_id,)
        ).fetchone()

        if not row:
            return None

        history = self.conn.execute(
            "SELECT * FROM notation_history WHERE notation_id = ? ORDER BY changed_at DESC",
            (notation_id,)
        ).fetchall()

        return {
            "id": row["id"],
            "current_symbol": row["current_symbol"],
            "meaning": row["meaning"],
            "project": row["project"],
            "domain": row["domain"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "history": [
                {
                    "old_symbol": h["old_symbol"],
                    "new_symbol": h["new_symbol"],
                    "reason": h["reason"],
                    "changed_at": h["changed_at"],
                }
                for h in history
            ],
        }

    # Greek letter mappings (lowercase and uppercase)
    GREEK_LETTERS = {
        "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
        "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
        "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
        "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π",
        "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
        "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
        "Alpha": "Α", "Beta": "Β", "Gamma": "Γ", "Delta": "Δ",
        "Epsilon": "Ε", "Zeta": "Ζ", "Eta": "Η", "Theta": "Θ",
        "Iota": "Ι", "Kappa": "Κ", "Lambda": "Λ", "Mu": "Μ",
        "Nu": "Ν", "Xi": "Ξ", "Omicron": "Ο", "Pi": "Π",
        "Rho": "Ρ", "Sigma": "Σ", "Tau": "Τ", "Upsilon": "Υ",
        "Phi": "Φ", "Chi": "Χ", "Psi": "Ψ", "Omega": "Ω",
    }
    GREEK_TO_LATIN = {v: k for k, v in GREEK_LETTERS.items()}

    def _expand_greek(self, query: str) -> list[str]:
        """Expand query to include Greek letter variants."""
        variants = [query]

        # Latin name -> Greek letter (word boundary match)
        for latin, greek in self.GREEK_LETTERS.items():
            pattern = rf'\b{latin}\b'
            if re.search(pattern, query, re.IGNORECASE):
                variants.append(re.sub(pattern, greek, query, flags=re.IGNORECASE))

        # Greek letter -> Latin name
        for greek, latin in self.GREEK_TO_LATIN.items():
            if greek in query:
                variants.append(query.replace(greek, latin))

        return list(set(variants))

    def notation_search(
        self,
        query: str,
        project: str | None = None,
        domain: str | None = None,
    ) -> list[dict]:
        """Search notations by symbol or meaning.

        Automatically expands Latin names to Greek letters and vice versa.
        E.g., searching "lambda" also finds "λ", and "Γ" finds "Gamma".
        """
        # Expand query to include Greek variants
        query_variants = self._expand_greek(query)

        # Build OR conditions for all variants
        conditions = []
        params: list = []
        for variant in query_variants:
            escaped = variant.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            conditions.append("(current_symbol LIKE ? ESCAPE '\\' OR meaning LIKE ? ESCAPE '\\')")
            params.extend([pattern, pattern])

        sql = f"SELECT * FROM notations WHERE ({' OR '.join(conditions)})"

        if project:
            sql += " AND project = ?"
            params.append(project)
        if domain:
            sql += " AND domain = ?"
            params.append(domain)

        sql += " ORDER BY updated_at DESC"

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "current_symbol": row["current_symbol"],
                "meaning": row["meaning"],
                "project": row["project"],
                "domain": row["domain"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def notation_list(
        self,
        project: str | None = None,
        domain: str | None = None,
    ) -> list[dict]:
        """List all notations with optional filters."""
        sql = "SELECT * FROM notations WHERE 1=1"
        params = []

        if project:
            sql += " AND project = ?"
            params.append(project)
        if domain:
            sql += " AND domain = ?"
            params.append(domain)

        sql += " ORDER BY current_symbol"

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "current_symbol": row["current_symbol"],
                "meaning": row["meaning"],
                "project": row["project"],
                "domain": row["domain"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def notation_history(self, notation_id: str) -> list[dict]:
        """Get the change history for a notation."""
        rows = self.conn.execute(
            "SELECT * FROM notation_history WHERE notation_id = ? ORDER BY changed_at DESC",
            (notation_id,)
        ).fetchall()

        return [
            {
                "id": row["id"],
                "old_symbol": row["old_symbol"],
                "new_symbol": row["new_symbol"],
                "reason": row["reason"],
                "changed_at": row["changed_at"],
            }
            for row in rows
        ]

    def notation_delete(self, notation_id: str) -> bool:
        """Delete a notation and its history."""
        self.conn.execute("DELETE FROM notation_history WHERE notation_id = ?", (notation_id,))
        cursor = self.conn.execute("DELETE FROM notations WHERE id = ?", (notation_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # =========================================================================
    # Error Tracking Methods
    # =========================================================================

    def error_add(
        self,
        signature: str,
        error_type: str | None = None,
        project: str | None = None,
        auto_normalize: bool = True,
    ) -> dict:
        """Record an error signature.

        If the error already exists, increments occurrence_count and updates last_seen.

        Args:
            signature: Raw error text
            error_type: Error category
            project: Project name
            auto_normalize: If True, normalize signature using LLM

        Returns:
            dict with 'id', 'normalized', 'is_new', 'occurrence_count'
        """
        result = {
            "id": None,
            "normalized": False,
            "original_signature": signature,
            "is_new": True,
            "occurrence_count": 1,
        }

        # Auto-normalize the signature
        if auto_normalize:
            normalized = self.normalize_error_signature(signature)
            if normalized and normalized != signature:
                signature = normalized
                result["normalized"] = True

        now = datetime.now().isoformat()

        # Check if error already exists
        existing = self.conn.execute(
            "SELECT id, occurrence_count FROM errors WHERE signature = ? AND project IS ?",
            (signature, project)
        ).fetchone()

        if existing:
            self.conn.execute(
                "UPDATE errors SET last_seen = ?, occurrence_count = ? WHERE id = ?",
                (now, existing["occurrence_count"] + 1, existing["id"])
            )
            self.conn.commit()
            result["id"] = existing["id"]
            result["is_new"] = False
            result["occurrence_count"] = existing["occurrence_count"] + 1
            return result

        # Create new error
        error_id = f"err-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
        self.conn.execute(
            """INSERT INTO errors (id, signature, error_type, project, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (error_id, signature, error_type, project, now, now)
        )
        self.conn.commit()
        result["id"] = error_id
        return result

    def error_link(
        self,
        error_id: str,
        finding_id: str,
        verified: bool = False,
    ) -> bool:
        """Link an error to a solution (finding).

        Returns True if link was created, False if it already exists.
        """
        now = datetime.now().isoformat()

        try:
            self.conn.execute(
                """INSERT INTO error_solutions (error_id, finding_id, linked_at, verified)
                   VALUES (?, ?, ?, ?)""",
                (error_id, finding_id, now, 1 if verified else 0)
            )
            self.conn.commit()
            return True
        except Exception:
            return False

    def error_verify(self, error_id: str, finding_id: str) -> bool:
        """Mark a solution as verified for an error."""
        cursor = self.conn.execute(
            "UPDATE error_solutions SET verified = 1 WHERE error_id = ? AND finding_id = ?",
            (error_id, finding_id)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def error_get(self, error_id: str) -> dict | None:
        """Get an error by ID with its linked solutions."""
        row = self.conn.execute(
            "SELECT * FROM errors WHERE id = ?", (error_id,)
        ).fetchone()

        if not row:
            return None

        # Get linked solutions
        solutions = self.conn.execute(
            """SELECT es.finding_id, es.linked_at, es.verified, f.content, f.type
               FROM error_solutions es
               JOIN findings f ON es.finding_id = f.id
               WHERE es.error_id = ?
               ORDER BY es.verified DESC, es.linked_at DESC""",
            (error_id,)
        ).fetchall()

        return {
            "id": row["id"],
            "signature": row["signature"],
            "error_type": row["error_type"],
            "project": row["project"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "occurrence_count": row["occurrence_count"],
            "solutions": [
                {
                    "finding_id": s["finding_id"],
                    "content": s["content"],
                    "type": s["type"],
                    "verified": bool(s["verified"]),
                    "linked_at": s["linked_at"],
                }
                for s in solutions
            ],
        }

    def error_search(
        self,
        query: str,
        project: str | None = None,
    ) -> list[dict]:
        """Search errors by signature pattern."""
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"

        sql = "SELECT * FROM errors WHERE signature LIKE ? ESCAPE '\\'"
        params = [pattern]

        if project:
            sql += " AND project = ?"
            params.append(project)

        sql += " ORDER BY last_seen DESC"

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "signature": row["signature"],
                "error_type": row["error_type"],
                "project": row["project"],
                "occurrence_count": row["occurrence_count"],
                "last_seen": row["last_seen"],
            }
            for row in rows
        ]

    def error_list(
        self,
        project: str | None = None,
        error_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """List errors with optional filters."""
        sql = "SELECT * FROM errors WHERE 1=1"
        params = []

        if project:
            sql += " AND project = ?"
            params.append(project)
        if error_type:
            sql += " AND error_type = ?"
            params.append(error_type)

        sql += " ORDER BY last_seen DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "signature": row["signature"],
                "error_type": row["error_type"],
                "project": row["project"],
                "occurrence_count": row["occurrence_count"],
                "last_seen": row["last_seen"],
            }
            for row in rows
        ]

    def error_solutions(self, error_id: str) -> list[dict]:
        """Get all solutions linked to an error."""
        rows = self.conn.execute(
            """SELECT es.*, f.content, f.type, f.evidence
               FROM error_solutions es
               JOIN findings f ON es.finding_id = f.id
               WHERE es.error_id = ?
               ORDER BY es.verified DESC, es.linked_at DESC""",
            (error_id,)
        ).fetchall()

        return [
            {
                "finding_id": row["finding_id"],
                "content": row["content"],
                "type": row["type"],
                "evidence": row["evidence"],
                "verified": bool(row["verified"]),
                "linked_at": row["linked_at"],
            }
            for row in rows
        ]

    def solution_errors(self, finding_id: str) -> list[dict]:
        """Get all errors that a solution (finding) fixes."""
        rows = self.conn.execute(
            """SELECT e.*, es.verified, es.linked_at
               FROM errors e
               JOIN error_solutions es ON e.id = es.error_id
               WHERE es.finding_id = ?
               ORDER BY es.linked_at DESC""",
            (finding_id,)
        ).fetchall()

        return [
            {
                "id": row["id"],
                "signature": row["signature"],
                "error_type": row["error_type"],
                "project": row["project"],
                "verified": bool(row["verified"]),
                "linked_at": row["linked_at"],
            }
            for row in rows
        ]

    def error_delete(self, error_id: str) -> bool:
        """Delete an error and its solution links."""
        self.conn.execute("DELETE FROM error_solutions WHERE error_id = ?", (error_id,))
        cursor = self.conn.execute("DELETE FROM errors WHERE id = ?", (error_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # =========================================================================
    # SCRIPT REGISTRY
    # =========================================================================

    VALID_SCRIPT_LANGUAGES = ("python", "sage", "bash", "other")
    VALID_SCRIPT_RELATIONSHIPS = ("generated_by", "validates", "contradicts")

    def script_add(
        self,
        path: str,
        purpose: str,
        project: str | None = None,
        language: str | None = None,
        store_content: bool = True,
        max_content_size: int = 100000,
    ) -> str:
        """Register a script in the knowledge base.

        Args:
            path: Path to the script file
            purpose: What hypothesis/question this script tests
            project: Project name
            language: Script language (auto-detected if not specified)
            store_content: Whether to store full script content
            max_content_size: Max content size to store (bytes)

        Returns:
            Script ID
        """
        import hashlib
        from pathlib import Path as P

        file_path = P(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Script not found: {path}")

        content = file_path.read_text()
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check for existing script with same hash
        existing = self.conn.execute(
            "SELECT id FROM scripts WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if existing:
            return existing[0]

        # Auto-detect language
        if language is None:
            suffix = file_path.suffix.lower()
            if suffix == ".py":
                language = "python"
            elif suffix == ".sage":
                language = "sage"
            elif suffix in (".sh", ".bash"):
                language = "bash"
            else:
                language = "other"

        # Only store content if within size limit
        stored_content = content if store_content and len(content) <= max_content_size else None

        script_id = f"script-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
        now = datetime.now().isoformat()

        self.conn.execute(
            """INSERT INTO scripts (id, path, filename, content_hash, content, purpose, project, language, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (script_id, str(file_path), file_path.name, content_hash, stored_content, purpose, project, language, now, now),
        )

        # Create embedding for purpose (for semantic search)
        embedding = self._embed(f"{file_path.name}: {purpose}")
        self.conn.execute(
            "INSERT INTO scripts_vec (id, embedding) VALUES (?, ?)",
            (script_id, embedding),
        )

        self.conn.commit()
        return script_id

    def script_get(self, script_id: str) -> dict | None:
        """Get a script by ID."""
        row = self.conn.execute(
            "SELECT * FROM scripts WHERE id = ?", (script_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "path": row[1],
            "filename": row[2],
            "content_hash": row[3],
            "content": row[4],
            "purpose": row[5],
            "project": row[6],
            "language": row[7],
            "created_at": row[8],
            "updated_at": row[9],
        }

    def script_search(
        self,
        query: str,
        project: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search scripts by purpose using semantic similarity."""
        embedding = self._embed(query)

        sql = """
            SELECT s.*, v.distance
            FROM scripts s
            JOIN scripts_vec v ON s.id = v.id
            WHERE v.embedding MATCH ?
              AND k = ?
        """
        params = [embedding, limit * 2]

        if project:
            sql = sql.replace("WHERE", "WHERE s.project = ? AND")
            params = [project] + params

        sql += " ORDER BY v.distance"

        results = []
        for row in self.conn.execute(sql, params).fetchall():
            script = {
                "id": row[0],
                "path": row[1],
                "filename": row[2],
                "content_hash": row[3],
                "purpose": row[5],
                "project": row[6],
                "language": row[7],
                "similarity": 1.0 - row[10] / 2.0,  # Convert distance to similarity
            }
            if project is None or script.get("project") == project:
                results.append(script)
                if len(results) >= limit:
                    break

        return results

    def script_list(
        self,
        project: str | None = None,
        language: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List registered scripts."""
        sql = "SELECT * FROM scripts WHERE 1=1"
        params = []

        if project:
            sql += " AND project = ?"
            params.append(project)
        if language:
            sql += " AND language = ?"
            params.append(language)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        results = []
        for row in self.conn.execute(sql, params).fetchall():
            results.append({
                "id": row[0],
                "path": row[1],
                "filename": row[2],
                "purpose": row[5],
                "project": row[6],
                "language": row[7],
                "created_at": row[8],
            })
        return results

    def script_link_finding(
        self,
        finding_id: str,
        script_id: str,
        relationship: str = "generated_by",
    ) -> None:
        """Link a finding to a script that generated/validated it."""
        if relationship not in self.VALID_SCRIPT_RELATIONSHIPS:
            raise ValueError(f"Invalid relationship: {relationship}. Must be one of {self.VALID_SCRIPT_RELATIONSHIPS}")

        # Verify both exist
        if not self.get(finding_id):
            raise ValueError(f"Finding not found: {finding_id}")
        if not self.script_get(script_id):
            raise ValueError(f"Script not found: {script_id}")

        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO finding_scripts (finding_id, script_id, relationship, linked_at)
               VALUES (?, ?, ?, ?)""",
            (finding_id, script_id, relationship, now),
        )
        self.conn.commit()

    def script_findings(self, script_id: str) -> list[dict]:
        """Get findings generated by a script."""
        rows = self.conn.execute(
            """SELECT f.*, fs.relationship
               FROM findings f
               JOIN finding_scripts fs ON f.id = fs.finding_id
               WHERE fs.script_id = ?
               ORDER BY f.created_at DESC""",
            (script_id,),
        ).fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "type": row[1],
                "content": row[7],
                "relationship": row[11],
            })
        return results

    def finding_scripts(self, finding_id: str) -> list[dict]:
        """Get scripts that generated a finding."""
        rows = self.conn.execute(
            """SELECT s.*, fs.relationship
               FROM scripts s
               JOIN finding_scripts fs ON s.id = fs.script_id
               WHERE fs.finding_id = ?""",
            (finding_id,),
        ).fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "filename": row[2],
                "purpose": row[5],
                "project": row[6],
                "language": row[7],
                "relationship": row[10],
            })
        return results

    def script_delete(self, script_id: str) -> bool:
        """Delete a script and its links."""
        self.conn.execute("DELETE FROM finding_scripts WHERE script_id = ?", (script_id,))
        self.conn.execute("DELETE FROM scripts_vec WHERE id = ?", (script_id,))
        cursor = self.conn.execute("DELETE FROM scripts WHERE id = ?", (script_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # =========================================================================
    # DOCUMENT TRACKING
    # =========================================================================

    VALID_DOC_TYPES = ("spec", "paper", "standard", "internal", "reference")
    VALID_CITATION_TYPES = ("references", "implements", "contradicts", "extends")

    def doc_add(
        self,
        title: str,
        doc_type: str,
        url: str | None = None,
        project: str | None = None,
        summary: str | None = None,
    ) -> str:
        """Add an authoritative document.

        doc_type: spec, paper, standard, internal, reference
        """
        if doc_type not in self.VALID_DOC_TYPES:
            raise ValueError(f"Invalid doc_type: {doc_type}. Must be one of {self.VALID_DOC_TYPES}")
        now = datetime.now().isoformat()
        doc_id = f"doc-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"

        self.conn.execute(
            """INSERT INTO documents (id, title, url, doc_type, project, summary, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'active', ?)""",
            (doc_id, title, url, doc_type, project, summary, now)
        )
        self.conn.commit()
        return doc_id

    def doc_get(self, doc_id: str) -> dict | None:
        """Get a document by ID with citation count."""
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()

        if not row:
            return None

        # Count citations
        citation_count = self.conn.execute(
            "SELECT COUNT(*) FROM document_citations WHERE document_id = ?",
            (doc_id,)
        ).fetchone()[0]

        return {
            "id": row["id"],
            "title": row["title"],
            "url": row["url"],
            "doc_type": row["doc_type"],
            "project": row["project"],
            "status": row["status"],
            "summary": row["summary"],
            "created_at": row["created_at"],
            "superseded_by": row["superseded_by"],
            "citation_count": citation_count,
        }

    def doc_list(
        self,
        project: str | None = None,
        doc_type: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        """List documents with optional filters."""
        sql = "SELECT * FROM documents WHERE 1=1"
        params: list = []

        if project:
            sql += " AND project = ?"
            params.append(project)
        if doc_type:
            sql += " AND doc_type = ?"
            params.append(doc_type)
        if not include_superseded:
            sql += " AND status = 'active'"

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "url": row["url"],
                "doc_type": row["doc_type"],
                "project": row["project"],
                "status": row["status"],
            }
            for row in rows
        ]

    def doc_search(self, query: str, project: str | None = None) -> list[dict]:
        """Search documents by title or summary."""
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"

        sql = "SELECT * FROM documents WHERE (title LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\')"
        params: list = [pattern, pattern]

        if project:
            sql += " AND project = ?"
            params.append(project)

        sql += " AND status = 'active' ORDER BY created_at DESC"

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "doc_type": row["doc_type"],
                "project": row["project"],
                "summary": row["summary"],
            }
            for row in rows
        ]

    def doc_supersede(self, doc_id: str, new_doc_id: str) -> bool:
        """Mark a document as superseded by another."""
        if not self.doc_get(new_doc_id):
            raise ValueError(f"Target document {new_doc_id} does not exist")

        cursor = self.conn.execute(
            "UPDATE documents SET status = 'superseded', superseded_by = ? WHERE id = ?",
            (new_doc_id, doc_id)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def doc_cite(
        self,
        finding_id: str,
        doc_id: str,
        citation_type: str = "references",
        notes: str | None = None,
    ) -> bool:
        """Link a finding to a document it cites.

        citation_type: references, implements, contradicts, extends
        """
        if citation_type not in self.VALID_CITATION_TYPES:
            raise ValueError(f"Invalid citation_type: {citation_type}. Must be one of {self.VALID_CITATION_TYPES}")

        if not self.get(finding_id):
            raise ValueError(f"Finding {finding_id} does not exist")

        if not self.doc_get(doc_id):
            raise ValueError(f"Document {doc_id} does not exist")

        existing = self.conn.execute(
            "SELECT 1 FROM document_citations WHERE finding_id = ? AND document_id = ?",
            (finding_id, doc_id)
        ).fetchone()
        if existing:
            return False

        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT INTO document_citations (finding_id, document_id, citation_type, notes, cited_at)
               VALUES (?, ?, ?, ?, ?)""",
            (finding_id, doc_id, citation_type, notes, now)
        )
        self.conn.commit()
        return True

    def doc_citations(self, doc_id: str) -> list[dict]:
        """Get all findings that cite a document."""
        rows = self.conn.execute(
            """SELECT dc.*, f.content, f.type
               FROM document_citations dc
               JOIN findings f ON dc.finding_id = f.id
               WHERE dc.document_id = ?
               ORDER BY dc.cited_at DESC""",
            (doc_id,)
        ).fetchall()

        return [
            {
                "finding_id": row["finding_id"],
                "content": row["content"],
                "type": row["type"],
                "citation_type": row["citation_type"],
                "notes": row["notes"],
                "cited_at": row["cited_at"],
            }
            for row in rows
        ]

    def finding_docs(self, finding_id: str) -> list[dict]:
        """Get all documents cited by a finding."""
        rows = self.conn.execute(
            """SELECT dc.*, d.title, d.doc_type, d.url
               FROM document_citations dc
               JOIN documents d ON dc.document_id = d.id
               WHERE dc.finding_id = ?
               ORDER BY dc.cited_at DESC""",
            (finding_id,)
        ).fetchall()

        return [
            {
                "document_id": row["document_id"],
                "title": row["title"],
                "doc_type": row["doc_type"],
                "url": row["url"],
                "citation_type": row["citation_type"],
                "notes": row["notes"],
            }
            for row in rows
        ]

    def doc_delete(self, doc_id: str) -> bool:
        """Delete a document and its citations.

        Raises ValueError if other documents reference this one via superseded_by.
        """
        referring = self.conn.execute(
            "SELECT id, title FROM documents WHERE superseded_by = ?", (doc_id,)
        ).fetchall()
        if referring:
            refs = ", ".join(f"{r['id']} ({r['title']})" for r in referring)
            raise ValueError(f"Cannot delete: other documents reference this one via superseded_by: {refs}")

        self.conn.execute("DELETE FROM document_citations WHERE document_id = ?", (doc_id,))
        cursor = self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # =========================================================================
    # WORKFLOW & ANALYSIS TOOLS
    # =========================================================================

    # Finding templates for common types
    TEMPLATES = {
        "computation_result": {
            "format": "Computed {claim} using {method}. Result: {result}",
            "required": ["claim", "method", "result"],
            "optional": ["script"],
            "default_type": "success",
        },
        "failed_approach": {
            "format": "Attempted {approach} for {goal}. Failed because: {reason}",
            "required": ["approach", "goal", "reason"],
            "optional": ["error"],
            "default_type": "failure",
        },
        "structural_discovery": {
            "format": "{structure} has {property}. This implies {implication}",
            "required": ["structure", "property", "implication"],
            "optional": ["proof_sketch"],
            "default_type": "discovery",
        },
        "verification": {
            "format": "Verified {claim} by {method}. {outcome}",
            "required": ["claim", "method", "outcome"],
            "optional": ["script", "tolerance"],
            "default_type": "success",
        },
        "hypothesis": {
            "format": "Hypothesis: {hypothesis}. Motivation: {motivation}. Status: {status}",
            "required": ["hypothesis", "motivation", "status"],
            "optional": ["tests_needed"],
            "default_type": "experiment",
        },
    }

    def add_from_template(
        self,
        template_name: str,
        project: str | None = None,
        tags: list[str] | None = None,
        **kwargs,
    ) -> dict:
        """Add a finding using a pre-defined template.

        Templates ensure consistent formatting for common finding types.

        Available templates:
        - computation_result: claim, method, result [script]
        - failed_approach: approach, goal, reason [error]
        - structural_discovery: structure, property, implication [proof_sketch]
        - verification: claim, method, outcome [script, tolerance]
        - hypothesis: hypothesis, motivation, status [tests_needed]

        Args:
            template_name: Name of template to use
            project: Project name
            tags: Tags (auto-suggested if not provided)
            **kwargs: Template fields (required and optional)

        Returns:
            Result from add() method
        """
        if template_name not in self.TEMPLATES:
            available = ", ".join(self.TEMPLATES.keys())
            raise ValueError(f"Unknown template: {template_name}. Available: {available}")

        template = self.TEMPLATES[template_name]

        # Check required fields
        missing = [f for f in template["required"] if f not in kwargs]
        if missing:
            raise ValueError(f"Missing required fields for {template_name}: {missing}")

        # Format content
        content = template["format"].format(**{k: kwargs.get(k, "") for k in template["required"]})

        # Add optional fields as evidence
        evidence_parts = []
        for opt in template.get("optional", []):
            if opt in kwargs and kwargs[opt]:
                evidence_parts.append(f"{opt}: {kwargs[opt]}")
        evidence = "\n".join(evidence_parts) if evidence_parts else None

        return self.add(
            content=content,
            finding_type=template["default_type"],
            project=project,
            tags=tags,
            evidence=evidence,
        )

    def find_citing_findings(self, finding_id: str) -> list[dict]:
        """Find findings that reference this finding ID in their content.

        Used for supersession notifications to warn about impacted findings.

        Args:
            finding_id: The finding ID to search for

        Returns:
            List of findings that contain references to this finding
        """
        # Search for findings containing this ID in their content
        pattern = f"%{finding_id}%"
        rows = self.conn.execute(
            """SELECT id, type, content, tags, project, created_at
               FROM findings
               WHERE content LIKE ? AND id != ? AND status = 'current'""",
            (pattern, finding_id)
        ).fetchall()

        return [
            {
                "id": row["id"],
                "type": row["type"],
                "content": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
                "tags": json.loads(row["tags"] or "[]"),
                "project": row["project"],
            }
            for row in rows
        ]

    def review_queue(
        self,
        project: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Get findings that need attention.

        Returns findings grouped by issue type:
        - untagged: Findings with no tags
        - low_quality: Findings flagged by validation
        - stale: Findings older than 30 days not recently cited
        - orphaned: Superseded findings with no replacement

        Args:
            project: Filter by project
            limit: Max findings per category

        Returns:
            dict with categories as keys, each containing list of findings
        """
        queue = {
            "untagged": [],
            "low_quality": [],
            "stale": [],
            "orphaned": [],
        }

        # Base query parts
        base_where = "WHERE status = 'current'"
        params: list = []
        if project:
            base_where += " AND project = ?"
            params = [project]

        # Untagged findings
        rows = self.conn.execute(
            f"""SELECT id, type, content, created_at, project
                FROM findings {base_where}
                AND (tags IS NULL OR tags = '[]')
                ORDER BY created_at DESC LIMIT ?""",
            params + [limit]
        ).fetchall()
        queue["untagged"] = [
            {"id": r["id"], "type": r["type"], "content": r["content"][:100], "created_at": r["created_at"]}
            for r in rows
        ]

        # Low quality (run validation)
        all_findings = self.conn.execute(
            f"SELECT id, type, content, tags, created_at FROM findings {base_where} LIMIT 100",
            params
        ).fetchall()
        for row in all_findings:
            warnings = validate_finding_content(row["content"], json.loads(row["tags"] or "[]"))
            if warnings:
                queue["low_quality"].append({
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"][:100],
                    "warnings": [w["message"] for w in warnings],
                })
                if len(queue["low_quality"]) >= limit:
                    break

        # Stale findings (older than 30 days)
        cutoff = (datetime.now() - __import__("datetime").timedelta(days=30)).isoformat()
        rows = self.conn.execute(
            f"""SELECT id, type, content, created_at
                FROM findings {base_where}
                AND created_at < ?
                ORDER BY created_at ASC LIMIT ?""",
            params + [cutoff, limit]
        ).fetchall()
        queue["stale"] = [
            {"id": r["id"], "type": r["type"], "content": r["content"][:100], "created_at": r["created_at"]}
            for r in rows
        ]

        # Orphaned: findings that supersede something but the chain is broken
        rows = self.conn.execute(
            f"""SELECT f.id, f.type, f.content, f.supersedes_id
                FROM findings f
                LEFT JOIN findings f2 ON f.supersedes_id = f2.id
                WHERE f.status = 'current' AND f.supersedes_id IS NOT NULL AND f2.id IS NULL
                LIMIT ?""",
            [limit]
        ).fetchall()
        queue["orphaned"] = [
            {"id": r["id"], "type": r["type"], "content": r["content"][:100], "missing_ref": r["supersedes_id"]}
            for r in rows
        ]

        return queue

    def generate_open_questions(
        self,
        project: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Analyze findings to identify knowledge gaps and open questions.

        Uses LLM to analyze existing findings and identify:
        - Areas lacking coverage
        - Unresolved issues
        - Natural next steps

        Args:
            project: Filter by project
            limit: Number of questions to generate

        Returns:
            List of dicts with 'question', 'context', 'related_findings'
        """
        # Get a sample of recent findings
        findings = self.list_findings(project=project, limit=50)
        if not findings:
            return []

        # Summarize what we know
        summaries = []
        for f in findings[:30]:
            summaries.append(f"[{f['type']}] {f['content'][:150]}")

        knowledge_summary = "\n".join(summaries)

        prompt = f"""Analyze these findings and identify {limit} open research questions.

Findings:
{knowledge_summary}

Return a JSON object with key "questions" containing an array of objects with "question", "importance", and "related_topics" fields."""

        response = self._llm_complete(prompt, max_tokens=800, json_mode=True)
        if not response:
            return []

        try:
            # With json_mode, response should be valid JSON
            data = json.loads(response)
            # Handle both {"questions": [...]} and direct [...] formats
            if isinstance(data, dict) and "questions" in data:
                return data["questions"][:limit]
            elif isinstance(data, list):
                return data[:limit]
        except json.JSONDecodeError:
            pass

        return []

    def check_contradictions(
        self,
        content: str,
        project: str | None = None,
        threshold: float = 0.4,
    ) -> list[dict]:
        """Check if new content contradicts existing findings.

        Uses semantic search + LLM analysis to find potential contradictions.

        Args:
            content: The new finding content to check
            project: Filter to specific project
            threshold: Similarity threshold for candidate findings (default 0.4)

        Returns:
            List of potential contradictions with analysis
        """
        # Find similar findings
        similar = self.search(
            query=content,
            project=project,
            limit=15,
            hybrid=True,
        )

        if not similar:
            return []

        # Filter by threshold (lower threshold to catch contradictions about same topic)
        candidates = [s for s in similar if s.get("similarity", 0) > threshold]
        if not candidates:
            return []

        # Use LLM to check for contradictions
        contradictions = []
        for candidate in candidates[:5]:  # Limit LLM calls
            prompt = f"""Compare these two statements. Do they CONTRADICT each other?

EXISTING: {candidate['content'][:500]}

NEW: {content}

Return JSON with "contradicts" (boolean) and "explanation" (string) fields."""

            response = self._llm_complete(prompt, max_tokens=200, json_mode=True)
            if response:
                try:
                    result = json.loads(response)
                    if result.get("contradicts"):
                        contradictions.append({
                            "existing_id": candidate["id"],
                            "existing_content": candidate["content"],
                            "explanation": result.get("explanation", "Potential contradiction detected"),
                            "similarity": candidate.get("similarity", 0),
                        })
                except json.JSONDecodeError:
                    pass

        return contradictions

    def summarize_topic(
        self,
        topic: str,
        project: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Synthesize a summary of all findings on a topic.

        Searches for relevant findings and uses LLM to create a coherent
        summary that captures the current state of knowledge.

        Args:
            topic: Topic to summarize
            project: Filter by project
            limit: Max findings to consider

        Returns:
            dict with 'summary', 'key_findings', 'open_questions', 'sources'
        """
        # Search for relevant findings
        findings = self.search(
            query=topic,
            project=project,
            limit=limit,
            expand=True,
            hybrid=True,
        )

        if not findings:
            return {
                "summary": f"No findings found for topic: {topic}",
                "key_findings": [],
                "open_questions": [],
                "sources": [],
            }

        # Group by type
        by_type: dict[str, list] = {}
        for f in findings:
            by_type.setdefault(f["type"], []).append(f)

        # Format for LLM
        context_parts = []
        for ftype, flist in by_type.items():
            context_parts.append(f"\n=== {ftype.upper()} ===")
            for f in flist[:10]:
                context_parts.append(f"[{f['id']}] {f['content']}")

        context = "\n".join(context_parts)

        prompt = f"""Summarize the current state of knowledge about "{topic}" based on these findings:

{context}

Provide:
1. A coherent 2-3 paragraph summary
2. Key established facts (bullet points)
3. Open questions or unresolved issues
4. Contradictions or tensions if any

Be specific and cite finding IDs where relevant."""

        response = self._llm_complete(prompt, max_tokens=1000)

        if response:
            # Extract from JSON if LLM returned JSON-wrapped response
            response = self._extract_text_from_json(response, keys=["summary", "text", "response"])

        return {
            "summary": response or "Failed to generate summary",
            "finding_count": len(findings),
            "types_found": list(by_type.keys()),
            "sources": [{"id": f["id"], "type": f["type"], "similarity": f.get("similarity", 0)} for f in findings[:10]],
        }

    def close(self):
        """Close database connection."""
        self.conn.close()


def parse_markdown_findings(file_path: Path) -> list[dict]:
    """Parse a markdown file and extract findings.

    Looks for patterns like:
    - **[SUCCESS]** or **[FAILURE]** markers
    - Bullet points with key findings
    - Sections with results/conclusions
    """
    import re

    content = file_path.read_text()
    findings = []

    # Pattern 1: Explicit markers like **[SUCCESS]**, **[FAILURE]**, etc.
    marker_pattern = re.compile(
        r'\*\*\[(SUCCESS|FAILURE|EXPERIMENT|DISCOVERY)\]\*\*[:\s]*(.+?)(?=\n\n|\n\*\*\[|\Z)',
        re.IGNORECASE | re.DOTALL
    )
    for match in marker_pattern.finditer(content):
        finding_type = match.group(1).lower()
        text = match.group(2).strip()
        findings.append({
            'type': finding_type,
            'content': text[:500],
            'evidence': None,
        })

    # Pattern 2: Key result sections (## Results, ## Findings, ## Conclusions)
    section_pattern = re.compile(
        r'^##\s+(Results?|Findings?|Conclusions?|Key\s+Findings?)\s*\n(.*?)(?=\n##|\Z)',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    for match in section_pattern.finditer(content):
        section_content = match.group(2).strip()
        # Extract bullet points
        bullets = re.findall(r'^[-*]\s+(.+)$', section_content, re.MULTILINE)
        for bullet in bullets:
            if len(bullet) > 30:  # Skip short bullets
                findings.append({
                    'type': 'discovery',
                    'content': bullet.strip()[:500],
                    'evidence': None,
                })

    # Pattern 3: Numbered conclusions/results
    numbered_pattern = re.compile(r'^\d+\.\s+\*\*(.+?)\*\*[:\s]*(.+?)(?=\n\d+\.|\n\n|\Z)', re.MULTILINE | re.DOTALL)
    for match in numbered_pattern.finditer(content):
        title = match.group(1).strip()
        desc = match.group(2).strip()
        full = f"{title}: {desc}" if desc else title
        if len(full) > 40:
            findings.append({
                'type': 'discovery',
                'content': full[:500],
                'evidence': None,
            })

    # Deduplicate by content similarity
    seen = set()
    unique = []
    for f in findings:
        key = f['content'][:100].lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


def parse_script_findings(file_path: Path) -> list[dict]:
    """Parse a Python script and extract docstrings as findings.

    Extracts:
    - Module-level docstrings
    - Class docstrings with class name
    - Function/method docstrings with function name
    """
    import ast

    content = file_path.read_text()
    findings = []

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return [{"type": "failure", "content": f"Syntax error in {file_path}: {e}", "evidence": None}]

    # Module docstring
    module_doc = ast.get_docstring(tree)
    if module_doc and len(module_doc) > 30:
        findings.append({
            "type": "discovery",
            "content": f"[{file_path.name}] {module_doc[:500]}",
            "evidence": None,
            "tags": ["docstring", "module"],
        })

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            if doc and len(doc) > 30:
                findings.append({
                    "type": "discovery",
                    "content": f"[{file_path.name}::{node.name}] {doc[:500]}",
                    "evidence": None,
                    "tags": ["docstring", "class"],
                })

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc and len(doc) > 30:
                findings.append({
                    "type": "discovery",
                    "content": f"[{file_path.name}::{node.name}()] {doc[:500]}",
                    "evidence": None,
                    "tags": ["docstring", "function"],
                })

    return findings


def format_finding(finding: dict, verbose: bool = False) -> str:
    """Format a finding for display."""
    type_colors = {
        "success": "\033[32m",     # green
        "failure": "\033[31m",     # red
        "experiment": "\033[33m",  # yellow
        "discovery": "\033[36m",   # cyan
        "correction": "\033[35m",  # magenta
    }
    reset = "\033[0m"
    dim = "\033[2m"

    color = type_colors.get(finding["type"], "")
    status_marker = " [SUPERSEDED]" if finding["status"] == "superseded" else ""

    lines = [
        f"{color}[{finding['type'].upper()}]{reset}{status_marker} {dim}{finding['id']}{reset}",
    ]

    if finding.get("project") or finding.get("sprint"):
        meta = []
        if finding.get("project"):
            meta.append(f"project={finding['project']}")
        if finding.get("sprint"):
            meta.append(f"sprint={finding['sprint']}")
        lines.append(f"  {dim}{' '.join(meta)}{reset}")

    # Always show similarity for search results
    if finding.get("similarity") is not None:
        sim = finding["similarity"]
        # Color code by similarity: green (>0.8), yellow (0.6-0.8), red (<0.6)
        if sim >= 0.8:
            sim_color = "\033[32m"  # green
        elif sim >= 0.6:
            sim_color = "\033[33m"  # yellow
        else:
            sim_color = "\033[31m"  # red
        lines[0] += f" {sim_color}({sim:.2f}){reset}"

    lines.append(f"  {finding['content']}")

    if verbose:
        if finding.get("evidence"):
            lines.append(f"  {dim}Evidence: {finding['evidence'][:200]}...{reset}" if len(finding.get("evidence", "")) > 200 else f"  {dim}Evidence: {finding['evidence']}{reset}")
        if finding.get("supersedes_id"):
            lines.append(f"  {dim}Supersedes: {finding['supersedes_id']}{reset}")
        if finding.get("tags"):
            lines.append(f"  {dim}Tags: {', '.join(finding['tags'])}{reset}")
        lines.append(f"  {dim}Created: {finding['created_at']}{reset}")
        if finding.get("similarity"):
            lines.append(f"  {dim}Similarity: {finding['similarity']:.3f}{reset}")

    return "\n".join(lines)


def format_finding_markdown(finding: dict) -> str:
    """Format a finding as Markdown for detailed display (kb get)."""
    lines = [f"## [{finding['type'].upper()}] {finding['id']}"]

    meta = []
    if finding.get("project"):
        meta.append(f"**Project:** {finding['project']}")
    if finding.get("sprint"):
        meta.append(f"**Sprint:** {finding['sprint']}")
    if finding.get("status") == "superseded":
        meta.append("*SUPERSEDED*")
    if meta:
        lines.append(" | ".join(meta))

    if finding.get("summary"):
        lines.append(f"\n**Summary:** {finding['summary']}")

    lines.append(f"\n### Content\n{finding['content']}")

    if finding.get("evidence"):
        lines.append(f"\n### Evidence\n```\n{finding['evidence']}\n```")

    if finding.get("tags"):
        lines.append(f"\n**Tags:** {', '.join(finding['tags'])}")

    if finding.get("supersedes_id"):
        lines.append(f"\n**Supersedes:** {finding['supersedes_id']}")

    lines.append(f"\n*Created: {finding['created_at']}*")

    return "\n".join(lines)


def markdown_to_html(text: str) -> str:
    """Convert simple markdown to HTML for web display."""
    # Escape HTML first (security)
    text = html.escape(text)
    # Headers
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    # Bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Code blocks
    text = re.sub(r'```\n?(.*?)\n?```', r'<pre><code>\1</code></pre>', text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    # Paragraphs (double newline)
    text = re.sub(r'\n\n+', '</p><p>', text)
    return f'<p>{text}</p>'


def render_html_page(title: str, content: str, sidebar: str = "") -> str:
    """Render an HTML page with consistent styling for kb serve."""
    sidebar_html = f'<aside class="sidebar">{sidebar}</aside>' if sidebar else ''
    main_class = "main-with-sidebar" if sidebar else "main-full"
    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(title)} - Knowledge Base</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #1a1a1a; color: #e0e0e0; }}
        .container {{ display: flex; min-height: 100vh; }}
        .sidebar {{ width: 220px; background: #151515; padding: 1rem; border-right: 1px solid #333; flex-shrink: 0; overflow-y: auto; }}
        .sidebar h3 {{ margin: 0.5rem 0; font-size: 0.85rem; color: #888; text-transform: uppercase; }}
        .sidebar ul {{ list-style: none; padding: 0; margin: 0 0 1rem 0; }}
        .sidebar li {{ margin: 0.2rem 0; }}
        .sidebar a {{ color: #e0e0e0; text-decoration: none; display: block; padding: 0.3rem 0.5rem; border-radius: 3px; font-size: 0.9rem; }}
        .sidebar a:hover {{ background: #252525; }}
        .sidebar a.active {{ background: #6db3f2; color: #000; }}
        .sidebar .count {{ color: #666; font-size: 0.8rem; }}
        .sidebar label {{ display: block; font-size: 0.9rem; padding: 0.3rem 0; cursor: pointer; }}
        .sidebar input[type="checkbox"] {{ margin-right: 0.5rem; }}
        .main-with-sidebar {{ flex: 1; padding: 1rem; max-width: 900px; }}
        .main-full {{ flex: 1; padding: 1rem; max-width: 900px; margin: 0 auto; }}
        a {{ color: #6db3f2; }}
        h1, h2, h3 {{ color: #fff; }}
        h1 {{ margin-top: 0; }}
        .finding {{ border: 1px solid #333; padding: 1rem; margin: 0.5rem 0; border-radius: 4px; background: #252525; }}
        .finding-type {{ font-weight: bold; text-transform: uppercase; font-size: 0.85rem; }}
        .success {{ color: #4caf50; }}
        .failure {{ color: #f44336; }}
        .discovery {{ color: #2196f3; }}
        .experiment {{ color: #ff9800; }}
        .correction {{ color: #9c27b0; }}
        .meta {{ color: #888; font-size: 0.9em; }}
        .tag {{ display: inline-block; background: #333; padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.8rem; margin: 0.1rem; }}
        pre {{ background: #333; padding: 1rem; overflow-x: auto; border-radius: 4px; }}
        code {{ background: #333; padding: 0.2em 0.4em; border-radius: 3px; }}
        .search-form {{ margin: 1rem 0; }}
        .search-form input[type="text"] {{ padding: 0.5rem; width: 300px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; }}
        .search-form button {{ padding: 0.5rem 1rem; background: #6db3f2; border: none; border-radius: 4px; cursor: pointer; }}
        nav {{ margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #333; }}
        nav a {{ margin-right: 1rem; }}
        .pagination {{ margin-top: 1rem; }}
        .pagination a {{ margin-right: 0.5rem; }}
        .filter-active {{ background: #333; padding: 0.3rem 0.6rem; border-radius: 3px; margin-right: 0.5rem; display: inline-block; }}
        .filter-active a {{ color: #f44; text-decoration: none; margin-left: 0.3rem; }}
        .live-indicator {{ position: fixed; bottom: 1rem; right: 1rem; padding: 0.3rem 0.6rem; border-radius: 3px; font-size: 0.8rem; }}
        .live-indicator.connected {{ background: #4caf50; color: #000; }}
        .live-indicator.disconnected {{ background: #f44336; color: #fff; }}
    </style>
</head>
<body>
    <div class="container">
        {sidebar_html}
        <main class="{main_class}">
            <nav><a href="/">Recent</a> <a href="/search">Search</a></nav>
            <h1>{html.escape(title)}</h1>
            {content}
        </main>
    </div>
    <div id="live-indicator" class="live-indicator disconnected">&#x25cf; Connecting...</div>
    <script>
    (function() {{
        var indicator = document.getElementById('live-indicator');
        var ws = null;
        var reconnectDelay = 1000;

        function connect() {{
            var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(proto + '//' + location.host + '/ws');

            ws.onopen = function() {{
                indicator.className = 'live-indicator connected';
                indicator.innerHTML = '&#x25cf; Live';
                reconnectDelay = 1000;
            }};

            ws.onmessage = function(e) {{
                var msg = JSON.parse(e.data);
                if (msg.type === 'update') {{
                    indicator.innerHTML = '&#x25cf; Updating...';
                    location.reload();
                }}
            }};

            ws.onclose = function() {{
                indicator.className = 'live-indicator disconnected';
                indicator.innerHTML = '&#x25cf; Reconnecting...';
                setTimeout(connect, reconnectDelay);
                reconnectDelay = Math.min(reconnectDelay * 2, 30000);
            }};

            ws.onerror = function() {{
                ws.close();
            }};
        }}

        connect();
    }})();
    </script>
</body>
</html>'''


def render_sidebar(stats: dict, all_tags: list, current_filters: dict) -> str:
    """Render the filter sidebar for kb serve."""
    project = current_filters.get('project', '')
    finding_type = current_filters.get('type', '')
    tag = current_filters.get('tag', '')
    include_superseded = current_filters.get('superseded', False)

    def build_url(add_params: dict = None, remove_params: list = None) -> str:
        params = dict(current_filters)
        if remove_params:
            for p in remove_params:
                params.pop(p, None)
        if add_params:
            params.update(add_params)
        params.pop('page', None)  # Reset page when filtering
        if not params:
            return "/"
        return "/?" + "&".join(f"{k}={html.escape(str(v))}" for k, v in params.items() if v)

    lines = []

    # Projects
    lines.append('<h3>Projects</h3><ul>')
    lines.append(f'<li><a href="{build_url(remove_params=["project"])}" class="{"active" if not project else ""}">All</a></li>')
    for proj, count in sorted(stats.get('by_project', {}).items()):
        active = 'active' if project == proj else ''
        lines.append(f'<li><a href="{build_url({"project": proj})}" class="{active}">{html.escape(proj)} <span class="count">({count})</span></a></li>')
    lines.append('</ul>')

    # Types
    lines.append('<h3>Types</h3><ul>')
    lines.append(f'<li><a href="{build_url(remove_params=["type"])}" class="{"active" if not finding_type else ""}">All</a></li>')
    for t, count in sorted(stats.get('by_type', {}).items()):
        active = 'active' if finding_type == t else ''
        lines.append(f'<li><a href="{build_url({"type": t})}" class="{active} {t}">{t} <span class="count">({count})</span></a></li>')
    lines.append('</ul>')

    # Tags (show top 20)
    if all_tags:
        lines.append('<h3>Tags</h3><ul>')
        lines.append(f'<li><a href="{build_url(remove_params=["tag"])}" class="{"active" if not tag else ""}">All</a></li>')
        for t in all_tags[:20]:
            active = 'active' if tag == t else ''
            lines.append(f'<li><a href="{build_url({"tag": t})}" class="{active}">{html.escape(t)}</a></li>')
        if len(all_tags) > 20:
            lines.append(f'<li><span class="count">+{len(all_tags) - 20} more</span></li>')
        lines.append('</ul>')

    # Superseded toggle
    lines.append('<h3>Status</h3>')
    checked = 'checked' if include_superseded else ''
    lines.append(f'''<label><input type="checkbox" {checked} onchange="window.location.href='{build_url({"superseded": "1"}) if not include_superseded else build_url(remove_params=["superseded"])}'"> Include superseded</label>''')

    return '\n'.join(lines)


def format_finding_summary(finding: dict) -> str:
    """Format a finding for search results - summary only, compact format."""
    type_colors = {
        "success": "\033[32m",
        "failure": "\033[31m",
        "experiment": "\033[33m",
        "discovery": "\033[36m",
        "correction": "\033[35m",
    }
    reset = "\033[0m"
    dim = "\033[2m"

    color = type_colors.get(finding["type"], "")

    # Build metadata string
    meta = []
    if finding.get("project"):
        meta.append(f"project={finding['project']}")
    if finding.get("sprint"):
        meta.append(f"sprint={finding['sprint']}")
    meta_str = f" ({', '.join(meta)})" if meta else ""

    # Similarity indicator
    sim_str = ""
    if finding.get("similarity") is not None:
        sim = finding["similarity"]
        if sim >= 0.8:
            sim_color = "\033[32m"
        elif sim >= 0.6:
            sim_color = "\033[33m"
        else:
            sim_color = "\033[31m"
        sim_str = f" {sim_color}sim={sim:.3f}{reset}"

    # Use summary if available, otherwise truncate content
    display_text = finding.get("summary") or finding.get("content", "")[:100]

    return f"{color}[{finding['type'].upper()}]{reset} {dim}{finding['id']}{reset}{meta_str}{sim_str}\n  {display_text}"


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Base - Record and retrieve findings with semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kb add --type=failure "AWQ quantization fails on gfx1100 with tile_size=64"
  kb add --type=success --project=ck --sprint=8 "FMHA works with page_block_size=256"
  kb search "AWQ quantization issues"
  kb correct kb-20250114-123456 --reason="Fixed in PR #42" "AWQ works after NAVI31 patch"
  kb list --project=ck --type=failure
  kb get kb-20250114-123456
  kb chain kb-20250114-123456
        """
    )

    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Database path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # add command
    add_parser = subparsers.add_parser("add", help="Add a new finding")
    add_parser.add_argument("content", help="Finding content")
    add_parser.add_argument("-t", "--type", choices=FINDING_TYPES, default="discovery", help="Finding type")
    add_parser.add_argument("-p", "--project", help="Project name")
    add_parser.add_argument("-s", "--sprint", help="Sprint identifier")
    add_parser.add_argument("--tags", help="Comma-separated tags")
    add_parser.add_argument("-e", "--evidence", help="Supporting evidence")
    add_parser.add_argument("--force", action="store_true", help="Skip duplicate check")

    # search command
    search_parser = subparsers.add_parser("search", help="Search findings")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    search_parser.add_argument("-p", "--project", help="Filter by project")
    search_parser.add_argument("-t", "--type", choices=FINDING_TYPES, help="Filter by type")
    search_parser.add_argument("--include-superseded", action="store_true", help="Include superseded")
    search_parser.add_argument("--fts", action="store_true", help="Use full-text search instead of vector")
    search_parser.add_argument("--expand", action="store_true", help="Expand query using LLM for better recall")

    # ask command (natural language Q&A)
    ask_parser = subparsers.add_parser("ask", help="Ask a natural language question")
    ask_parser.add_argument("question", help="Natural language question")
    ask_parser.add_argument("-p", "--project", help="Filter to specific project")
    ask_parser.add_argument("-n", "--limit", type=int, default=10, help="Max findings to consider")

    # correct command
    correct_parser = subparsers.add_parser("correct", help="Correct an existing finding")
    correct_parser.add_argument("supersedes_id", help="ID of finding to supersede")
    correct_parser.add_argument("content", help="Corrected content")
    correct_parser.add_argument("-r", "--reason", help="Reason for correction")
    correct_parser.add_argument("-e", "--evidence", help="Supporting evidence")

    # list command
    list_parser = subparsers.add_parser("list", help="List findings")
    list_parser.add_argument("-p", "--project", help="Filter by project")
    list_parser.add_argument("-s", "--sprint", help="Filter by sprint")
    list_parser.add_argument("-t", "--type", choices=FINDING_TYPES, help="Filter by type")
    list_parser.add_argument("-n", "--limit", type=int, default=20, help="Max results")
    list_parser.add_argument("--include-superseded", action="store_true", help="Include superseded")

    # get command
    get_parser = subparsers.add_parser("get", help="Get a finding by ID")
    get_parser.add_argument("id", help="Finding ID")
    get_parser.add_argument("--raw", action="store_true", help="Output raw markdown without rendering")

    # chain command
    chain_parser = subparsers.add_parser("chain", help="Show supersession chain for a finding")
    chain_parser.add_argument("id", help="Finding ID")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a finding")
    delete_parser.add_argument("id", help="Finding ID")
    delete_parser.add_argument("-f", "--force", action="store_true", help="Skip confirmation")

    # stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest findings from a markdown file")
    ingest_parser.add_argument("file", type=Path, help="Markdown file to ingest")
    ingest_parser.add_argument("-p", "--project", help="Project name for all findings")
    ingest_parser.add_argument("-s", "--sprint", help="Sprint identifier for all findings")
    ingest_parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested without adding")

    # export command
    export_parser = subparsers.add_parser("export", help="Export findings as JSON")
    export_parser.add_argument("-o", "--output", type=Path, help="Output file (default: stdout)")
    export_parser.add_argument("-p", "--project", help="Filter by project")
    export_parser.add_argument("--include-superseded", action="store_true", help="Include superseded")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Serve KB via web browser")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1, localhost only)")

    # batch command
    batch_parser = subparsers.add_parser("batch", help="Batch ingest findings from JSON/JSONL file")
    batch_parser.add_argument("file", type=Path, help="JSON or JSONL file to ingest")
    batch_parser.add_argument("-p", "--project", help="Override project for all findings")
    batch_parser.add_argument("-s", "--sprint", help="Override sprint for all findings")
    batch_parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    batch_parser.add_argument("--force", action="store_true", help="Skip duplicate check")

    # docstrings command (ingest docstrings from Python files as findings)
    docstrings_parser = subparsers.add_parser("docstrings", help="Ingest docstrings from Python scripts as findings")
    docstrings_parser.add_argument("files", type=Path, nargs="+", help="Python files to ingest")
    docstrings_parser.add_argument("-p", "--project", help="Project name for all findings")
    docstrings_parser.add_argument("-s", "--sprint", help="Sprint identifier for all findings")
    docstrings_parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    docstrings_parser.add_argument("--force", action="store_true", help="Skip duplicate check")

    # script command (with subcommands for script registry)
    script_parser = subparsers.add_parser("script", help="Manage hypothesis-testing script registry")
    script_subparsers = script_parser.add_subparsers(dest="script_command", required=True)

    # script add
    script_add_parser = script_subparsers.add_parser("add", help="Register a script")
    script_add_parser.add_argument("file", type=Path, help="Script file to register")
    script_add_parser.add_argument("--purpose", "-u", required=True, help="What hypothesis this script tests")
    script_add_parser.add_argument("-p", "--project", help="Project name")
    script_add_parser.add_argument("-l", "--language", choices=["python", "sage", "bash", "other"],
                                   help="Script language (auto-detected if not specified)")
    script_add_parser.add_argument("--no-content", action="store_true", help="Don't store script content")

    # script get
    script_get_parser = script_subparsers.add_parser("get", help="Get script by ID")
    script_get_parser.add_argument("id", help="Script ID")
    script_get_parser.add_argument("--show-content", action="store_true", help="Show stored content")

    # script list
    script_list_parser = script_subparsers.add_parser("list", help="List registered scripts")
    script_list_parser.add_argument("-p", "--project", help="Filter by project")
    script_list_parser.add_argument("-l", "--language", choices=["python", "sage", "bash", "other"],
                                    help="Filter by language")
    script_list_parser.add_argument("-n", "--limit", type=int, default=50, help="Max results")

    # script search
    script_search_parser = script_subparsers.add_parser("search", help="Search scripts by purpose")
    script_search_parser.add_argument("query", help="Search query")
    script_search_parser.add_argument("-p", "--project", help="Filter by project")
    script_search_parser.add_argument("-n", "--limit", type=int, default=10, help="Max results")

    # script link
    script_link_parser = script_subparsers.add_parser("link", help="Link a finding to a script")
    script_link_parser.add_argument("script_id", help="Script ID")
    script_link_parser.add_argument("finding_id", help="Finding ID")
    script_link_parser.add_argument("-r", "--relationship", default="generated_by",
                                    choices=["generated_by", "validates", "contradicts"],
                                    help="Relationship type")

    # script findings
    script_findings_parser = script_subparsers.add_parser("findings", help="Get findings linked to a script")
    script_findings_parser.add_argument("id", help="Script ID")

    # script delete
    script_delete_parser = script_subparsers.add_parser("delete", help="Delete a script")
    script_delete_parser.add_argument("id", help="Script ID")
    script_delete_parser.add_argument("-f", "--force", action="store_true", help="Skip confirmation")

    # notation command (with subcommands)
    notation_parser = subparsers.add_parser("notation", help="Manage notation tracking")
    notation_subparsers = notation_parser.add_subparsers(dest="notation_command", required=True)

    # notation add
    notation_add_parser = notation_subparsers.add_parser("add", help="Add a new notation")
    notation_add_parser.add_argument("symbol", help="The notation symbol")
    notation_add_parser.add_argument("meaning", help="What the notation means")
    notation_add_parser.add_argument("-p", "--project", help="Project this notation belongs to")
    notation_add_parser.add_argument("-d", "--domain", choices=NOTATION_DOMAINS, help="Domain")

    # notation update
    notation_update_parser = notation_subparsers.add_parser("update", help="Update a notation symbol")
    notation_update_parser.add_argument("new_symbol", help="The new notation symbol")
    notation_update_parser.add_argument("-o", "--old-symbol", help="Old symbol to find and update")
    notation_update_parser.add_argument("-i", "--id", dest="notation_id", help="Notation ID to update")
    notation_update_parser.add_argument("-m", "--meaning", help="Updated meaning")
    notation_update_parser.add_argument("-r", "--reason", help="Reason for the change")
    notation_update_parser.add_argument("-p", "--project", help="Project to search in")

    # notation list
    notation_list_parser = notation_subparsers.add_parser("list", help="List notations")
    notation_list_parser.add_argument("-p", "--project", help="Filter by project")
    notation_list_parser.add_argument("-d", "--domain", choices=NOTATION_DOMAINS, help="Filter by domain")

    # notation search
    notation_search_parser = notation_subparsers.add_parser("search", help="Search notations")
    notation_search_parser.add_argument("query", help="Search query")
    notation_search_parser.add_argument("-p", "--project", help="Filter by project")
    notation_search_parser.add_argument("-d", "--domain", choices=NOTATION_DOMAINS, help="Filter by domain")

    # notation history
    notation_history_parser = notation_subparsers.add_parser("history", help="Show notation history")
    notation_history_parser.add_argument("id", help="Notation ID")

    # notation get
    notation_get_parser = notation_subparsers.add_parser("get", help="Get notation by ID")
    notation_get_parser.add_argument("id", help="Notation ID")

    # notation delete
    notation_delete_parser = notation_subparsers.add_parser("delete", help="Delete a notation")
    notation_delete_parser.add_argument("id", help="Notation ID")
    notation_delete_parser.add_argument("-f", "--force", action="store_true", help="Skip confirmation")

    # notation audit
    notation_audit_parser = notation_subparsers.add_parser("audit", help="Audit notation usage in documents")
    notation_audit_parser.add_argument("doc_dir", type=Path, help="Document directory to audit")
    notation_audit_parser.add_argument("-p", "--project", required=True, help="Project name")
    notation_audit_parser.add_argument("-o", "--output", type=Path, help="Output report file")

    # error command (with subcommands)
    error_parser = subparsers.add_parser("error", help="Manage error→solution tracking")
    error_subparsers = error_parser.add_subparsers(dest="error_command", required=True)

    # error add
    error_add_parser = error_subparsers.add_parser("add", help="Record an error signature")
    error_add_parser.add_argument("signature", help="The error signature/message")
    error_add_parser.add_argument("-t", "--type", help="Error type (e.g., build, runtime, test)")
    error_add_parser.add_argument("-p", "--project", help="Project name")

    # error link
    error_link_parser = error_subparsers.add_parser("link", help="Link an error to a solution")
    error_link_parser.add_argument("error_id", help="Error ID")
    error_link_parser.add_argument("finding_id", help="Finding ID (the solution)")
    error_link_parser.add_argument("-v", "--verified", action="store_true", help="Mark as verified")

    # error verify
    error_verify_parser = error_subparsers.add_parser("verify", help="Mark a solution as verified")
    error_verify_parser.add_argument("error_id", help="Error ID")
    error_verify_parser.add_argument("finding_id", help="Finding ID")

    # error get
    error_get_parser = error_subparsers.add_parser("get", help="Get error by ID with solutions")
    error_get_parser.add_argument("id", help="Error ID")

    # error search
    error_search_parser = error_subparsers.add_parser("search", help="Search errors by signature")
    error_search_parser.add_argument("query", help="Search query")
    error_search_parser.add_argument("-p", "--project", help="Filter by project")

    # error list
    error_list_parser = error_subparsers.add_parser("list", help="List errors")
    error_list_parser.add_argument("-p", "--project", help="Filter by project")
    error_list_parser.add_argument("-t", "--type", help="Filter by error type")
    error_list_parser.add_argument("-n", "--limit", type=int, default=20, help="Max results")

    # error solutions
    error_solutions_parser = error_subparsers.add_parser("solutions", help="Get solutions for an error")
    error_solutions_parser.add_argument("id", help="Error ID")

    # error delete
    error_delete_parser = error_subparsers.add_parser("delete", help="Delete an error")
    error_delete_parser.add_argument("id", help="Error ID")
    error_delete_parser.add_argument("-f", "--force", action="store_true", help="Skip confirmation")

    # bulk command (with subcommands)
    bulk_parser = subparsers.add_parser("bulk", help="Bulk operations on findings")
    bulk_subparsers = bulk_parser.add_subparsers(dest="bulk_command", required=True)

    # bulk tag
    bulk_tag_parser = bulk_subparsers.add_parser("tag", help="Add tags to multiple findings")
    bulk_tag_parser.add_argument("ids", nargs="+", help="Finding IDs to tag")
    bulk_tag_parser.add_argument("--add", dest="add_tags", required=True, help="Comma-separated tags to add")

    # bulk consolidate
    bulk_consolidate_parser = bulk_subparsers.add_parser("consolidate", help="Merge multiple findings into one")
    bulk_consolidate_parser.add_argument("ids", nargs="+", help="Finding IDs to consolidate")
    bulk_consolidate_parser.add_argument("--summary", "-s", required=True, help="Summary content for consolidated finding")
    bulk_consolidate_parser.add_argument("--reason", "-r", required=True, help="Reason for consolidation")
    bulk_consolidate_parser.add_argument("-t", "--type", choices=FINDING_TYPES, default="discovery", help="Finding type")
    bulk_consolidate_parser.add_argument("--tags", help="Comma-separated tags (default: merge from source findings)")
    bulk_consolidate_parser.add_argument("-e", "--evidence", help="Evidence text")

    # doc command (with subcommands)
    doc_parser = subparsers.add_parser("doc", help="Manage authoritative documents")
    doc_subparsers = doc_parser.add_subparsers(dest="doc_command", required=True)

    # doc add
    doc_add_parser = doc_subparsers.add_parser("add", help="Add an authoritative document")
    doc_add_parser.add_argument("title", help="Document title")
    doc_add_parser.add_argument("-t", "--type", dest="doc_type", required=True,
                                 choices=["spec", "paper", "standard", "internal", "reference"],
                                 help="Document type")
    doc_add_parser.add_argument("-u", "--url", help="URL or file path")
    doc_add_parser.add_argument("-p", "--project", help="Project name")
    doc_add_parser.add_argument("-s", "--summary", help="Brief description")

    # doc get
    doc_get_parser = doc_subparsers.add_parser("get", help="Get document by ID")
    doc_get_parser.add_argument("id", help="Document ID")

    # doc list
    doc_list_parser = doc_subparsers.add_parser("list", help="List documents")
    doc_list_parser.add_argument("-p", "--project", help="Filter by project")
    doc_list_parser.add_argument("-t", "--type", dest="doc_type",
                                  choices=["spec", "paper", "standard", "internal", "reference"],
                                  help="Filter by type")
    doc_list_parser.add_argument("-n", "--limit", type=int, default=50, help="Max results")
    doc_list_parser.add_argument("--include-superseded", action="store_true", help="Include superseded")

    # doc search
    doc_search_parser = doc_subparsers.add_parser("search", help="Search documents")
    doc_search_parser.add_argument("query", help="Search query")
    doc_search_parser.add_argument("-p", "--project", help="Filter by project")

    # doc cite
    doc_cite_parser = doc_subparsers.add_parser("cite", help="Link a finding to a document")
    doc_cite_parser.add_argument("finding_id", help="Finding ID")
    doc_cite_parser.add_argument("doc_id", help="Document ID")
    doc_cite_parser.add_argument("-t", "--type", dest="citation_type", default="references",
                                  choices=["references", "implements", "contradicts", "extends"],
                                  help="Citation type")
    doc_cite_parser.add_argument("-n", "--notes", help="Citation notes")

    # doc citations
    doc_citations_parser = doc_subparsers.add_parser("citations", help="Get findings citing a document")
    doc_citations_parser.add_argument("id", help="Document ID")

    # doc finding-docs
    doc_finding_docs_parser = doc_subparsers.add_parser("finding-docs", help="Get documents cited by a finding")
    doc_finding_docs_parser.add_argument("id", help="Finding ID")

    # doc supersede
    doc_supersede_parser = doc_subparsers.add_parser("supersede", help="Mark document as superseded")
    doc_supersede_parser.add_argument("doc_id", help="Document to supersede")
    doc_supersede_parser.add_argument("new_doc_id", help="Document that supersedes it")

    # doc delete
    doc_delete_parser = doc_subparsers.add_parser("delete", help="Delete a document")
    doc_delete_parser.add_argument("id", help="Document ID")
    doc_delete_parser.add_argument("-f", "--force", action="store_true", help="Skip confirmation")

    # Re-embed all findings (for when embedding model/algorithm changes)
    subparsers.add_parser("reembed", help="Re-generate embeddings for all findings")

    # Backfill summaries for existing findings
    backfill_parser = subparsers.add_parser("backfill-summaries", help="Generate summaries for findings missing them")
    backfill_parser.add_argument("-p", "--project", help="Project filter")
    backfill_parser.add_argument("-n", "--batch-size", type=int, default=20, help="Batch size (default: 20)")

    # Reconcile documents with KB
    reconcile_parser = subparsers.add_parser("reconcile", help="Reconcile documents with KB findings")
    reconcile_parser.add_argument("doc_dir", type=Path, help="Document directory to reconcile")
    reconcile_parser.add_argument("-p", "--project", required=True, help="Project name")
    reconcile_parser.add_argument("-o", "--output", type=Path, help="Output report file")
    reconcile_parser.add_argument("--export-missing", type=Path, help="Export missing claims as JSON")
    reconcile_parser.add_argument("--claim-types", help="Filter: theorem,definition,table_row (comma-separated)")
    reconcile_parser.add_argument("--min-length", type=int, default=50, help="Min claim length (default: 50)")
    reconcile_parser.add_argument("--import-missing", action="store_true", help="Import missing claims to KB")

    # LLM-powered features
    llm_parser = subparsers.add_parser("llm", help="LLM-powered analysis tools")
    llm_subparsers = llm_parser.add_subparsers(dest="llm_command")

    llm_suggest_tags_parser = llm_subparsers.add_parser("suggest-tags", help="Suggest tags for content")
    llm_suggest_tags_parser.add_argument("content", help="Content to analyze")
    llm_suggest_tags_parser.add_argument("-p", "--project", help="Project for context")

    llm_classify_parser = llm_subparsers.add_parser("classify", help="Classify finding type from content")
    llm_classify_parser.add_argument("content", help="Content to classify")

    llm_duplicates_parser = llm_subparsers.add_parser("duplicates", help="Check for duplicate findings")
    llm_duplicates_parser.add_argument("content", help="Content to check")
    llm_duplicates_parser.add_argument("-p", "--project", help="Project filter")
    llm_duplicates_parser.add_argument("-t", "--threshold", type=float, default=0.85, help="Similarity threshold")

    llm_normalize_parser = llm_subparsers.add_parser("normalize-error", help="Normalize error signature")
    llm_normalize_parser.add_argument("error", help="Error text to normalize")

    llm_xref_parser = llm_subparsers.add_parser("xref", help="Suggest cross-references for a finding")
    llm_xref_parser.add_argument("finding_id", help="Finding ID")
    llm_xref_parser.add_argument("-p", "--project", help="Project filter")

    llm_summarize_parser = llm_subparsers.add_parser("summarize", help="Summarize long evidence")
    llm_summarize_parser.add_argument("evidence", help="Evidence text")
    llm_summarize_parser.add_argument("-n", "--max-length", type=int, default=200, help="Max summary length")

    llm_notations_parser = llm_subparsers.add_parser("detect-notations", help="Detect notations in content")
    llm_notations_parser.add_argument("content", help="Content to analyze")
    llm_notations_parser.add_argument("-p", "--project", help="Project for context")

    llm_claims_parser = llm_subparsers.add_parser("extract-claims", help="Extract claims from text")
    llm_claims_parser.add_argument("text", help="Text to analyze")

    llm_consolidate_parser = llm_subparsers.add_parser("consolidate", help="Suggest finding consolidations")
    llm_consolidate_parser.add_argument("-p", "--project", help="Project filter")
    llm_consolidate_parser.add_argument("-n", "--limit", type=int, default=50, help="Max findings to analyze")

    args = parser.parse_args()

    kb = KnowledgeBase(
        db_path=args.db,
        embedding_url=os.environ.get("KB_EMBEDDING_URL", ""),
        embedding_dim=int(os.environ.get("KB_EMBEDDING_DIM", "4096")),
    )

    try:
        if args.command == "add":
            tags = args.tags.split(",") if args.tags else None
            result = kb.add(
                content=args.content,
                finding_type=args.type,
                project=args.project,
                sprint=args.sprint,
                tags=tags,
                evidence=args.evidence,
                check_duplicate=not args.force,
            )
            print(f"Added: {result['id']}")
            if result.get("type_suggested"):
                # Get the finding to show what type was chosen
                f = kb.get(result["id"])
                if f:
                    print(f"  [auto] Type classified as: {f['type']}")
            if result.get("tags_suggested"):
                f = kb.get(result["id"])
                if f and f.get("tags"):
                    print(f"  [auto] Tags suggested: {', '.join(f['tags'])}")
            if result.get("evidence_summarized"):
                print(f"  [auto] Evidence summarized (was > 500 chars)")
            if result.get("cross_refs"):
                xr = result["cross_refs"]
                if xr.get("findings") or xr.get("scripts") or xr.get("docs"):
                    print("  [auto] Related items found:")
                    for rf in xr.get("findings", [])[:2]:
                        print(f"    Finding: {rf['id']} (sim: {rf['similarity']:.2f})")
                    for rs in xr.get("scripts", [])[:2]:
                        print(f"    Script: {rs['filename']}")
                    for rd in xr.get("docs", [])[:2]:
                        print(f"    Doc: {rd['title']}")
            if result.get("notations_detected"):
                new_notations = [n for n in result["notations_detected"] if not n.get("exists")]
                if new_notations:
                    print(f"  [auto] New notations detected: {', '.join(n['symbol'] for n in new_notations[:3])}")

        elif args.command == "search":
            results = kb.search(
                query=args.query,
                limit=args.limit,
                project=args.project,
                finding_type=args.type,
                include_superseded=args.include_superseded,
                hybrid=not args.fts,
                expand=args.expand,
                verbose=args.verbose,
            )
            if not results:
                print("No findings found.")
            else:
                for finding in results:
                    print(format_finding_summary(finding))
                print()

        elif args.command == "ask":
            result = kb.ask(
                question=args.question,
                project=args.project,
                limit=args.limit,
                verbose=args.verbose,
            )
            print(result["answer"])
            print()
            if result["sources"]:
                print("Sources:")
                for src in result["sources"]:
                    print(f"  [{src['id']}] ({src['similarity']:.2f}) {src['content']}")

        elif args.command == "correct":
            finding_id = kb.correct(
                supersedes_id=args.supersedes_id,
                content=args.content,
                reason=args.reason,
                evidence=args.evidence,
            )
            print(f"Created correction: {finding_id}")
            print(f"Superseded: {args.supersedes_id}")

        elif args.command == "list":
            results = kb.list_findings(
                project=args.project,
                sprint=args.sprint,
                finding_type=args.type,
                include_superseded=args.include_superseded,
                limit=args.limit,
            )
            if not results:
                print("No findings found.")
            else:
                for finding in results:
                    print(format_finding(finding, verbose=args.verbose))
                    print()

        elif args.command == "get":
            finding = kb.get(args.id)
            if not finding:
                print(f"Finding not found: {args.id}")
                sys.exit(1)
            md = format_finding_markdown(finding)
            if args.raw or not RICH_AVAILABLE:
                print(md)
            else:
                Console().print(Markdown(md))

        elif args.command == "chain":
            chain = kb.get_supersession_chain(args.id)
            if not chain:
                print(f"Finding not found: {args.id}")
                sys.exit(1)
            print(f"Supersession chain ({len(chain)} findings):\n")
            for i, finding in enumerate(chain):
                prefix = "└── " if i == len(chain) - 1 else "├── "
                if i > 0:
                    print("│")
                print(f"{prefix}{format_finding(finding, verbose=args.verbose)}")

        elif args.command == "delete":
            if not args.force:
                finding = kb.get(args.id)
                if finding:
                    print(format_finding(finding, verbose=True))
                    confirm = input("\nDelete this finding? [y/N] ")
                    if confirm.lower() != "y":
                        print("Cancelled.")
                        sys.exit(0)

            if kb.delete(args.id):
                print(f"Deleted: {args.id}")
            else:
                print(f"Finding not found: {args.id}")
                sys.exit(1)

        elif args.command == "stats":
            stats = kb.stats()
            print(f"Database: {stats['db_path']}")
            print(f"Total findings: {stats['total']}")
            print(f"  Current: {stats['current']}")
            print(f"  Superseded: {stats['superseded']}")
            print(f"\nBy type:")
            for t, count in sorted(stats["by_type"].items()):
                print(f"  {t}: {count}")
            if stats["by_project"]:
                print(f"\nBy project:")
                for p, count in sorted(stats["by_project"].items()):
                    print(f"  {p}: {count}")

        elif args.command == "ingest":
            try:
                findings = parse_markdown_findings(args.file)
            except FileNotFoundError:
                print(f"Error: File not found: {args.file}")
                sys.exit(1)
            if not findings:
                print(f"No findings extracted from {args.file}")
                sys.exit(1)

            print(f"Found {len(findings)} potential findings in {args.file}:\n")
            for i, f in enumerate(findings, 1):
                print(f"{i}. [{f['type']}] {f['content'][:80]}...")
                if f.get('evidence'):
                    print(f"   Evidence: {f['evidence'][:60]}...")

            if args.dry_run:
                print("\n(dry-run mode, nothing added)")
            else:
                print()
                added = 0
                skipped = 0
                for f in findings:
                    try:
                        finding_id = kb.add(
                            content=f['content'],
                            finding_type=f['type'],
                            project=args.project or f.get('project'),
                            sprint=args.sprint or f.get('sprint'),
                            tags=f.get('tags'),
                            evidence=f.get('evidence'),
                        )
                        print(f"Added: {finding_id}")
                        added += 1
                    except ValueError:
                        print(f"Skipped (duplicate): {f['content'][:50]}...")
                        skipped += 1
                print(f"\nIngested {added} findings, skipped {skipped} duplicates.")

        elif args.command == "export":
            results = kb.list_findings(
                project=args.project,
                include_superseded=args.include_superseded,
                limit=10000,
            )
            output = json.dumps(results, indent=2)
            if args.output:
                args.output.write_text(output)
                print(f"Exported {len(results)} findings to {args.output}")
            else:
                print(output)

        elif args.command == "serve":
            if not SERVE_AVAILABLE:
                print("Error: starlette and uvicorn required for 'kb serve'")
                print("Install with: pip install starlette uvicorn")
                sys.exit(1)

            # Cache stats and tags for sidebar (refresh on each request for simplicity)
            def get_sidebar_data():
                return kb.stats(), kb.get_all_tags()

            async def index(request):
                # Parse filter params
                page = int(request.query_params.get('page', '1'))
                project_filter = request.query_params.get('project', '')
                type_filter = request.query_params.get('type', '')
                tag_filter = request.query_params.get('tag', '')
                include_superseded = request.query_params.get('superseded', '') == '1'

                current_filters = {}
                if project_filter:
                    current_filters['project'] = project_filter
                if type_filter:
                    current_filters['type'] = type_filter
                if tag_filter:
                    current_filters['tag'] = tag_filter
                if include_superseded:
                    current_filters['superseded'] = '1'
                if page > 1:
                    current_filters['page'] = str(page)

                per_page = 50
                # Fetch more for tag filtering (since list_findings doesn't support it)
                fetch_limit = per_page * 3 if tag_filter else per_page + 1

                findings = kb.list_findings(
                    project=project_filter or None,
                    finding_type=type_filter or None,
                    include_superseded=include_superseded,
                    limit=fetch_limit,
                )

                # Filter by tag if needed
                if tag_filter:
                    findings = [f for f in findings if tag_filter in f.get('tags', [])]

                # Paginate
                start = (page - 1) * per_page
                has_more = len(findings) > start + per_page
                findings = findings[start:start + per_page]

                # Build sidebar
                stats, all_tags = get_sidebar_data()
                sidebar = render_sidebar(stats, all_tags, current_filters)

                # Build active filters display
                active_filters = ''
                if project_filter or type_filter or tag_filter:
                    active_filters = '<div style="margin-bottom: 1rem;">'
                    if project_filter:
                        active_filters += f'<span class="filter-active">project: {html.escape(project_filter)} <a href="/?{("&".join(f"{k}={v}" for k, v in current_filters.items() if k != "project")) or ""}">×</a></span>'
                    if type_filter:
                        active_filters += f'<span class="filter-active">type: {type_filter} <a href="/?{("&".join(f"{k}={v}" for k, v in current_filters.items() if k != "type")) or ""}">×</a></span>'
                    if tag_filter:
                        active_filters += f'<span class="filter-active">tag: {html.escape(tag_filter)} <a href="/?{("&".join(f"{k}={v}" for k, v in current_filters.items() if k != "tag")) or ""}">×</a></span>'
                    active_filters += '</div>'

                if not findings:
                    content = active_filters + '<p>No findings match the current filters.</p>'
                else:
                    items = [active_filters] if active_filters else []
                    for f in findings:
                        type_class = f['type']
                        summary = html.escape(f.get('summary') or f['content'][:100])
                        proj = html.escape(f.get('project') or '')
                        tags_html = ' '.join(f'<span class="tag">{html.escape(t)}</span>' for t in f.get('tags', [])[:5])
                        items.append(f'''<div class="finding">
                            <span class="finding-type {type_class}">[{f['type']}]</span>
                            <a href="/finding/{f['id']}">{f['id'][:12]}...</a>
                            <span class="meta">{proj}</span>
                            <p>{summary}</p>
                            {f'<div>{tags_html}</div>' if tags_html else ''}
                        </div>''')
                    content = '\n'.join(items)

                    # Pagination with current filters
                    filter_params = '&'.join(f'{k}={v}' for k, v in current_filters.items() if k != 'page')
                    pagination = '<div class="pagination">'
                    if page > 1:
                        pagination += f'<a href="/?page={page-1}{"&" + filter_params if filter_params else ""}">&laquo; Prev</a>'
                    if has_more:
                        pagination += f'<a href="/?page={page+1}{"&" + filter_params if filter_params else ""}">Next &raquo;</a>'
                    pagination += '</div>'
                    content += pagination

                title = "Findings"
                if project_filter:
                    title = f"Findings: {project_filter}"
                return HTMLResponse(render_html_page(title, content, sidebar))

            async def search_page(request):
                query = request.query_params.get('q', '')
                stats, all_tags = get_sidebar_data()
                sidebar = render_sidebar(stats, all_tags, {})

                content = f'''<form class="search-form" method="get">
                    <input type="text" name="q" value="{html.escape(query)}" placeholder="Search..." autofocus>
                    <button type="submit">Search</button>
                </form>'''
                if query:
                    results = kb.search(query, limit=30)
                    if results:
                        items = []
                        for f in results:
                            type_class = f['type']
                            summary = html.escape(f.get('summary') or f['content'][:100])
                            sim = f.get('similarity', 0)
                            proj = html.escape(f.get('project') or '')
                            tags_html = ' '.join(f'<span class="tag">{html.escape(t)}</span>' for t in f.get('tags', [])[:5])
                            items.append(f'''<div class="finding">
                                <span class="finding-type {type_class}">[{f['type']}]</span>
                                <a href="/finding/{f['id']}">{f['id'][:12]}...</a>
                                <span class="meta">sim={sim:.3f} {proj}</span>
                                <p>{summary}</p>
                                {f'<div>{tags_html}</div>' if tags_html else ''}
                            </div>''')
                        content += '\n'.join(items)
                    else:
                        content += '<p>No results found.</p>'
                return HTMLResponse(render_html_page("Search", content, sidebar))

            async def finding_page(request):
                finding_id = request.path_params['id']
                finding = kb.get(finding_id)
                if not finding:
                    return HTMLResponse(render_html_page("Not Found", "<p>Finding not found.</p>"), status_code=404)

                stats, all_tags = get_sidebar_data()
                sidebar = render_sidebar(stats, all_tags, {})

                md = format_finding_markdown(finding)
                content = markdown_to_html(md)
                return HTMLResponse(render_html_page(f"Finding {finding_id[:12]}...", content, sidebar))

            # WebSocket for live updates
            connected_clients: set = set()
            last_state = {"count": 0, "latest": ""}

            async def ws_updates(websocket: WebSocket):
                await websocket.accept()
                connected_clients.add(websocket)
                try:
                    # Send current state on connect
                    count, latest = kb.get_latest_update()
                    await websocket.send_json({"type": "state", "count": count, "latest": latest})
                    # Keep connection alive
                    while True:
                        try:
                            await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                        except asyncio.TimeoutError:
                            # Send ping to keep alive
                            await websocket.send_json({"type": "ping"})
                except Exception:
                    pass
                finally:
                    connected_clients.discard(websocket)

            async def check_for_updates():
                """Background task to check for DB changes and notify clients."""
                while True:
                    await asyncio.sleep(2)  # Check every 2 seconds
                    if connected_clients:
                        count, latest = kb.get_latest_update()
                        if count != last_state["count"] or latest != last_state["latest"]:
                            last_state["count"] = count
                            last_state["latest"] = latest
                            # Broadcast to all connected clients
                            dead = set()
                            for ws in connected_clients:
                                try:
                                    await ws.send_json({"type": "update", "count": count, "latest": latest})
                                except Exception:
                                    dead.add(ws)
                            connected_clients.difference_update(dead)

            async def on_startup():
                asyncio.create_task(check_for_updates())

            routes = [
                Route("/", index),
                Route("/search", search_page),
                Route("/finding/{id:path}", finding_page),
                WebSocketRoute("/ws", ws_updates),
            ]
            app = Starlette(routes=routes, on_startup=[on_startup])
            print(f"Starting KB server at http://{args.host}:{args.port}")
            print("WebSocket live updates enabled at /ws")
            uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

        elif args.command == "batch":
            file_path = args.file
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                sys.exit(1)

            content = file_path.read_text()
            findings = []

            # Try JSON array first
            if content.strip().startswith("["):
                try:
                    findings = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    sys.exit(1)
            else:
                # Try JSONL (one JSON object per line)
                for i, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        findings.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i}: {e}")
                        sys.exit(1)

            if not findings:
                print(f"No findings found in {file_path}")
                sys.exit(1)

            print(f"Found {len(findings)} findings in {file_path}")

            if args.dry_run:
                for i, f in enumerate(findings, 1):
                    f_type = f.get("type", "discovery")
                    f_content = f.get("content", "")[:80]
                    print(f"{i}. [{f_type}] {f_content}...")
                print("\n(dry-run mode, nothing added)")
            else:
                added = 0
                skipped = 0
                for f in findings:
                    try:
                        finding_id = kb.add(
                            content=f.get("content", ""),
                            finding_type=f.get("type", "discovery"),
                            project=args.project or f.get("project"),
                            sprint=args.sprint or f.get("sprint"),
                            tags=f.get("tags"),
                            evidence=f.get("evidence"),
                            check_duplicate=not args.force,
                        )
                        print(f"Added: {finding_id}")
                        added += 1
                    except ValueError as e:
                        if "Similar finding" in str(e):
                            print(f"Skipped (duplicate): {f.get('content', '')[:50]}...")
                            skipped += 1
                        else:
                            raise
                print(f"\nBatch ingested {added} findings, skipped {skipped} duplicates.")

        elif args.command == "notation":
            if args.notation_command == "add":
                notation_id = kb.notation_add(
                    symbol=args.symbol,
                    meaning=args.meaning,
                    project=args.project,
                    domain=args.domain,
                )
                print(f"Added: {notation_id}")
                print(f"Symbol: {args.symbol}")
                print(f"Meaning: {args.meaning}")

            elif args.notation_command == "update":
                if not args.old_symbol and not args.notation_id:
                    print("Error: Must provide --old-symbol or --id")
                    sys.exit(1)
                updated_id = kb.notation_update(
                    new_symbol=args.new_symbol,
                    old_symbol=args.old_symbol,
                    notation_id=args.notation_id,
                    meaning=args.meaning,
                    reason=args.reason,
                    project=args.project,
                )
                print(f"Updated: {updated_id}")
                print(f"New symbol: {args.new_symbol}")

            elif args.notation_command == "list":
                results = kb.notation_list(
                    project=args.project,
                    domain=args.domain,
                )
                if not results:
                    print("No notations recorded.")
                else:
                    for n in results:
                        meta = []
                        if n.get("project"):
                            meta.append(n["project"])
                        if n.get("domain"):
                            meta.append(n["domain"])
                        meta_str = f" [{', '.join(meta)}]" if meta else ""
                        print(f"{n['current_symbol']} → {n['meaning']}{meta_str}")
                        print(f"  ID: {n['id']}")

            elif args.notation_command == "search":
                results = kb.notation_search(
                    query=args.query,
                    project=args.project,
                    domain=args.domain,
                )
                if not results:
                    print("No notations found.")
                else:
                    for n in results:
                        meta = []
                        if n.get("project"):
                            meta.append(f"project={n['project']}")
                        if n.get("domain"):
                            meta.append(f"domain={n['domain']}")
                        meta_str = f" ({', '.join(meta)})" if meta else ""
                        print(f"{n['id']}{meta_str}")
                        print(f"  Symbol: {n['current_symbol']}")
                        print(f"  Meaning: {n['meaning']}")

            elif args.notation_command == "history":
                history = kb.notation_history(args.id)
                if not history:
                    print(f"No history for: {args.id}")
                else:
                    print(f"History for {args.id}:\n")
                    for h in history:
                        print(f"{h['changed_at']}: {h['old_symbol']} → {h['new_symbol']}")
                        if h.get("reason"):
                            print(f"  Reason: {h['reason']}")

            elif args.notation_command == "get":
                notation = kb.notation_get(args.id)
                if not notation:
                    print(f"Notation not found: {args.id}")
                    sys.exit(1)
                print(f"ID: {notation['id']}")
                print(f"Symbol: {notation['current_symbol']}")
                print(f"Meaning: {notation['meaning']}")
                if notation.get("project"):
                    print(f"Project: {notation['project']}")
                if notation.get("domain"):
                    print(f"Domain: {notation['domain']}")
                print(f"Created: {notation['created_at']}")
                print(f"Updated: {notation['updated_at']}")
                if notation.get("history"):
                    print(f"\nHistory ({len(notation['history'])} changes):")
                    for h in notation["history"]:
                        print(f"  {h['changed_at']}: {h['old_symbol']} → {h['new_symbol']}")

            elif args.notation_command == "delete":
                if not args.force:
                    notation = kb.notation_get(args.id)
                    if notation:
                        print(f"Symbol: {notation['current_symbol']}")
                        print(f"Meaning: {notation['meaning']}")
                        confirm = input("\nDelete this notation? [y/N] ")
                        if confirm.lower() != "y":
                            print("Cancelled.")
                            sys.exit(0)
                if kb.notation_delete(args.id):
                    print(f"Deleted: {args.id}")
                else:
                    print(f"Notation not found: {args.id}")
                    sys.exit(1)

        elif args.command == "error":
            if args.error_command == "add":
                result = kb.error_add(
                    signature=args.signature,
                    error_type=args.type,
                    project=args.project,
                )
                if result.get("is_new"):
                    print(f"Recorded: {result['id']}")
                else:
                    print(f"Updated: {result['id']} (occurrence #{result['occurrence_count']})")
                if result.get("normalized"):
                    print(f"  [auto] Signature normalized")

            elif args.error_command == "link":
                if kb.error_link(args.error_id, args.finding_id, verified=args.verified):
                    print(f"Linked: {args.error_id} → {args.finding_id}")
                    if args.verified:
                        print("  (marked as verified)")
                else:
                    print("Link already exists or invalid IDs")
                    sys.exit(1)

            elif args.error_command == "verify":
                if kb.error_verify(args.error_id, args.finding_id):
                    print(f"Verified: {args.error_id} → {args.finding_id}")
                else:
                    print("Link not found")
                    sys.exit(1)

            elif args.error_command == "get":
                error = kb.error_get(args.id)
                if not error:
                    print(f"Error not found: {args.id}")
                    sys.exit(1)
                print(f"ID: {error['id']}")
                print(f"Signature: {error['signature']}")
                if error.get("error_type"):
                    print(f"Type: {error['error_type']}")
                if error.get("project"):
                    print(f"Project: {error['project']}")
                print(f"Occurrences: {error['occurrence_count']}")
                print(f"First seen: {error['first_seen']}")
                print(f"Last seen: {error['last_seen']}")
                if error["solutions"]:
                    print(f"\nSolutions ({len(error['solutions'])}):")
                    for s in error["solutions"]:
                        verified = " [VERIFIED]" if s["verified"] else ""
                        print(f"  [{s['type'].upper()}]{verified} {s['finding_id']}")
                        print(f"    {s['content'][:100]}...")

            elif args.error_command == "search":
                results = kb.error_search(
                    query=args.query,
                    project=args.project,
                )
                if not results:
                    print("No errors found.")
                else:
                    for e in results:
                        meta = f" [{e['project']}]" if e.get("project") else ""
                        print(f"{e['id']}{meta} (×{e['occurrence_count']})")
                        print(f"  {e['signature'][:100]}...")

            elif args.error_command == "list":
                results = kb.error_list(
                    project=args.project,
                    error_type=args.type,
                    limit=args.limit,
                )
                if not results:
                    print("No errors recorded.")
                else:
                    for e in results:
                        meta = f" [{e['project']}]" if e.get("project") else ""
                        print(f"{e['id']}{meta} (×{e['occurrence_count']})")
                        print(f"  {e['signature'][:100]}...")

            elif args.error_command == "solutions":
                solutions = kb.error_solutions(args.id)
                if not solutions:
                    print(f"No solutions for: {args.id}")
                else:
                    print(f"Solutions for {args.id}:\n")
                    for s in solutions:
                        verified = " [VERIFIED]" if s["verified"] else ""
                        print(f"[{s['type'].upper()}]{verified} {s['finding_id']}")
                        print(f"  {s['content'][:100]}...")
                        print()

            elif args.error_command == "delete":
                if not args.force:
                    error = kb.error_get(args.id)
                    if error:
                        print(f"Signature: {error['signature']}")
                        print(f"Occurrences: {error['occurrence_count']}")
                        confirm = input("\nDelete this error? [y/N] ")
                        if confirm.lower() != "y":
                            print("Cancelled.")
                            sys.exit(0)
                if kb.error_delete(args.id):
                    print(f"Deleted: {args.id}")
                else:
                    print(f"Error not found: {args.id}")
                    sys.exit(1)

        elif args.command == "bulk":
            if args.bulk_command == "tag":
                tags = [t.strip() for t in args.add_tags.split(",") if t.strip()]
                if not tags:
                    print("Error: No valid tags provided")
                    sys.exit(1)
                result = kb.bulk_add_tags(args.ids, tags)
                print(f"Updated: {result['updated']} findings")
                if result["skipped"] > 0:
                    print(f"Skipped: {result['skipped']} (not found)")

            elif args.bulk_command == "consolidate":
                tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
                result = kb.consolidate_cluster(
                    finding_ids=args.ids,
                    summary=args.summary,
                    reason=args.reason,
                    finding_type=args.type,
                    tags=tags,
                    evidence=args.evidence,
                )
                print(f"Created: {result['new_id']}")
                print(f"Superseded: {result['superseded_count']} findings")
                if result["skipped"] > 0:
                    print(f"Skipped: {result['skipped']} (not found or already superseded)")

        elif args.command == "doc":
            if args.doc_command == "add":
                doc_id = kb.doc_add(
                    title=args.title,
                    doc_type=args.doc_type,
                    url=args.url,
                    project=args.project,
                    summary=args.summary,
                )
                print(f"Added: {doc_id}")
                print(f"Title: {args.title}")
                print(f"Type: {args.doc_type}")

            elif args.doc_command == "get":
                doc = kb.doc_get(args.id)
                if not doc:
                    print(f"Document not found: {args.id}")
                    sys.exit(1)
                print(f"ID: {doc['id']}")
                print(f"Title: {doc['title']}")
                print(f"Type: {doc['doc_type']}")
                if doc.get("url"):
                    print(f"URL: {doc['url']}")
                if doc.get("project"):
                    print(f"Project: {doc['project']}")
                print(f"Status: {doc['status']}")
                if doc.get("summary"):
                    print(f"Summary: {doc['summary']}")
                print(f"Citations: {doc['citation_count']}")
                print(f"Created: {doc['created_at']}")
                if doc.get("superseded_by"):
                    print(f"Superseded by: {doc['superseded_by']}")

            elif args.doc_command == "list":
                results = kb.doc_list(
                    project=args.project,
                    doc_type=args.doc_type,
                    include_superseded=args.include_superseded,
                    limit=args.limit,
                )
                if not results:
                    print("No documents found.")
                else:
                    for d in results:
                        meta = [d["doc_type"]]
                        if d.get("project"):
                            meta.append(d["project"])
                        if d["status"] != "active":
                            meta.append(d["status"].upper())
                        print(f"{d['id']} [{', '.join(meta)}]")
                        print(f"  {d['title']}")

            elif args.doc_command == "search":
                results = kb.doc_search(
                    query=args.query,
                    project=args.project,
                )
                if not results:
                    print("No documents found.")
                else:
                    for d in results:
                        meta = [d["doc_type"]]
                        if d.get("project"):
                            meta.append(d["project"])
                        print(f"{d['id']} [{', '.join(meta)}]")
                        print(f"  {d['title']}")
                        if d.get("summary"):
                            print(f"  {d['summary'][:80]}...")

            elif args.doc_command == "cite":
                if kb.doc_cite(args.finding_id, args.doc_id, args.citation_type, args.notes):
                    print(f"Linked: {args.finding_id} → {args.doc_id}")
                    print(f"  Type: {args.citation_type}")
                else:
                    print("Citation already exists or invalid IDs")
                    sys.exit(1)

            elif args.doc_command == "citations":
                citations = kb.doc_citations(args.id)
                if not citations:
                    print(f"No citations for: {args.id}")
                else:
                    print(f"Findings citing {args.id}:\n")
                    for c in citations:
                        print(f"[{c['type'].upper()}] {c['finding_id']} ({c['citation_type']})")
                        print(f"  {c['content'][:100]}...")
                        if c.get("notes"):
                            print(f"  Notes: {c['notes']}")
                        print()

            elif args.doc_command == "finding-docs":
                docs = kb.finding_docs(args.id)
                if not docs:
                    print(f"No documents cited by: {args.id}")
                else:
                    print(f"Documents cited by {args.id}:\n")
                    for d in docs:
                        print(f"[{d['doc_type'].upper()}] {d['document_id']} ({d['citation_type']})")
                        print(f"  {d['title']}")
                        if d.get("url"):
                            print(f"  URL: {d['url']}")
                        print()

            elif args.doc_command == "supersede":
                if kb.doc_supersede(args.doc_id, args.new_doc_id):
                    print(f"Superseded: {args.doc_id} → {args.new_doc_id}")
                else:
                    print(f"Document not found: {args.doc_id}")
                    sys.exit(1)

            elif args.doc_command == "delete":
                if not args.force:
                    doc = kb.doc_get(args.id)
                    if doc:
                        print(f"Title: {doc['title']}")
                        print(f"Type: {doc['doc_type']}")
                        print(f"Citations: {doc['citation_count']}")
                        confirm = input("\nDelete this document? [y/N] ")
                        if confirm.lower() != "y":
                            print("Cancelled.")
                            sys.exit(0)
                if kb.doc_delete(args.id):
                    print(f"Deleted: {args.id}")
                else:
                    print(f"Document not found: {args.id}")
                    sys.exit(1)

        elif args.command == "reembed":
            print("Re-generating embeddings for all findings...")
            result = kb.reembed_all()
            print(f"Updated: {result['updated']}/{result['total']}")
            if result['failed'] > 0:
                print(f"Failed: {result['failed']}")

        elif args.command == "backfill-summaries":
            print("Generating summaries for findings without them...")
            result = kb.backfill_summaries(
                project=args.project,
                batch_size=args.batch_size,
            )
            print(f"\nUpdated: {result['updated']}/{result['processed']}")
            if result['failed'] > 0:
                print(f"Failed: {result['failed']}")
            if result['remaining'] > 0:
                print(f"Remaining: {result['remaining']} (run again to continue)")

        elif args.command == "docstrings":
            all_findings = []
            for file_path in args.files:
                if not file_path.exists():
                    print(f"Warning: File not found: {file_path}")
                    continue
                if not file_path.suffix == ".py":
                    print(f"Warning: Skipping non-Python file: {file_path}")
                    continue
                findings = parse_script_findings(file_path)
                if findings:
                    all_findings.extend(findings)

            if not all_findings:
                print("No docstrings found in specified files.")
                sys.exit(1)

            print(f"Found {len(all_findings)} docstrings:\n")

            if args.dry_run:
                for i, f in enumerate(all_findings, 1):
                    tags = f.get("tags", [])
                    tag_str = f" [{', '.join(tags)}]" if tags else ""
                    print(f"{i}. [{f['type']}]{tag_str} {f['content'][:80]}...")
                print("\n(dry-run mode, nothing added)")
            else:
                added = 0
                skipped = 0
                for f in all_findings:
                    try:
                        finding_id = kb.add(
                            content=f["content"],
                            finding_type=f["type"],
                            project=args.project or f.get("project"),
                            sprint=args.sprint or f.get("sprint"),
                            tags=f.get("tags"),
                            evidence=f.get("evidence"),
                            check_duplicate=not args.force,
                        )
                        print(f"Added: {finding_id}")
                        added += 1
                    except ValueError as e:
                        if "Similar finding" in str(e):
                            print(f"Skipped (duplicate): {f['content'][:50]}...")
                            skipped += 1
                        else:
                            raise
                print(f"\nIngested {added} docstrings, skipped {skipped} duplicates.")

        elif args.command == "script":
            if args.script_command == "add":
                script_id = kb.script_add(
                    path=str(args.file),
                    purpose=args.purpose,
                    project=args.project,
                    language=args.language,
                    store_content=not args.no_content,
                )
                print(f"Registered: {script_id}")
                print(f"  File: {args.file}")
                print(f"  Purpose: {args.purpose}")

            elif args.script_command == "get":
                script = kb.script_get(args.id)
                if not script:
                    print(f"Script not found: {args.id}")
                    sys.exit(1)
                print(f"ID: {script['id']}")
                print(f"File: {script['filename']}")
                print(f"Path: {script['path']}")
                print(f"Purpose: {script['purpose']}")
                if script.get("project"):
                    print(f"Project: {script['project']}")
                print(f"Language: {script['language']}")
                print(f"Created: {script['created_at']}")
                if args.show_content and script.get("content"):
                    print(f"\n--- Content ---\n{script['content']}")

            elif args.script_command == "list":
                results = kb.script_list(
                    project=args.project,
                    language=args.language,
                    limit=args.limit,
                )
                if not results:
                    print("No scripts found.")
                else:
                    print(f"Found {len(results)} scripts:\n")
                    for s in results:
                        meta = []
                        if s.get("project"):
                            meta.append(s["project"])
                        meta.append(s.get("language", "unknown"))
                        meta_str = f" [{', '.join(meta)}]" if meta else ""
                        print(f"  {s['id']}{meta_str}")
                        print(f"    {s['filename']}: {s['purpose'][:60]}...")

            elif args.script_command == "search":
                results = kb.script_search(
                    query=args.query,
                    project=args.project,
                    limit=args.limit,
                )
                if not results:
                    print("No scripts found.")
                else:
                    print(f"Found {len(results)} scripts:\n")
                    for s in results:
                        sim = s.get("similarity", 0)
                        print(f"  {s['id']} (sim: {sim:.2f})")
                        print(f"    {s['filename']}: {s['purpose'][:60]}...")

            elif args.script_command == "link":
                kb.script_link_finding(
                    finding_id=args.finding_id,
                    script_id=args.script_id,
                    relationship=args.relationship,
                )
                print(f"Linked {args.finding_id} -> {args.script_id} ({args.relationship})")

            elif args.script_command == "findings":
                findings = kb.script_findings(args.id)
                if not findings:
                    print(f"No findings linked to script: {args.id}")
                else:
                    print(f"Findings for script {args.id}:\n")
                    for f in findings:
                        rel = f.get("relationship", "generated_by").upper()
                        print(f"  [{rel}] {f['id']} [{f['type']}]")
                        print(f"    {f['content'][:70]}...")

            elif args.script_command == "delete":
                if not args.force:
                    script = kb.script_get(args.id)
                    if not script:
                        print(f"Script not found: {args.id}")
                        sys.exit(1)
                    print(f"About to delete: {script['filename']}")
                    confirm = input("Type 'yes' to confirm: ")
                    if confirm.lower() != "yes":
                        print("Aborted.")
                        sys.exit(0)
                if kb.script_delete(args.id):
                    print(f"Deleted: {args.id}")
                else:
                    print(f"Script not found: {args.id}")
                    sys.exit(1)

        elif args.command == "reconcile":
            from kb_reconcile import DocumentReconciler
            if not args.doc_dir.is_dir():
                print(f"Error: {args.doc_dir} is not a directory", file=sys.stderr)
                sys.exit(1)
            reconciler = DocumentReconciler(kb, args.project)
            report = reconciler.reconcile(args.doc_dir)

            # Handle export/import of missing claims
            claim_types = args.claim_types.split(",") if args.claim_types else None
            if args.export_missing or args.import_missing:
                findings = reconciler.export_missing_json(
                    report,
                    claim_types=claim_types,
                    min_length=args.min_length,
                )
                if args.export_missing:
                    args.export_missing.write_text(json.dumps(findings, indent=2))
                    print(f"Exported {len(findings)} claims to {args.export_missing}")
                elif args.import_missing:
                    added, skipped = 0, 0
                    for f in findings:
                        try:
                            kb.add(
                                content=f["content"],
                                finding_type=f["type"],
                                project=f.get("project"),
                                tags=f.get("tags"),
                                evidence=f.get("evidence"),
                            )
                            added += 1
                        except ValueError:
                            skipped += 1
                    print(f"Imported {added} claims, skipped {skipped} duplicates")
            else:
                output = reconciler.format_report(report)
                if args.output:
                    args.output.write_text(output)
                    print(f"Report written to {args.output}")
                else:
                    print(output)

        elif args.command == "notation" and args.notation_command == "audit":
            from kb_notation_audit import NotationAuditor
            if not args.doc_dir.is_dir():
                print(f"Error: {args.doc_dir} is not a directory", file=sys.stderr)
                sys.exit(1)
            auditor = NotationAuditor(kb, args.project)
            report = auditor.audit(args.doc_dir)
            output = auditor.format_report(report)
            if args.output:
                args.output.write_text(output)
                print(f"Report written to {args.output}")
            else:
                print(output)

        elif args.command == "llm":
            if args.llm_command == "suggest-tags":
                tags = kb.suggest_tags(args.content, project=args.project)
                if tags:
                    print("Suggested tags:", ", ".join(tags))
                else:
                    print("No tags suggested (LLM unavailable or failed)")

            elif args.llm_command == "classify":
                finding_type = kb.classify_finding_type(args.content)
                print(f"Suggested type: {finding_type}")

            elif args.llm_command == "duplicates":
                duplicates = kb.detect_duplicates(
                    args.content, project=args.project, threshold=args.threshold
                )
                if duplicates:
                    print(f"Found {len(duplicates)} potential duplicate(s):")
                    for d in duplicates:
                        print(f"  {d['id']} (sim: {d.get('similarity', 0):.2f})")
                        print(f"    {d['content'][:100]}...")
                else:
                    print("No duplicates found")

            elif args.llm_command == "normalize-error":
                signature = kb.normalize_error_signature(args.error)
                print(f"Normalized: {signature}")

            elif args.llm_command == "xref":
                finding = kb.get(args.finding_id)
                if not finding:
                    print(f"Finding not found: {args.finding_id}")
                else:
                    xrefs = kb.suggest_cross_references(
                        args.finding_id, finding["content"], project=args.project
                    )
                    if xrefs["findings"]:
                        print("Related findings:")
                        for f in xrefs["findings"]:
                            print(f"  {f['id']} (sim: {f['similarity']:.2f})")
                    if xrefs["scripts"]:
                        print("Related scripts:")
                        for s in xrefs["scripts"]:
                            print(f"  {s['id']}: {s['filename']}")
                    if xrefs["docs"]:
                        print("Related documents:")
                        for d in xrefs["docs"]:
                            print(f"  {d['id']}: {d['title']}")
                    if not any(xrefs.values()):
                        print("No cross-references found")

            elif args.llm_command == "summarize":
                summary = kb.summarize_evidence(args.evidence, max_length=args.max_length)
                print(summary)

            elif args.llm_command == "detect-notations":
                notations = kb.detect_notations(args.content, project=args.project)
                if notations:
                    print(f"Found {len(notations)} notation(s):")
                    for n in notations:
                        status = "[exists]" if n["exists"] else "[new]"
                        print(f"  {status} {n['symbol']}: {n['meaning']}")
                else:
                    print("No notations detected")

            elif args.llm_command == "extract-claims":
                claims = kb.extract_claims(args.text)
                if claims:
                    print(f"Extracted {len(claims)} claim(s):")
                    for i, c in enumerate(claims, 1):
                        print(f"  {i}. {c}")
                else:
                    print("No claims extracted")

            elif args.llm_command == "consolidate":
                clusters = kb.suggest_consolidation(project=args.project, limit=args.limit)
                if clusters:
                    print(f"Found {len(clusters)} cluster(s) to potentially consolidate:\n")
                    for i, c in enumerate(clusters, 1):
                        print(f"Cluster {i}:")
                        for m in c["members"]:
                            print(f"  - {m['id']}: {m['content'][:60]}...")
                        print(f"Analysis: {c['analysis']}\n")
                else:
                    print("No consolidation opportunities found")

            else:
                print("Unknown llm subcommand. Use: suggest-tags, classify, duplicates, normalize-error, xref, summarize, detect-notations, extract-claims, consolidate")

    finally:
        kb.close()


if __name__ == "__main__":
    main()
