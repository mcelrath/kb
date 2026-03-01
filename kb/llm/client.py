"""
LLM Client

Handles communication with LLM endpoint for query expansion and completions.
"""

import json
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..constants import DEFAULT_LLM_URL


class LLMClient:
    """Client for LLM completions with query expansion caching."""

    llm_url: str
    _expansion_cache: dict[str, str]

    def __init__(self, llm_url: str = DEFAULT_LLM_URL):
        self.llm_url = llm_url
        self._expansion_cache = {}

    def complete(
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
        if not self.llm_url:
            return None

        try:
            if use_chat:
                # Use chat completion API for better format adherence
                chat_url = self.llm_url.replace("/completion", "/v1/chat/completions")
                messages: list[dict[str, str]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                request_body: dict[str, object] = {
                    "messages": messages,
                    "max_tokens": -1,
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
                    msg = data["choices"][0]["message"]
                    content = (msg.get("content") or "").strip()
                    if not content:
                        # Thinking models (e.g. Qwen3.5) put output in reasoning_content
                        reasoning = (msg.get("reasoning_content") or "").strip()
                        if reasoning:
                            content = self._extract_from_thinking(reasoning)
                    return self._strip_thinking(content)
            else:
                # Raw completion API
                if stop is None:
                    stop = ["\n\n"]
                req = Request(
                    self.llm_url,
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
        except (URLError, TimeoutError, KeyError, json.JSONDecodeError):
            return None

    def _strip_thinking(self, text: str) -> str:
        """Remove <think>...</think> blocks from LLM output."""
        if not text:
            return text
        # Remove thinking blocks (handles multiline)
        result = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        return result.strip()

    def _extract_from_thinking(self, reasoning: str) -> str:
        """Extract the actual answer from a thinking model's reasoning_content.

        Thinking models like Qwen3.5 may wrap their answer in JSON within
        the reasoning_content field when content is empty.
        """
        # Try to parse as JSON and extract answer
        try:
            parsed = json.loads(reasoning)
            if isinstance(parsed, dict):
                for key in ["answer", "response", "output", "result", "content", "text"]:
                    if key in parsed and isinstance(parsed[key], str) and parsed[key].strip():
                        return parsed[key].strip()
        except json.JSONDecodeError:
            pass
        # If not JSON, return the reasoning text itself (model may have just output text)
        return reasoning

    def extract_text_from_json(self, text: str, keys: list[str] | None = None) -> str:
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

    def expand_query(
        self,
        query: str,
        project: str | None = None,
        embedding_url: str | None = None,
        verbose: bool = False,
    ) -> str:
        """Expand a search query using a local LLM for better recall.

        Uses few-shot prompting optimized for technical/scientific content:
        - Expands acronyms (FMHA -> Flash Multi-Head Attention)
        - Preserves compound terms (vector similarity -> "vector similarity")
        - Adds domain-specific related terms based on project context
        - Generates synonyms and alternative phrasings

        Args:
            query: The original search query
            project: Optional project name for domain context
            embedding_url: Optional embedding URL for fallback LLM URL derivation
            verbose: If True, print the expanded query to stderr

        Returns:
            Expanded query string combining original + generated terms
        """
        import sys

        # Check cache first (keyed by query + project)
        cache_key = f"{query}|{project or ''}"
        if cache_key in self._expansion_cache:
            expanded = self._expansion_cache[cache_key]
            if verbose:
                print(f"[cached] Expanded: {expanded}", file=sys.stderr)
            return expanded

        # Determine LLM URL: explicit > derived from embedding URL
        llm_url = self.llm_url
        if not llm_url and embedding_url:
            base = embedding_url.rsplit("/", 1)[0]
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
1. Expand acronyms: FMHA -> "Flash Multi-Head Attention" FMHA
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
                expansion = self.extract_text_from_json(expansion, keys=["expansion", "terms", "output", "result"])
                # Clean up: remove any newlines, extra whitespace
                expansion = " ".join(expansion.split())
                if expansion:
                    expanded = f"{query} {expansion}"
                    self._expansion_cache[cache_key] = expanded
                    if verbose:
                        print(f"Expanded: {expanded}", file=sys.stderr)
                    return expanded
        except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            if verbose:
                print(f"Warning: Query expansion failed ({e})", file=sys.stderr)

        # Cache and return original on failure
        self._expansion_cache[cache_key] = query
        return query
