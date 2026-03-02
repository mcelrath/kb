"""
Content Analysis

LLM-based content analysis: tagging, classification, validation, summarization.
"""

import json
import re
from typing import Any

from .client import LLMClient
from ..constants import UNICODE_TO_ASCII, ALLOWED_UNICODE, GREEK_MEANINGS


class ContentAnalyzer:
    """LLM-based content analysis for findings."""

    llm_client: LLMClient

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_summary(self, content: str, evidence: str | None = None) -> str | None:
        """Generate a one-line summary for a finding.

        Args:
            content: The finding content to summarize
            evidence: Optional evidence to consider

        Returns:
            A short summary (max ~100 chars) or None on failure
        """
        system_prompt = (
            "You write concise one-line summaries. Output ONLY JSON: "
            '{\"summary\": \"...\"}. No intro phrases. Max 80 chars.'
        )

        # Normalize Unicode math symbols to ASCII to avoid confusing the LLM
        text = content[:500]
        # Strip emotional/error-like prefixes that confuse the JSON-only LLM
        text = re.sub(
            r'^(?:CRITICAL\s+)?(?:CORRECTION|ERROR|FATAL\s+FLAW|WARNING|NOTE|IMPORTANT):\s*',
            '', text, flags=re.IGNORECASE
        )
        for uc, asc in UNICODE_TO_ASCII.items():
            text = text.replace(uc, asc)
        # Strip remaining non-ASCII
        text = text.encode('ascii', 'ignore').decode('ascii')

        if evidence:
            ev = evidence[:200].encode('ascii', 'ignore').decode('ascii')
            text += f"\nEvidence: {ev}"

        prompt = f"Summarize in ONE technical line (max 80 chars):\n{text}"

        result = self.llm_client.complete(
            prompt,
            max_tokens=150,
            temperature=0.2,
            system_prompt=system_prompt,
            timeout=30,
        )

        if result:
            # Extract from JSON wrapper using consolidated helper
            result = self.llm_client.extract_text_from_json(
                result, keys=["summary", "result", "text", "output"]
            )

            if result:
                # Clean: strip quotes, remove garbage
                summary = result.strip().strip('"').strip("'")
                summary = re.sub(r'\\u[0-9a-fA-F]{4}', '', summary)
                summary = re.sub(r'[\x00-\x1f\x7f]', '', summary)
                summary = ''.join(c for c in summary if ord(c) < 128 or c in ALLOWED_UNICODE)
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

    def suggest_tags(
        self,
        content: str,
        existing_tags: set[str] | None = None,
    ) -> list[str]:
        """Suggest tags for a finding based on its content using LLM."""
        existing_list = ", ".join(sorted(existing_tags)[:30]) if existing_tags else "none yet"

        system_prompt = "You suggest tags for knowledge base findings. Return JSON with a 'tags' array."

        prompt = f"""Suggest 2-5 tags for this finding. Return JSON: {{"tags": ["tag1", "tag2", ...]}}

Existing tags to prefer: {existing_list}
Status tags: proven, heuristic, open-problem
Importance tags: core-result, technique, detail
Dimension tags: dim-2, dim-4, dim-8

Content: {content[:500]}"""

        result = self.llm_client.complete(
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
        result = self.llm_client.extract_text_from_json(result, keys=["tags"])

        # Parse comma-separated tags, handle potential formatting issues
        result = result.strip().strip('`').strip()
        if result.startswith("Tags:"):
            result = result[5:]

        # Clean and validate tags
        tags = []
        for t in result.split(","):
            t = t.strip().lower().replace(" ", "-")
            t = t.strip('"\'').replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
            if t and len(t) < 30 and re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$', t):
                tags.append(t)
        return tags[:5]

    def classify_type(self, content: str) -> str:
        """Suggest finding type based on content."""
        system_prompt = "You classify findings. Return JSON with 'type' field."

        prompt = f"""Classify this finding. Return JSON: {{"type": "<type>"}}

Types:
- success: Verified working approach
- failure: Something that doesn't work
- discovery: New understanding or insight
- experiment: Inconclusive, needs more work

Content: {content[:400]}"""

        result = self.llm_client.complete(
            prompt,
            max_tokens=50,
            temperature=0.1,
            system_prompt=system_prompt,
            timeout=30,
            json_mode=True,
        )
        if result:
            result = self.llm_client.extract_text_from_json(result, keys=["type", "classification", "result"])
            result = result.lower().strip().split()[0] if result else ""
            if result in ("success", "failure", "discovery", "experiment"):
                return result

        return "discovery"

    def normalize_error_signature(self, error_text: str) -> str:
        """Normalize an error message to a canonical signature for matching."""
        system_prompt = "You extract error signatures. Return JSON with 'signature' field."

        prompt = f"""Extract a canonical error signature. Return JSON: {{"signature": "<normalized error>"}}

Rules:
- Remove paths, line numbers, memory addresses
- Keep error type and core message
- Use placeholders: <N> for numbers, <PATH> for paths, <ADDR> for addresses

Error: {error_text[:500]}"""

        result = self.llm_client.complete(
            prompt,
            max_tokens=150,
            temperature=0.1,
            system_prompt=system_prompt,
            timeout=30,
            json_mode=True,
        )
        if result:
            result = self.llm_client.extract_text_from_json(result, keys=["signature", "error", "result"])
            if result:
                return result.strip().split('\n')[0].strip()

        # Fallback: basic normalization
        sig = re.sub(r'/[\w/.-]+', '<PATH>', error_text)
        sig = re.sub(r':\d+', ':<N>', sig)
        sig = re.sub(r'0x[0-9a-fA-F]+', '<ADDR>', sig)
        sig = re.sub(r'\b\d+\b', '<N>', sig)
        return sig[:200]

    def validate_finding(self, content: str, tags: list[str] | None = None) -> dict[str, Any]:
        """LLM-based semantic validation of finding content."""
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

        result = self.llm_client.complete(
            prompt,
            max_tokens=300,
            temperature=0.2,
            system_prompt=system_prompt,
            timeout=60,
            json_mode=True,
        )

        validation: dict[str, Any] = {
            "is_valid": True,
            "quality_score": 3,
            "issues": [],
            "suggestions": [],
        }

        if result:
            result_stripped = result.strip()
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
                except json.JSONDecodeError:
                    pass

        return validation

    def suggest_fix(self, content: str, issues: list[str]) -> str | None:
        """Generate corrected content for a finding with issues."""
        if not issues:
            return None

        system_prompt = """You fix knowledge base findings. Output JSON: {"corrected": "..."}.
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

        result = self.llm_client.complete(
            prompt,
            max_tokens=500,
            temperature=0.3,
            system_prompt=system_prompt,
            timeout=60,
        )

        if result:
            result = self.llm_client.extract_text_from_json(result, keys=["content", "corrected", "fix", "text"])
            if len(result.strip()) > 20:
                return result.strip()
        return None

    def summarize_evidence(self, evidence: str, max_length: int = 200) -> str:
        """Summarize long evidence text."""
        if len(evidence) <= max_length:
            return evidence

        prompt = f"""Summarize this evidence/output concisely, preserving key technical details.
Output JSON: {{"summary": "..."}}.

Evidence:
{evidence[:1500]}"""

        result = self.llm_client.complete(prompt, max_tokens=100, temperature=0.2)
        if result:
            result = self.llm_client.extract_text_from_json(result, keys=["summary", "text", "result"])
            return result[:max_length]

        return evidence[:max_length - 3] + "..."

    def detect_notations(self, content: str, existing_symbols: set[str] | None = None) -> list[dict[str, Any]]:
        """Detect mathematical/physics notations in content that should be tracked.

        Uses hybrid approach: regex extraction + hardcoded meanings for reliability.
        """
        # Extract Greek letters using regex (reliable)
        greek_pattern = r'[α-ωΑ-Ω]'
        found_greek = sorted(set(re.findall(greek_pattern, content)))

        parsed: list[dict[str, Any]] = []
        for letter in found_greek:
            meaning = GREEK_MEANINGS.get(letter, '')
            exists = letter in existing_symbols if existing_symbols else False
            parsed.append({"symbol": letter, "meaning": meaning, "exists": exists})

        return parsed

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

        result = self.llm_client.complete(prompt, max_tokens=500, temperature=0.3, system_prompt=system_prompt, json_mode=True)
        if not result:
            return []

        claims: list[str] = []
        result_stripped = result.strip()

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
            if line and len(line) > 20:
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line:
                    claims.append(line)

        return claims[:10]
