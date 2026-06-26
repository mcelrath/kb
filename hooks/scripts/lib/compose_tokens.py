"""Shared token + input extraction helpers for compose-time check hooks."""
import re


_FRAC_CONTEXT_RE = re.compile(
    r'(?:'
    r'[A-Za-zα-ωΑ-Ω_]\w*\s*=\s*\d{1,4}/\d{1,4}'
    r'|\d{1,4}/\d{1,4}\s*[A-Za-zα-ωΑ-Ω_]'
    r'|\b(?:value|exact|result|equals?|is)\s+\d{1,4}/\d{1,4}'
    r')',
    re.IGNORECASE,
)


def extract_candidate_tokens(text: str) -> list[str]:
    """Extract candidate symbol/quantity tokens from prompt text."""
    candidates: set[str] = set()

    verb_re = re.compile(
        r'\b(?:compute|derive|implement|prove|find|calculate|determine|evaluate|check)\s+'
        r'([A-Za-z_][A-Za-z0-9_]{2,})',
        re.IGNORECASE,
    )
    for m in verb_re.finditer(text):
        candidates.add(m.group(1))

    eq_re = re.compile(r'\b([A-Za-z_]\w*)\s*=\s*[\d/\.]+')
    for m in eq_re.finditer(text):
        tok = m.group(1)
        if len(tok) >= 2:
            candidates.add(tok)

    frac_re = re.compile(r'\b(\d{1,4}/\d{1,4})\b')
    for m in frac_re.finditer(text):
        candidates.add(m.group(1))

    snake_re = re.compile(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,})\b')
    for m in snake_re.finditer(text):
        tok = m.group(1)
        if len(tok) >= 6 and tok not in {
            'the_user', 'for_the', 'in_the', 'to_the', 'of_the', 'with_the',
        }:
            candidates.add(tok)

    camel_re = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b')
    for m in camel_re.finditer(text):
        candidates.add(m.group(1))

    mixed_re = re.compile(r'\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b')
    for m in mixed_re.finditer(text):
        tok = m.group(1)
        if len(tok) >= 3:
            candidates.add(tok)

    upper_re = re.compile(r'\b([A-Z][A-Z0-9_]{2,})\b')
    for m in upper_re.finditer(text):
        candidates.add(m.group(1))

    greek_re = re.compile(r'[α-ωΑ-Ω]')
    for ch in greek_re.findall(text):
        candidates.add(ch)

    return list(candidates)


def extract_fractions(text: str) -> list[str]:
    """Extract 'N/D' style exact fractions that appear in an operator/value context."""
    if not _FRAC_CONTEXT_RE.search(text):
        return []
    return re.findall(r'\b\d{1,4}/\d{1,4}\b', text)


def parse_prompt_text(tool_name: str, ti: dict) -> str:
    """Extract the scannable text from a tool invocation (Agent dispatch or bridge send)."""
    if tool_name in ('Task', 'Agent'):
        return ti.get('prompt', '')
    if tool_name == 'Bash':
        cmd = ti.get('command', '')
        if 'bridge send' not in cmd:
            return ''
        parts = []
        m = re.search(r"<<\s*'?EOF'?\s*\n(.+?)(?:\nEOF\b|\Z)", cmd, re.DOTALL)
        if m:
            parts.append(m.group(1))
        m2 = re.search(r'bridge send\s+\S+\s+"([^"]+)"', cmd)
        if m2:
            parts.append(m2.group(1))
        return '\n'.join(parts)
    return ''
