#!/usr/bin/env python3
"""Compose a persona's injected text from archetype (L1) + augmentation (L2) + instance body.

Called by hooks/scripts/session-persona.sh in place of a flat `cat` of the persona file.
Layer order: L1 archetype -> L2 augmentation -> persona instance body.

Frontmatter keys on the persona file:
  archetype: <name>     -> $plugin_root/skills/persona/archetypes/<name>.md   (L1)
  augmentation: <name>  -> <persona_dir>/<name>.md                            (L2, optional)

Contract:
  - augmentation present  -> L2 = that file's body; persona file's own body = instance body (last).
  - augmentation absent    -> persona file's own body serves as L2 (emitted once, no dup).
  - archetype absent       -> only the persona body (the legacy flat behavior).
  - a DECLARED archetype/augmentation file that is MISSING -> skip with a stderr warning and
    emit whatever remains (partial > blank). The caller hook falls back to a flat cat only on
    a non-zero exit, so missing-file is intentionally NOT fatal.

Frontmatter is parsed by the real parser in kb.ingest.personas (no second parser).
"""

import sys
from pathlib import Path

# hooks/scripts/lib/persona_compose.py -> plugin root is parents[3]
_PKG_ROOT = Path(__file__).resolve().parents[3]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from kb.ingest.personas import _parse_frontmatter


def _strip_frontmatter(text: str) -> str:
    """Return the body after a leading ---...--- frontmatter block (or the whole text if none).

    Slices AFTER the closing '---' LINE. Does NOT use lstrip('---\\n'): that strips the
    character set {'-', '\\n'} and would silently eat a body that starts with a markdown
    list dash ('- item') — the bug in kb/ingest/personas.py:181.
    """
    if not text.startswith("---"):
        return text.strip()
    close = text.find("\n---", 3)
    if close == -1:
        return text.strip()
    nl = text.find("\n", close + 1)
    if nl == -1:
        return ""
    return text[nl + 1:].strip()


def _load_body(path: Path, label: str) -> str:
    """Read path and return its frontmatter-stripped body, or '' (with a warning) if missing."""
    if not path.is_file():
        print(f"persona_compose: {label} file not found: {path}", file=sys.stderr)
        return ""
    return _strip_frontmatter(path.read_text(encoding="utf-8", errors="replace"))


def compose(persona_file: str, plugin_root: str, persona_dir: str) -> str:
    text = Path(persona_file).read_text(encoding="utf-8", errors="replace")
    fm = _parse_frontmatter(text)
    instance_body = _strip_frontmatter(text)

    parts: list[str] = []

    archetype = fm.get("archetype")
    if archetype:
        l1 = Path(plugin_root) / "skills" / "persona" / "archetypes" / f"{archetype}.md"
        body = _load_body(l1, "archetype")
        if body:
            parts.append(body)

    augmentation = fm.get("augmentation")
    if augmentation:
        l2 = _load_body(Path(persona_dir) / f"{augmentation}.md", "augmentation")
        if l2:
            parts.append(l2)
        # the persona file's own body is the instance layer, emitted last
        if instance_body:
            parts.append(instance_body)
    else:
        # no augmentation: the persona body serves as L2 (single emit, no duplication)
        if instance_body:
            parts.append(instance_body)

    return "\n\n".join(parts)


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: persona_compose.py <persona_file> <plugin_root> <persona_dir>", file=sys.stderr)
        return 2
    sys.stdout.write(compose(argv[1], argv[2], argv[3]))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
