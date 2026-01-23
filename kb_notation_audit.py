#!/usr/bin/env python3
"""
Notation Audit Tool for Knowledge Base.

Compares notation usage in markdown documents against KB notation entries to identify:
- Notations used in docs but not defined in KB
- KB notations not used in docs (possibly obsolete)
- Definition conflicts (same symbol, different meanings)
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kb import KnowledgeBase

Notation = dict[str, Any]


@dataclass
class NotationUsage:
    """Tracks how a notation is used in documents."""
    symbol: str
    definitions: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)  # "filename:line"
    contexts: list[str] = field(default_factory=list)  # Section headers


@dataclass
class AuditReport:
    """Full notation audit report."""
    missing_from_kb: list[NotationUsage] = field(default_factory=list)
    not_in_docs: list[Notation] = field(default_factory=list)
    conflicts: list[tuple[Notation, NotationUsage]] = field(default_factory=list)
    well_matched: list[tuple[Notation, NotationUsage]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


class NotationAuditor:
    """Audit notation consistency between documents and KB."""

    # Regex patterns for notation extraction
    PATTERNS = {
        # **X** = definition or **X**: definition
        'bold_definition': re.compile(
            r'\*\*([^*]{1,20})\*\*\s*[=:]\s*(.{10,100}?)(?=\n|$)',
            re.MULTILINE
        ),
        # Greek letters with optional subscripts
        'greek': re.compile(
            r'[ΓΔΛΣΩΠΘΦΨΞαβγδεζηθικλμνξπρστυφχψω][₀₁₂₃₄₅₆₇₈₉ᵢⱼₖₗₘₙ]*'
        ),
        # Math constructs like Cl(p,q), SO(n,m), SU(n)
        'math_construct': re.compile(
            r'\b(?:Cl|SO|SU|Sp|GL|U|O|Pin|Spin)\s*\([^)]+\)'
        ),
        # Function notation like Q(x), μ(a,b), N(M)
        'function_notation': re.compile(
            r'\b([A-Z])\s*\(\s*[a-z][a-z,\s]*\)'
        ),
        # Subscript/superscript patterns
        'subscript': re.compile(
            r'[A-Za-z][₀₁₂₃₄₅₆₇₈₉ᵢⱼₖₗₘₙ⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ†‡]+'
        ),
        # LaTeX-style like \Gamma, \lambda
        'latex': re.compile(
            r'\\(?:Gamma|Delta|Lambda|Sigma|Omega|Pi|Theta|Phi|Psi|Xi|'
            r'alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|'
            r'lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)'
        ),  # noqa: E501
    }

    # Section header pattern
    HEADER_PATTERN = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)

    # Common words to exclude from notation detection
    EXCLUDE_WORDS = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 'was',
        'will', 'can', 'has', 'have', 'not', 'but', 'all', 'any', 'each',
        'which', 'where', 'when', 'how', 'why', 'what', 'who', 'then',
        'than', 'only', 'also', 'just', 'more', 'most', 'some', 'such',
        'note', 'see', 'ref', 'fig', 'table', 'section', 'chapter',
    }

    def __init__(self, kb: KnowledgeBase, project: str):
        self.kb = kb
        self.project = project

    def extract_from_doc(self, doc_path: Path) -> dict[str, NotationUsage]:
        """Extract all notation usages from a document."""
        content = doc_path.read_text()
        filename = doc_path.name
        notations: dict[str, NotationUsage] = {}

        # Get section headers for context
        headers = list(self.HEADER_PATTERN.finditer(content))

        def get_section_at(pos: int) -> str:
            for i, h in enumerate(headers):
                if h.start() > pos:
                    return headers[i-1].group(2) if i > 0 else ""
            return headers[-1].group(2) if headers else ""

        def get_line_num(pos: int) -> int:
            return content[:pos].count('\n') + 1

        def add_notation(symbol: str, pos: int, definition: str = ""):
            symbol = symbol.strip()
            if len(symbol) < 1 or symbol.lower() in self.EXCLUDE_WORDS:
                return

            if symbol not in notations:
                notations[symbol] = NotationUsage(symbol=symbol)

            location = f"{filename}:{get_line_num(pos)}"
            if location not in notations[symbol].locations:
                notations[symbol].locations.append(location)

            if definition and definition not in notations[symbol].definitions:
                notations[symbol].definitions.append(definition)

            context = get_section_at(pos)
            if context and context not in notations[symbol].contexts:
                notations[symbol].contexts.append(context)

        # Extract bold definitions (highest priority - explicit definitions)
        for match in self.PATTERNS['bold_definition'].finditer(content):
            symbol = match.group(1).strip()
            definition = match.group(2).strip()
            add_notation(symbol, match.start(), definition)

        # Extract Greek letters
        for match in self.PATTERNS['greek'].finditer(content):
            add_notation(match.group(), match.start())

        # Extract math constructs
        for match in self.PATTERNS['math_construct'].finditer(content):
            add_notation(match.group(), match.start())

        # Extract function notations
        for match in self.PATTERNS['function_notation'].finditer(content):
            # Get the full match, not just the letter
            add_notation(match.group(), match.start())

        # Extract subscript patterns
        for match in self.PATTERNS['subscript'].finditer(content):
            symbol = match.group()
            if len(symbol) >= 2:  # Only multi-char subscript patterns
                add_notation(symbol, match.start())

        # Extract LaTeX notations
        for match in self.PATTERNS['latex'].finditer(content):
            add_notation(match.group(), match.start())

        return notations

    def extract_from_all_docs(self, doc_dir: Path) -> dict[str, NotationUsage]:
        """Extract notations from all markdown files in directory."""
        all_notations: dict[str, NotationUsage] = {}

        for doc_path in doc_dir.rglob('*.md'):
            if doc_path.name.startswith('.'):
                continue
            try:
                doc_notations = self.extract_from_doc(doc_path)
                for symbol, usage in doc_notations.items():
                    if symbol not in all_notations:
                        all_notations[symbol] = usage
                    else:
                        # Merge usages
                        existing = all_notations[symbol]
                        for loc in usage.locations:
                            if loc not in existing.locations:
                                existing.locations.append(loc)
                        for defn in usage.definitions:
                            if defn not in existing.definitions:
                                existing.definitions.append(defn)
                        for ctx in usage.contexts:
                            if ctx not in existing.contexts:
                                existing.contexts.append(ctx)
            except Exception as e:
                print(f"Warning: Could not process {doc_path}: {e}", file=sys.stderr)

        return all_notations

    def audit(self, doc_dir: Path) -> AuditReport:
        """Run full notation audit."""
        report = AuditReport()

        # Get KB notations for this project
        kb_notations = self.kb.notation_list(project=self.project)
        kb_by_symbol: dict[str, Notation] = {}
        for n in kb_notations:
            symbol = n.get('symbol', '')
            if symbol:
                kb_by_symbol[symbol] = n

        report.stats['kb_notations'] = len(kb_notations)

        # Extract document notations
        doc_notations = self.extract_from_all_docs(doc_dir)
        report.stats['doc_notations'] = len(doc_notations)

        # Compare
        matched_kb_symbols = set()

        for symbol, usage in doc_notations.items():
            if symbol in kb_by_symbol:
                kb_notation = kb_by_symbol[symbol]
                matched_kb_symbols.add(symbol)

                # Check for definition conflicts
                kb_meaning = kb_notation.get('meaning', '')
                if usage.definitions:
                    # Simple conflict detection: check if meanings overlap
                    has_conflict = False
                    for doc_def in usage.definitions:
                        # Normalize and compare
                        kb_words = set(kb_meaning.lower().split())
                        doc_words = set(doc_def.lower().split())
                        overlap = len(kb_words & doc_words) / max(len(kb_words | doc_words), 1)
                        if overlap < 0.3 and len(doc_def) > 20:
                            has_conflict = True
                            break

                    if has_conflict:
                        report.conflicts.append((kb_notation, usage))
                    else:
                        report.well_matched.append((kb_notation, usage))
                else:
                    report.well_matched.append((kb_notation, usage))
            else:
                # Notation in docs but not in KB
                # Only report if it appears multiple times or has a definition
                if len(usage.locations) >= 2 or usage.definitions:
                    report.missing_from_kb.append(usage)

        # Find KB notations not in docs
        for symbol, notation in kb_by_symbol.items():
            if symbol not in matched_kb_symbols:
                report.not_in_docs.append(notation)

        report.stats['missing_from_kb'] = len(report.missing_from_kb)
        report.stats['not_in_docs'] = len(report.not_in_docs)
        report.stats['conflicts'] = len(report.conflicts)
        report.stats['well_matched'] = len(report.well_matched)

        return report

    def format_report(self, report: AuditReport) -> str:
        """Format audit report as text."""
        lines = []
        lines.append("=" * 60)
        lines.append("NOTATION AUDIT REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Stats
        lines.append(f"KB notations: {report.stats.get('kb_notations', 0)}")
        lines.append(f"Document notations: {report.stats.get('doc_notations', 0)}")
        lines.append("")
        lines.append(f"Well matched: {report.stats.get('well_matched', 0)}")
        lines.append(f"Missing from KB: {report.stats.get('missing_from_kb', 0)}")
        lines.append(f"Not in docs: {report.stats.get('not_in_docs', 0)}")
        lines.append(f"Potential conflicts: {report.stats.get('conflicts', 0)}")
        lines.append("")

        # Missing from KB
        if report.missing_from_kb:
            lines.append("=" * 60)
            lines.append("NOTATIONS IN DOCS BUT NOT KB")
            lines.append("=" * 60)
            for usage in sorted(report.missing_from_kb, key=lambda u: -len(u.locations))[:30]:
                lines.append(f"\n{usage.symbol} (used {len(usage.locations)} times)")
                if usage.definitions:
                    lines.append(f"  Definition: {usage.definitions[0][:80]}")
                lines.append(f"  Locations: {', '.join(usage.locations[:5])}")
                if usage.contexts:
                    lines.append(f"  Contexts: {', '.join(usage.contexts[:3])}")
                lines.append("  → Action: Consider adding with `kb notation add`")

        # Not in docs
        if report.not_in_docs:
            lines.append("")
            lines.append("=" * 60)
            lines.append("KB NOTATIONS NOT IN DOCS (possibly obsolete)")
            lines.append("=" * 60)
            for notation in report.not_in_docs[:20]:
                lines.append(f"\n{notation.get('id', 'unknown')}: \"{notation.get('symbol', '')}\"")
                lines.append(f"  Meaning: {notation.get('meaning', '')[:60]}")
                lines.append(f"  Domain: {notation.get('domain', 'unspecified')}")
                lines.append("  → Action: Review and possibly remove")

        # Conflicts
        if report.conflicts:
            lines.append("")
            lines.append("=" * 60)
            lines.append("POTENTIAL DEFINITION CONFLICTS")
            lines.append("=" * 60)
            for notation, usage in report.conflicts[:20]:
                lines.append(f"\nSymbol: {usage.symbol}")
                lines.append(f"  KB: {notation.get('meaning', '')[:60]}")
                if usage.definitions:
                    lines.append(f"  Doc: {usage.definitions[0][:60]}")
                lines.append(f"  Locations: {', '.join(usage.locations[:3])}")
                lines.append("  → Action: Clarify or update KB notation")

        return '\n'.join(lines)


def main():
    """CLI entry point."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Audit notation consistency")
    parser.add_argument("doc_dir", type=Path, help="Document directory to audit")
    parser.add_argument("--project", required=True, help="Project filter")
    parser.add_argument("--output", type=Path, help="Output report file")
    args = parser.parse_args()

    if not args.doc_dir.is_dir():
        print(f"Error: {args.doc_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Initialize KB with environment config
    kb = KnowledgeBase(
        embedding_url=os.environ.get("KB_EMBEDDING_URL", ""),
        embedding_dim=int(os.environ.get("KB_EMBEDDING_DIM", "4096")),
    )

    auditor = NotationAuditor(kb, args.project)
    report = auditor.audit(args.doc_dir)
    output = auditor.format_report(report)

    if args.output:
        args.output.write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
