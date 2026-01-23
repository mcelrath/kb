#!/usr/bin/env python3
"""
Document-Driven Reconciliation for Knowledge Base.

Compares markdown documents against KB findings to identify:
- Stale findings (no document backing)
- Missing findings (document claims not in KB)
- Potential conflicts (finding vs document mismatch)
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import from kb.py (same directory)
from kb import KnowledgeBase

# Type alias for finding dicts
Finding = dict[str, Any]


@dataclass
class Claim:
    """A claim extracted from a document."""
    text: str
    source_file: str
    line_number: int
    claim_type: str  # "theorem", "table_row", "definition", "result"
    context: str = ""  # Section header context


@dataclass
class MatchResult:
    """Result of matching a claim to KB findings."""
    claim: Claim
    matches: list[Finding] = field(default_factory=list)
    best_similarity: float = 0.0
    match_quality: str = "none"  # "strong", "partial", "none"


@dataclass
class ReconciliationReport:
    """Full reconciliation report."""
    stale_findings: list[Finding] = field(default_factory=list)
    missing_claims: list[Claim] = field(default_factory=list)
    conflicts: list[tuple[Finding, Claim]] = field(default_factory=list)
    well_matched: list[tuple[Finding, Claim]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


class DocumentReconciler:
    """Reconcile documents with KB findings."""

    # Patterns for claim extraction
    THEOREM_PATTERN = re.compile(
        r'\*\*(?:Theorem|Result|Claim|Proposition|Lemma|Corollary)[^*]*\*\*[:\s]*(.+?)(?=\n\n|\n\*\*|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    DEFINITION_PATTERN = re.compile(
        r'\*\*([^*]+)\*\*\s*[=:]\s*(.+?)(?=\n|$)',
        re.MULTILINE
    )

    TABLE_ROW_PATTERN = re.compile(
        r'^\|([^|]+\|)+\s*$',
        re.MULTILINE
    )

    HEADER_PATTERN = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)

    # Similarity thresholds
    STRONG_THRESHOLD = 0.75  # Good semantic match
    PARTIAL_THRESHOLD = 0.65  # Related content
    STALE_THRESHOLD = 0.55  # Below this, finding has no document backing

    def __init__(self, kb: KnowledgeBase, project: str):
        self.kb = kb
        self.project = project

    def extract_claims(self, doc_path: Path) -> list[Claim]:
        """Extract structured claims from a markdown document."""
        claims = []
        content = doc_path.read_text()
        filename = doc_path.name

        # Track current section context
        current_section = ""

        # Find section headers for context
        headers = list(self.HEADER_PATTERN.finditer(content))

        def get_section_at(pos: int) -> str:
            """Get the section header context at a position."""
            for i, h in enumerate(headers):
                if h.start() > pos:
                    return headers[i-1].group(2) if i > 0 else ""
            return headers[-1].group(2) if headers else ""

        # Extract theorems/results
        for match in self.THEOREM_PATTERN.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            text = match.group(1).strip()
            if len(text) > 50:  # Skip very short matches
                claims.append(Claim(
                    text=text[:500],  # Limit length
                    source_file=filename,
                    line_number=line_num,
                    claim_type="theorem",
                    context=get_section_at(match.start())
                ))

        # Extract definitions
        for match in self.DEFINITION_PATTERN.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            term = match.group(1).strip()
            definition = match.group(2).strip()
            if len(definition) > 20:
                claims.append(Claim(
                    text=f"{term}: {definition}",
                    source_file=filename,
                    line_number=line_num,
                    claim_type="definition",
                    context=get_section_at(match.start())
                ))

        # Extract significant table rows (skip headers)
        lines = content.split('\n')
        in_table = False
        for i, line in enumerate(lines):
            if line.strip().startswith('|') and '---' not in line:
                # Check if this looks like a data row (not header)
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if len(cells) >= 2 and not all(c.startswith('*') for c in cells):
                    # Skip header-like rows
                    if not any(c.lower() in ['signature', 'property', 'type', 'description'] for c in cells):
                        text = ' | '.join(cells)
                        if len(text) > 30:
                            claims.append(Claim(
                                text=text,
                                source_file=filename,
                                line_number=i + 1,
                                claim_type="table_row",
                                context=get_section_at(sum(len(l)+1 for l in lines[:i]))
                            ))

        return claims

    def match_claim_to_findings(self, claim: Claim) -> MatchResult:
        """Find KB findings that match this claim."""
        results = self.kb.search(
            claim.text,
            project=self.project,
            limit=5,
            use_vector=True
        )

        if not results:
            return MatchResult(claim=claim, match_quality="none")

        best = results[0]
        similarity = best.get('similarity', 0)

        if similarity >= self.STRONG_THRESHOLD:
            quality = "strong"
        elif similarity >= self.PARTIAL_THRESHOLD:
            quality = "partial"
        else:
            quality = "none"

        return MatchResult(
            claim=claim,
            matches=results,
            best_similarity=similarity,
            match_quality=quality
        )

    def check_finding_backing(self, finding: Finding, doc_contents: dict[str, str]) -> list[tuple[str, int, float]]:
        """Find document text supporting this finding.

        Returns list of (filename, line_number, similarity) tuples.
        """
        backing = []
        finding_text = finding.get('content', '')

        # Search each document for similar content
        for filename, content in doc_contents.items():
            lines = content.split('\n')
            # Check paragraphs (groups of non-empty lines)
            para_start = 0
            para_text = []

            for i, line in enumerate(lines):
                if line.strip():
                    if not para_text:
                        para_start = i
                    para_text.append(line)
                elif para_text:
                    # End of paragraph - check similarity
                    full_para = ' '.join(para_text)
                    if len(full_para) > 50:
                        # Use simple word overlap as proxy for similarity
                        # (Embedding comparison would be expensive for all paragraphs)
                        similarity = self._text_similarity(finding_text, full_para)
                        if similarity >= self.STALE_THRESHOLD:
                            backing.append((filename, para_start + 1, similarity))
                    para_text = []

        return sorted(backing, key=lambda x: -x[2])[:3]  # Top 3 matches

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word-overlap similarity (Jaccard-like)."""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union else 0.0

    def reconcile(self, doc_dir: Path) -> ReconciliationReport:
        """Run full reconciliation between documents and KB."""
        report = ReconciliationReport()

        # Get all findings for this project
        findings = self.kb.list_findings(project=self.project, limit=1000)
        report.stats['total_findings'] = len(findings)

        # Load all markdown documents
        doc_contents = {}
        all_claims = []

        for doc_path in doc_dir.rglob('*.md'):
            if doc_path.name.startswith('.'):
                continue
            try:
                doc_contents[doc_path.name] = doc_path.read_text()
                claims = self.extract_claims(doc_path)
                all_claims.extend(claims)
            except Exception as e:
                print(f"Warning: Could not read {doc_path}: {e}", file=sys.stderr)

        report.stats['total_documents'] = len(doc_contents)
        report.stats['total_claims'] = len(all_claims)

        # Match claims to findings
        matched_finding_ids = set()

        for claim in all_claims:
            result = self.match_claim_to_findings(claim)

            if result.match_quality == "strong":
                for match in result.matches:
                    if match.get('similarity', 0) >= self.STRONG_THRESHOLD:
                        matched_finding_ids.add(match['id'])
                        report.well_matched.append((match, claim))
            elif result.match_quality == "partial":
                # Partial match - potential conflict or update needed
                if result.matches:
                    report.conflicts.append((result.matches[0], claim))
                    matched_finding_ids.add(result.matches[0]['id'])
            else:
                # No match - missing from KB
                report.missing_claims.append(claim)

        # Check for stale findings (not matched to any claim)
        for finding in findings:
            if finding['id'] not in matched_finding_ids:
                # Double-check with document content search
                backing = self.check_finding_backing(finding, doc_contents)
                if not backing:
                    report.stale_findings.append(finding)

        report.stats['well_matched'] = len(report.well_matched)
        report.stats['missing'] = len(report.missing_claims)
        report.stats['stale'] = len(report.stale_findings)
        report.stats['conflicts'] = len(report.conflicts)

        return report

    def format_report(self, report: ReconciliationReport) -> str:
        """Format reconciliation report as text."""
        lines = []
        lines.append("=" * 60)
        lines.append("KB RECONCILIATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Stats
        lines.append(f"Documents scanned: {report.stats.get('total_documents', 0)}")
        lines.append(f"Claims extracted: {report.stats.get('total_claims', 0)}")
        lines.append(f"KB findings: {report.stats.get('total_findings', 0)}")
        lines.append("")
        lines.append(f"Well matched: {report.stats.get('well_matched', 0)}")
        lines.append(f"Missing from KB: {report.stats.get('missing', 0)}")
        lines.append(f"Potentially stale: {report.stats.get('stale', 0)}")
        lines.append(f"Potential conflicts: {report.stats.get('conflicts', 0)}")
        lines.append("")

        # Stale findings
        if report.stale_findings:
            lines.append("=" * 60)
            lines.append("STALE FINDINGS (no document backing)")
            lines.append("=" * 60)
            for f in report.stale_findings[:20]:
                lines.append(f"\n{f['id']} [{f['type']}]")
                lines.append(f"  {f['content'][:100]}...")
                lines.append(f"  → Action: Review and possibly supersede")

        # Missing claims
        if report.missing_claims:
            lines.append("")
            lines.append("=" * 60)
            lines.append("MISSING FROM KB")
            lines.append("=" * 60)
            for claim in report.missing_claims[:20]:
                lines.append(f"\n{claim.source_file}:{claim.line_number} [{claim.claim_type}]")
                lines.append(f"  {claim.text[:100]}...")
                lines.append(f"  Context: {claim.context}")
                lines.append(f"  → Action: Consider adding to KB")

        # Conflicts
        if report.conflicts:
            lines.append("")
            lines.append("=" * 60)
            lines.append("POTENTIAL CONFLICTS")
            lines.append("=" * 60)
            for finding, claim in report.conflicts[:20]:
                lines.append(f"\n{finding['id']} vs {claim.source_file}:{claim.line_number}")
                lines.append(f"  Finding: {finding['content'][:80]}...")
                lines.append(f"  Document: {claim.text[:80]}...")
                lines.append(f"  → Action: Review and reconcile")

        return '\n'.join(lines)

    def export_missing_json(
        self,
        report: ReconciliationReport,
        claim_types: list[str] | None = None,
        min_length: int = 50,
    ) -> list[dict]:
        """Export missing claims as JSON for import.

        Args:
            report: Reconciliation report
            claim_types: Filter to specific types (theorem, definition, table_row)
            min_length: Minimum claim text length
        """
        findings = []
        for claim in report.missing_claims:
            if claim_types and claim.claim_type not in claim_types:
                continue
            if len(claim.text) < min_length:
                continue

            # Map claim type to finding type
            finding_type = "discovery"
            if claim.claim_type == "theorem":
                finding_type = "success"  # Proven results

            findings.append({
                "content": claim.text,
                "type": finding_type,
                "project": self.project,
                "tags": [f"source:{claim.source_file}", claim.claim_type],
                "evidence": f"{claim.source_file}:{claim.line_number} ({claim.context})",
            })
        return findings


def main():
    """CLI entry point."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Reconcile documents with KB")
    parser.add_argument("doc_dir", type=Path, help="Document directory to reconcile")
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

    reconciler = DocumentReconciler(kb, args.project)
    report = reconciler.reconcile(args.doc_dir)
    output = reconciler.format_report(report)

    if args.output:
        args.output.write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
