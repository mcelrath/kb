#!/usr/bin/env python3
"""Link orphan tags to findings that mention them, remove true orphans."""

import argparse
import json
from kb import KnowledgeBase


# Known notation mappings (tag -> possible content variants)
NOTATION_ALIASES = {
    'g2-prime': ['G₂\'', 'G2\'', 'G₂-prime', 'G2-prime', 'G_2\''],
    'su2l': ['SU(2)_L', 'SU(2)L', 'SU₂_L', 'su(2)_l'],
    'su2r': ['SU(2)_R', 'SU(2)R', 'SU₂_R'],
    'u1': ['U(1)', 'U₁'],
    'u1-em': ['U(1)_em', 'U(1)_EM', 'U(1)_{em}'],
    'u1-y': ['U(1)_Y', 'U(1)Y', 'hypercharge'],
    'u1y': ['U(1)_Y', 'U(1)Y'],
    'su4': ['SU(4)', 'SU₄'],
    'so21': ['SO(2,1)', 'SO₂₁'],
    'so22': ['SO(2,2)', 'SO₂₂'],
    'sl2r': ['SL(2,R)', 'SL₂(R)', 'SL(2,ℝ)'],
    'tau-m': ['τ_m', 'τm', 'tau_m', 'τ-m'],
    'anti-de-sitter': ['AdS', 'anti de Sitter', 'Anti-de-Sitter'],
    'baryogenesis': ['baryon asymmetry', 'matter-antimatter', 'leptogenesis'],
    'gamma-star': ['γ_*', 'γ*', 'Γ_*', 'gamma*'],
    'inverse-r-squared': ['1/r²', '1/r^2', '1/r2'],
    'n-equals-1': ['n=1', 'N=1'],
    'n-equals-2': ['n=2', 'N=2'],
    'n2-sector': ['n=2-sector', 'N=2-sector', 'N=2 sector'],
    'z-equals-minus-1': ['z=-1', 'z = -1'],
}

# Tags too generic to auto-link (common English words)
GENERIC_TAGS = {
    'core', 'summary', 'unified', 'intersection', 'observable', 'bug', 'ansatz',
    'fix', 'algebra', 'algebraic', 'adjoint', 'analysis', 'approach', 'axiom',
    'basis', 'block', 'bound', 'branch', 'breaking', 'calculation', 'canonical',
    'channel', 'charge', 'class', 'classical', 'closure', 'code', 'component',
    'condition', 'connection', 'constraint', 'construction', 'continuum',
    'contribution', 'convention', 'convergence', 'correction', 'coupling',
    'critical', 'current', 'decomposition', 'definition', 'degeneracy',
    'derivation', 'detail', 'dimension', 'direction', 'distribution', 'domain',
    'effect', 'eigenvalue', 'element', 'embedding', 'emergence', 'energy',
    'equation', 'error', 'evidence', 'evolution', 'example', 'expansion',
    'expression', 'extension', 'factor', 'failure', 'feature', 'field',
    'flow', 'form', 'formula', 'framework', 'function', 'gap', 'generation',
    'geometry', 'global', 'graph', 'group', 'hierarchy', 'identity', 'index',
    'insight', 'integration', 'interpretation', 'invariant', 'issue', 'kernel',
    'level', 'limit', 'linear', 'link', 'local', 'loop', 'mass', 'matrix',
    'mechanism', 'method', 'metric', 'mode', 'model', 'momentum', 'motion',
    'multiplicity', 'notation', 'number', 'operator', 'order', 'origin',
    'parameter', 'particle', 'path', 'pattern', 'phase', 'point', 'potential',
    'prediction', 'principle', 'problem', 'product', 'projection', 'proof',
    'property', 'proposal', 'question', 'ratio', 'reality', 'reduction',
    'reference', 'relation', 'representation', 'requirement', 'resolution',
    'result', 'rotation', 'rule', 'scalar', 'scale', 'scheme', 'section',
    'selection', 'sequence', 'series', 'set', 'sign', 'signal', 'solution',
    'source', 'space', 'spectrum', 'spin', 'stability', 'state', 'step',
    'strategy', 'string', 'structure', 'subgroup', 'success', 'sum', 'symmetry',
    'system', 'table', 'technique', 'tensor', 'term', 'test', 'theorem',
    'theory', 'trace', 'transformation', 'transition', 'type', 'uniqueness',
    'unit', 'vacuum', 'value', 'variable', 'variation', 'vector', 'verification',
    'version', 'violation', 'wave', 'weight', 'work',
}


def tag_to_patterns(tag: str) -> list[str]:
    """Convert tag to search patterns, including known aliases."""
    # Replace hyphens/underscores with spaces for matching
    base = tag.replace('-', ' ').replace('_', ' ').lower()
    patterns = [base]
    # Also try the literal tag (for things like "8x8")
    if tag.lower() != base:
        patterns.append(tag.lower())
    # Add known aliases
    if tag in NOTATION_ALIASES:
        patterns.extend([a.lower() for a in NOTATION_ALIASES[tag]])
    return patterns


def analyze_tags(kb):
    """Analyze all tags and categorize them."""
    # Get all current findings
    findings = {}
    for row in kb.conn.execute(
        'SELECT id, tags, content FROM findings WHERE status = "current"'
    ):
        findings[row[0]] = {
            'tags': set(json.loads(row[1]) if row[1] else []),
            'content': row[2].lower()
        }

    # Get all unique tags
    all_tags = set()
    for f in findings.values():
        all_tags.update(f['tags'])

    results = {
        'linkable': [],      # Tag text appears in content missing the tag
        'well_tagged': [],   # Tag text appears and findings have the tag
        'orphan': [],        # Tag text doesn't appear in any content
    }

    for tag in sorted(all_tags):
        patterns = tag_to_patterns(tag)

        # Find findings containing tag text but missing the tag
        missing_tag = []
        for fid, f in findings.items():
            if tag not in f['tags']:
                for p in patterns:
                    if p in f['content']:
                        missing_tag.append(fid)
                        break

        # Find findings that have this tag
        has_tag = [fid for fid, f in findings.items() if tag in f['tags']]

        # Check if tag text appears in content of tagged findings
        tagged_has_text = sum(
            1 for fid in has_tag
            if any(p in findings[fid]['content'] for p in patterns)
        )

        if missing_tag:
            # Only report if reasonable number to link (not "algebra" -> 382)
            if len(missing_tag) <= 20:
                results['linkable'].append({
                    'tag': tag,
                    'has_tag': len(has_tag),
                    'tagged_has_text': tagged_has_text,
                    'missing_tag': missing_tag,
                })
            else:
                # Too generic, treat as well-tagged
                results['well_tagged'].append({
                    'tag': tag,
                    'has_tag': len(has_tag),
                    'too_generic': len(missing_tag)
                })
        elif tagged_has_text == 0:
            # Tag text doesn't appear even in tagged findings
            results['orphan'].append({
                'tag': tag,
                'count': len(has_tag),
                'finding_ids': has_tag
            })
        else:
            results['well_tagged'].append({
                'tag': tag,
                'has_tag': len(has_tag)
            })

    return results


def link_tags(kb, tag: str, finding_ids: list[str], dry_run: bool = False):
    """Add a tag to findings that are missing it."""
    updated = 0
    for fid in finding_ids:
        cursor = kb.conn.execute(
            'SELECT tags FROM findings WHERE id = ?', (fid,)
        )
        row = cursor.fetchone()
        if not row:
            continue

        tags = json.loads(row[0]) if row[0] else []
        if tag not in tags:
            tags.append(tag)
            if not dry_run:
                kb.conn.execute(
                    'UPDATE findings SET tags = ? WHERE id = ?',
                    (json.dumps(tags), fid)
                )
            updated += 1

    if not dry_run:
        kb.conn.commit()
    return updated


def remove_tag_from_findings(kb, tag: str, finding_ids: list[str], dry_run: bool = False):
    """Remove a tag from findings."""
    updated = 0
    for fid in finding_ids:
        cursor = kb.conn.execute(
            'SELECT tags FROM findings WHERE id = ?', (fid,)
        )
        row = cursor.fetchone()
        if not row:
            continue

        tags = json.loads(row[0]) if row[0] else []
        if tag in tags:
            tags.remove(tag)
            if not dry_run:
                kb.conn.execute(
                    'UPDATE findings SET tags = ? WHERE id = ?',
                    (json.dumps(tags), fid)
                )
            updated += 1

    if not dry_run:
        kb.conn.commit()
    return updated


def main():
    parser = argparse.ArgumentParser(description="Link or remove orphan tags")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze tags and report linkable/orphans")
    parser.add_argument("--link-all", action="store_true",
                        help="Link all linkable tags to findings")
    parser.add_argument("--remove-orphans", action="store_true",
                        help="Remove tags that don't appear in any content")
    parser.add_argument("--min-ratio", type=float, default=0.5,
                        help="Min ratio of (has_tag)/(missing_tag) to auto-link")
    args = parser.parse_args()

    kb = KnowledgeBase()

    if args.analyze or not (args.link_all or args.remove_orphans):
        print("=== Analyzing tags ===")
        results = analyze_tags(kb)

        print(f"\nLinkable tags (text in content, missing tag): {len(results['linkable'])}")
        for item in results['linkable'][:20]:
            print(f"  {item['tag']}: {item['has_tag']} have tag, "
                  f"{len(item['missing_tag'])} could be linked")

        print(f"\nOrphan tags (text not in any content): {len(results['orphan'])}")
        for item in results['orphan']:
            print(f"  {item['tag']}: {item['count']} findings")

        print(f"\nWell-tagged: {len(results['well_tagged'])}")

    if args.link_all:
        print("\n=== Linking tags ===")
        results = analyze_tags(kb)
        total_linked = 0
        skipped_generic = 0

        for item in results['linkable']:
            tag = item['tag']
            
            # Skip generic tags
            if tag in GENERIC_TAGS:
                skipped_generic += 1
                continue
                
            has = item['has_tag']
            missing = len(item['missing_tag'])
            ratio = has / missing if missing > 0 else float('inf')

            # Only link if the tag is reasonably specific
            # (more findings already have it than are missing it)
            if ratio >= args.min_ratio:
                if args.dry_run:
                    print(f"  Would link '{item['tag']}' to {missing} findings")
                else:
                    linked = link_tags(kb, item['tag'], item['missing_tag'])
                    print(f"  Linked '{item['tag']}' to {linked} findings")
                    total_linked += linked

        print(f"\nTotal: {'would link' if args.dry_run else 'linked'} {total_linked} tag-finding pairs (skipped {skipped_generic} generic tags)")

    if args.remove_orphans:
        print("\n=== Removing orphan tags ===")
        results = analyze_tags(kb)
        total_removed = 0

        for item in results['orphan']:
            tag = item['tag']
            finding_ids = item['finding_ids']

            if args.dry_run:
                print(f"  Would remove '{tag}' from {len(finding_ids)} findings")
            else:
                removed = remove_tag_from_findings(kb, tag, finding_ids)
                print(f"  Removed '{tag}' from {removed} findings")
                total_removed += removed

        print(f"\nTotal: {'would remove' if args.dry_run else 'removed'} {total_removed} tags")


if __name__ == "__main__":
    main()
