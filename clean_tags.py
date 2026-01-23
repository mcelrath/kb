#!/usr/bin/env python3
"""KB tag cleanup: consolidate case duplicates, normalize tags, remove orphans."""

import argparse
import json
from collections import defaultdict
from kb import KnowledgeBase


# Explicit tag transformations: old_tag -> new_tag (None = remove)
TAG_TRANSFORMS = {
    # Remove entirely (invalid/file references)
    'source:8D_GAP.md': None,
    'source:HYPERCOMPLEX.md': None,
    'source:KINETIC.md': None,
    'source:SO44_DECOMPOSITION.md': None,
    '????"': None,

    # Fix malformed (spaces, punctuation)
    'Jordan algebra': 'jordan-algebra',
    'outer automorphism': 'outer-automorphism',
    'balanced-signature??': 'balanced-signature',
    'definite-signatures??': 'definite-signatures',
    "also-'polynomial-logarithm'.": 'polynomial-logarithm',
    'U(1)_Y': 'u1-y',
    'su(2)+': 'su2-plus',
    'su(2)-': 'su2-minus',
    'so(4)': 'so4',

    # Unicode/special chars to ASCII
    '1/r²': 'inverse-r-squared',
    'γ_*': 'gamma-star',
    'n=1': 'n-equals-1',
    'n=2': 'n-equals-2',
}


def find_case_duplicates(kb):
    """Find tags that differ only by case."""
    cursor = kb.conn.execute(
        "SELECT id, tags FROM findings WHERE tags IS NOT NULL"
    )
    tag_counts = defaultdict(int)
    for row in cursor.fetchall():
        tags = json.loads(row[1]) if row[1] else []
        for tag in tags:
            tag_counts[tag] += 1

    # Group by lowercase
    all_tags = defaultdict(list)
    for tag, count in tag_counts.items():
        all_tags[tag.lower()].append((tag, count))

    return {k: v for k, v in all_tags.items() if len(v) > 1}


def consolidate_case_duplicates(kb, dry_run=False):
    """Consolidate case-insensitive duplicate tags to lowercase form."""
    duplicates = find_case_duplicates(kb)
    print(f"Found {len(duplicates)} case-insensitive duplicate groups")

    updated = 0
    for canonical, variants in sorted(duplicates.items()):
        non_canonical = [v for v, _ in variants if v != canonical]
        if not non_canonical:
            continue

        print(f"\n{canonical}:")
        for v, c in variants:
            marker = " -> " + canonical if v != canonical else " (canonical)"
            print(f"  {v} ({c} uses){marker}")

        if dry_run:
            continue

        for old_tag in non_canonical:
            cursor = kb.conn.execute(
                "SELECT id, tags FROM findings WHERE tags LIKE ?",
                (f'%"{old_tag}"%',)
            )
            for row in cursor.fetchall():
                tags = json.loads(row[1])
                # Replace tag and deduplicate (preserves order)
                new_tags = list(dict.fromkeys(
                    canonical if t == old_tag else t for t in tags
                ))
                if tags != new_tags:
                    kb.conn.execute(
                        "UPDATE findings SET tags = ? WHERE id = ?",
                        (json.dumps(new_tags), row[0])
                    )
                    updated += 1

        kb.conn.commit()

    return {"groups": len(duplicates), "updated": updated}


def delete_project_findings(kb, project, dry_run=False):
    """Delete all findings for a project."""
    cursor = kb.conn.execute(
        "SELECT id, content FROM findings WHERE project = ?", (project,)
    )
    findings = cursor.fetchall()
    print(f"Found {len(findings)} finding(s) for project '{project}'")

    deleted = 0
    for row in findings:
        print(f"  {row[0]}: {row[1][:80]}...")
        if not dry_run:
            try:
                kb.delete(row[0])
                deleted += 1
            except ValueError as e:
                print(f"    Error: {e}")

    return {"found": len(findings), "deleted": deleted}


def normalize_all_tags(kb, dry_run=False):
    """Normalize all tags: lowercase, fix malformed, remove invalid."""

    # Build full transformation map
    transforms = dict(TAG_TRANSFORMS)

    # Find all unique tags and add lowercase transforms for uppercase-only
    all_tags = set()
    cursor = kb.conn.execute("SELECT tags FROM findings WHERE tags IS NOT NULL")
    for row in cursor.fetchall():
        tags = json.loads(row[0]) if row[0] else []
        all_tags.update(tags)

    for tag in all_tags:
        if tag not in transforms and tag != tag.lower():
            transforms[tag] = tag.lower()

    # Report what will be changed
    print(f"Tags to transform: {len(transforms)}")
    print("\nExplicit transforms:")
    for old, new in sorted(TAG_TRANSFORMS.items()):
        action = f"-> {new}" if new else "(remove)"
        print(f"  {old} {action}")

    lowercase_transforms = {k: v for k, v in transforms.items() if k not in TAG_TRANSFORMS}
    print(f"\nUppercase -> lowercase: {len(lowercase_transforms)} tags")
    for old, new in sorted(lowercase_transforms.items())[:20]:
        print(f"  {old} -> {new}")
    if len(lowercase_transforms) > 20:
        print(f"  ... and {len(lowercase_transforms) - 20} more")

    if dry_run:
        return {"transforms": len(transforms), "updated": 0}

    # Apply transforms to all findings
    updated = 0
    cursor = kb.conn.execute("SELECT id, tags FROM findings WHERE tags IS NOT NULL")
    for row in cursor.fetchall():
        finding_id = row[0]
        tags = json.loads(row[1]) if row[1] else []

        new_tags = []
        for tag in tags:
            if tag in transforms:
                new_tag = transforms[tag]
                if new_tag is not None:  # None = remove
                    new_tags.append(new_tag)
            else:
                new_tags.append(tag)

        # Deduplicate while preserving order
        new_tags = list(dict.fromkeys(new_tags))

        # Safety: don't leave findings with empty tags if they had tags before
        if tags and not new_tags:
            print(f"  WARNING: {finding_id} would have no tags, keeping original")
            continue

        if new_tags != tags:
            kb.conn.execute(
                "UPDATE findings SET tags = ? WHERE id = ?",
                (json.dumps(new_tags), finding_id)
            )
            updated += 1

    kb.conn.commit()
    return {"transforms": len(transforms), "updated": updated}


def main():
    parser = argparse.ArgumentParser(description="KB tag cleanup")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--consolidate-case", action="store_true",
                        help="Consolidate case-insensitive duplicate tags")
    parser.add_argument("--normalize-all", action="store_true",
                        help="Normalize all tags: lowercase, fix malformed, remove invalid")
    parser.add_argument("--delete-project", type=str,
                        help="Delete all findings for a project")
    args = parser.parse_args()

    kb = KnowledgeBase()

    if args.consolidate_case:
        print("=== Consolidating case duplicates ===")
        result = consolidate_case_duplicates(kb, dry_run=args.dry_run)
        print(f"\nUpdated {result['updated']} findings")

    if args.normalize_all:
        print("=== Normalizing all tags ===")
        result = normalize_all_tags(kb, dry_run=args.dry_run)
        print(f"\nUpdated {result['updated']} findings")

    if args.delete_project:
        print(f"\n=== Deleting project: {args.delete_project} ===")
        result = delete_project_findings(kb, args.delete_project, dry_run=args.dry_run)
        print(f"Deleted {result['deleted']} findings")


if __name__ == "__main__":
    main()
