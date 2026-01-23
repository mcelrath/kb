#!/usr/bin/env python3
"""Automated KB curation: tagging, consolidation, and entry point generation."""

import argparse
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from kb import KnowledgeBase


def auto_tag_findings(kb: KnowledgeBase, project: str | None = None, dry_run: bool = False) -> dict:
    """Auto-tag findings that have no tags using LLM suggestions.

    Returns:
        dict with 'tagged' count and 'errors' list
    """
    # Find findings without tags
    findings = kb.list_findings(project=project, limit=500)
    untagged = [f for f in findings if not f.get("tags")]

    print(f"Found {len(untagged)} untagged findings")

    tagged = 0
    errors = []

    for f in untagged:
        try:
            suggested = kb.suggest_tags(f["content"], project=f.get("project"))
            if suggested:
                if dry_run:
                    print(f"  {f['id']}: would add {suggested}")
                else:
                    result = kb.bulk_add_tags([f["id"]], suggested)
                    if result["updated"] > 0:
                        print(f"  {f['id']}: added {suggested}")
                        tagged += 1
        except Exception as e:
            errors.append((f["id"], str(e)))

    return {"tagged": tagged, "errors": errors}


def consolidate_duplicates(kb: KnowledgeBase, project: str | None = None, dry_run: bool = False) -> dict:
    """Run consolidation suggestions and optionally apply them.

    Returns:
        dict with 'clusters' found and 'consolidated' count
    """
    clusters = kb.suggest_consolidation(project=project, limit=100)

    print(f"Found {len(clusters)} clusters to potentially consolidate")

    consolidated = 0

    for cluster in clusters:
        members = cluster["members"]
        ids = [m["id"] for m in members]
        reason = cluster.get("analysis", "Similar content")

        # Get the findings to create a merged summary
        findings = [kb.get(fid) for fid in ids]
        findings = [f for f in findings if f]  # Filter out None

        if len(findings) < 2:
            continue

        # Create summary from findings
        contents = [f["content"] for f in findings]
        combined = "\n---\n".join(contents)

        # Use LLM to summarize (with fallback)
        try:
            summary = kb._llm_complete(f"""Merge these related findings into a single concise finding:

{combined}

Output only the merged content (1-3 sentences, preserve technical accuracy):""")
        except Exception as e:
            print(f"  LLM error: {e}")
            summary = None

        if not summary or len(summary) < 20:
            summary = contents[0]  # Fallback to first finding

        if dry_run:
            print(f"\nCluster ({len(ids)} findings): {reason}")
            for fid in ids:
                print(f"  - {fid}")
            print(f"  Summary: {summary[:100]}...")
        else:
            try:
                result = kb.consolidate_cluster(
                    finding_ids=ids,
                    summary=summary,
                    reason=reason,
                )
                print(f"Consolidated {len(ids)} findings into {result['new_id']}")
                consolidated += 1
            except ValueError as e:
                print(f"  Skipped cluster: {e}")

    return {"clusters": len(clusters), "consolidated": consolidated}


def generate_entry_points(kb: KnowledgeBase, project: str, dry_run: bool = False) -> dict:
    """Generate INDEX and GOTCHAS entry point findings.

    DEPRECATED: INDEX/GOTCHAS entries get stale when referenced findings are
    superseded or corrected. Use kb_search() directly instead.

    Returns:
        dict with 'created' list of entry points (always empty - function disabled)
    """
    print("WARNING: generate_entry_points is DEPRECATED and disabled.")
    print("INDEX/GOTCHAS entries get stale. Use kb_search() instead.")
    return {"created": []}

    # Original code below (disabled):
    created = []  # noqa: F841 - unreachable

    # Dimension-specific indexes
    for dim in [2, 4, 8]:
        tag = f"dim-{dim}"
        findings = kb.search(tag, project=project, limit=50)
        findings = [f for f in findings if f.get("tags") and tag in f["tags"]]

        if findings:
            content = f"INDEX: Dimension {dim} results for {project}\n\n"
            content += f"Found {len(findings)} findings:\n"
            for f in findings[:20]:
                content += f"- [{f['type']}] {f['content'][:80]}... ({f['id']})\n"

            if dry_run:
                print(f"\nWould create INDEX for dim-{dim}:")
                print(content[:500])
            else:
                try:
                    result = kb.add(
                        content=content,
                        finding_type="discovery",
                        project=project,
                        tags=[f"index", tag, "entry-point"],
                        check_duplicate=False,
                        auto_tag=False,
                        auto_classify=False,
                    )
                    created.append(f"INDEX-dim-{dim}: {result['id']}")
                    print(f"Created INDEX for dim-{dim}: {result['id']}")
                except Exception as e:
                    print(f"Failed to create INDEX for dim-{dim}: {e}")

    # GOTCHAS entry point
    failures = kb.list_findings(project=project, finding_type="failure", limit=50)
    if failures:
        content = f"GOTCHAS: Known pitfalls for {project}\n\n"
        for f in failures[:15]:
            content += f"- {f['content'][:100]}...\n"

        if dry_run:
            print(f"\nWould create GOTCHAS:")
            print(content[:500])
        else:
            try:
                result = kb.add(
                    content=content,
                    finding_type="discovery",
                    project=project,
                    tags=["gotchas", "entry-point", "failures"],
                    check_duplicate=False,
                    auto_tag=False,
                    auto_classify=False,
                )
                created.append(f"GOTCHAS: {result['id']}")
                print(f"Created GOTCHAS: {result['id']}")
            except Exception as e:
                print(f"Failed to create GOTCHAS: {e}")

    return {"created": created}


def main():
    parser = argparse.ArgumentParser(description="Automated KB curation")
    parser.add_argument("-p", "--project", help="Project to curate")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--tag-only", action="store_true", help="Only run auto-tagging")
    parser.add_argument("--consolidate-only", action="store_true", help="Only run consolidation")
    parser.add_argument("--entry-points-only", action="store_true", help="Only generate entry points")
    args = parser.parse_args()

    kb = KnowledgeBase(
        embedding_url=os.environ.get("KB_EMBEDDING_URL", ""),
        embedding_dim=int(os.environ.get("KB_EMBEDDING_DIM", "4096")),
    )

    run_all = not (args.tag_only or args.consolidate_only or args.entry_points_only)

    results = {}

    if run_all or args.tag_only:
        print("\n=== Auto-tagging ===")
        results["tagging"] = auto_tag_findings(kb, project=args.project, dry_run=args.dry_run)

    if run_all or args.consolidate_only:
        print("\n=== Consolidation ===")
        results["consolidation"] = consolidate_duplicates(kb, project=args.project, dry_run=args.dry_run)

    if (run_all or args.entry_points_only) and args.project:
        print("\n=== Entry Points ===")
        results["entry_points"] = generate_entry_points(kb, project=args.project, dry_run=args.dry_run)

    print("\n=== Summary ===")
    if "tagging" in results:
        print(f"Tagged: {results['tagging']['tagged']} findings")
    if "consolidation" in results:
        print(f"Consolidated: {results['consolidation']['consolidated']} clusters")
    if "entry_points" in results:
        print(f"Created: {len(results['entry_points']['created'])} entry points")


if __name__ == "__main__":
    main()
