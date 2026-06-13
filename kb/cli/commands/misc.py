"""CLI handlers for miscellaneous commands: reconcile, notation-audit."""

import sys


def run_reconcile(kb, args) -> None:
    try:
        from kb_reconcile import DocumentReconciler
    except ImportError:
        print("Error: kb_reconcile module not found")
        sys.exit(1)

    reconciler = DocumentReconciler(kb)

    if args.import_missing:
        result = reconciler.import_missing_claims(args.import_missing)
        print(f"Imported {result['imported']} claims")
    else:
        result = reconciler.reconcile(args.document, project=args.project)
        print(f"\nReconciliation complete:")
        print(f"  Document claims: {result['doc_claims']}")
        print(f"  KB findings: {result['kb_findings']}")
        print(f"  Matched: {result['matched']}")
        print(f"  Missing from KB: {result['missing']}")
        print(f"  Extra in KB: {result['extra']}")

        if args.export_missing and result.get('missing_claims'):
            reconciler.export_missing_claims(args.export_missing, result['missing_claims'])
            print(f"\nExported {len(result['missing_claims'])} missing claims to {args.export_missing}")


def run_notation_audit(kb, args) -> None:
    try:
        from kb_notation_audit import NotationAuditor
    except ImportError:
        print("Error: kb_notation_audit module not found")
        sys.exit(1)

    auditor = NotationAuditor(kb)
    result = auditor.audit(args.document, project=args.project)
    print(f"\nNotation audit complete:")
    print(f"  Document notations: {result['doc_notations']}")
    print(f"  KB notations: {result['kb_notations']}")
    print(f"  Matched: {result['matched']}")
    print(f"  Missing from KB: {result['missing']}")
    print(f"  Conflicts: {result['conflicts']}")
