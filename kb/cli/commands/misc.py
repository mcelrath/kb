"""CLI handlers for miscellaneous commands: reconcile, notation-audit."""

import sys

import kb.cli.output as output


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
        print(output.c("\nReconciliation complete:", "bold"))
        rows = [
            ("Document claims", result['doc_claims']),
            ("KB findings",     result['kb_findings']),
            ("Matched",         result['matched']),
            ("Missing from KB", result['missing']),
            ("Extra in KB",     result['extra']),
        ]
        for label, val in rows:
            color = "green" if label == "Matched" else ("yellow" if val else None)
            row = "  " + output.c(f"{label}: {val}", color)
            print(output.fit_line(row))

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
    print(output.c("\nNotation audit complete:", "bold"))
    rows = [
        ("Document notations", result['doc_notations']),
        ("KB notations",       result['kb_notations']),
        ("Matched",            result['matched']),
        ("Missing from KB",    result['missing']),
        ("Conflicts",          result['conflicts']),
    ]
    for label, val in rows:
        color = "green" if label == "Matched" else ("red" if label == "Conflicts" and val else None)
        row = "  " + output.c(f"{label}: {val}", color)
        print(output.fit_line(row))
