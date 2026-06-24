"""CLI handlers for miscellaneous commands: reconcile, notation-audit."""

import json
import sys
from pathlib import Path


def run_reconcile(kb, args) -> None:
    from kb_reconcile import DocumentReconciler

    if not args.project:
        print("Error: --project is required", file=sys.stderr)
        sys.exit(1)
    doc_dir = Path(args.document)
    if not doc_dir.is_dir():
        print(f"Error: {doc_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    reconciler = DocumentReconciler(kb, args.project)
    report = reconciler.reconcile(doc_dir)
    print(reconciler.format_report(report))

    if args.export_missing:
        claims = reconciler.export_missing_json(report)
        Path(args.export_missing).write_text(json.dumps(claims, indent=2))
        print(f"\nExported {len(claims)} missing claims to {args.export_missing}")


def run_notation_audit(kb, args) -> None:
    from kb_notation_audit import NotationAuditor

    if not args.project:
        print("Error: --project is required", file=sys.stderr)
        sys.exit(1)
    doc_dir = Path(args.document)
    if not doc_dir.is_dir():
        print(f"Error: {doc_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    auditor = NotationAuditor(kb, args.project)
    report = auditor.audit(doc_dir)
    print(auditor.format_report(report))
