"""Patch scripts/ingest_python.py: add PRUNE and --deleted mode."""

with open('/home/mcelrath/Projects/ai/kb/scripts/ingest_python.py', 'r') as f:
    content = f.read()

# 1. Add --deleted argument after --no-notations arg
OLD_ARGS = '''\
    parser.add_argument(
        "--no-notations",
        action="store_true",
        help="Skip populating notations table",
    )
    args = parser.parse_args()'''

NEW_ARGS = '''\
    parser.add_argument(
        "--no-notations",
        action="store_true",
        help="Skip populating notations table",
    )
    parser.add_argument(
        "--deleted",
        nargs="+",
        metavar="FILE",
        help="Remove all python_symbols rows for these deleted/renamed files",
    )
    args = parser.parse_args()'''

assert OLD_ARGS in content, "OLD_ARGS not found"
content = content.replace(OLD_ARGS, NEW_ARGS, 1)

# 2. Add deleted-file handling right after `kb = KnowledgeBase(args.db)` and before root computation
OLD_KB_INIT = '''\
    kb = KnowledgeBase(db_path=args.db)

    root = args.root.expanduser().resolve()'''

NEW_KB_INIT = '''\
    kb = KnowledgeBase(db_path=args.db)

    # Handle --deleted: remove all rows for explicitly-deleted files and exit.
    if args.deleted:
        for fpath in args.deleted:
            fpath_str = str(Path(fpath).expanduser().resolve())
            n = kb.delete_python_symbols_for_file(fpath_str)
            print(f"Deleted {n} rows for removed file: {fpath_str}", file=sys.stderr)
        return

    root = args.root.expanduser().resolve()'''

assert OLD_KB_INIT in content, "OLD_KB_INIT not found"
content = content.replace(OLD_KB_INIT, NEW_KB_INIT, 1)

# 3. After the per-file ingest loop (the insert + commit block), add per-file prune.
# The ingest loop ends just before "# Populate also_in_modules".
# We insert prune logic between the insert loop's try/except block and the also_in_modules block.

# Find the spot: after sym_iter close, before also_in_modules comment
OLD_PRUNE_ANCHOR = '''\
    if _tqdm and hasattr(sym_iter, 'close'):
        sym_iter.close()

    # Populate also_in_modules'''

NEW_PRUNE_ANCHOR = '''\
    if _tqdm and hasattr(sym_iter, 'close'):
        sym_iter.close()

    # PRUNE: when ingesting with --files, delete stale rows (symbols that vanished from each file).
    # Guard: only prune when a specific file list was provided, and only after parse succeeded.
    # A parse failure returns [] from parse_python_file and skips prune for that file (empty guard
    # in prune_python_symbols_for_file ensures empty live set => no deletion).
    if args.files and not args.dry_run:
        pruned_total = 0
        # Build per-file live (name, module) sets from successfully-parsed symbols
        file_to_live: dict[str, set[tuple[str, str]]] = {}
        for s in all_symbols:
            fpath_str = s["file"]
            if fpath_str not in file_to_live:
                file_to_live[fpath_str] = set()
            file_to_live[fpath_str].add((s["name"], s["module"]))
        for fpath in [str(Path(f).expanduser().resolve()) for f in args.files]:
            live = file_to_live.get(fpath, set())
            # If parse returned nothing for this file, live is empty; prune guard fires => 0 deleted.
            n = kb.prune_python_symbols_for_file(fpath, live)
            if n:
                print(f"  Pruned {n} stale symbol(s) from {fpath}", file=sys.stderr)
            pruned_total += n
        if pruned_total:
            print(f"Total pruned: {pruned_total}", file=sys.stderr)

    # Populate also_in_modules'''

assert OLD_PRUNE_ANCHOR in content, "OLD_PRUNE_ANCHOR not found"
content = content.replace(OLD_PRUNE_ANCHOR, NEW_PRUNE_ANCHOR, 1)

with open('/home/mcelrath/Projects/ai/kb/scripts/ingest_python.py', 'w') as f:
    f.write(content)

print(f"Done. Lines: {content.count(chr(10))}")
