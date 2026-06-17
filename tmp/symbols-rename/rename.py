"""A1: mechanical rename python_symbols -> symbols across the kb codebase.

Ordered replacements (longest/most-specific first so substrings don't collide):
  python_symbols_vec -> symbols_vec
  add_python_symbol  -> add_symbol
  idx_python_symbols -> idx_symbols
  python_symbols     -> symbols      (the table name + remaining refs)

Scoped to the known referencing files (from the python_symbols scan); schema.py's
rename MIGRATION block is added separately by Edit (this only renames refs).
"""
import pathlib

FILES = [
    "kb/facade.py",
    "kb/core/schema.py",
    "kb/ingest/python.py",
    "kb/ingest/tex.py",
    "kb/ingest/typescript.py",
    "hooks/scripts/symbol_surface.py",
    "kb/surface/producers.py",
    "hooks/scripts/compose_time_check.py",
    "kb.py",
    "kb/cli/commands/surface.py",
    "kb/code_ingest/chunker.py",
    "tests/test_code_ingest_chunker.py",
    "hooks/scripts/lib/_seen.py",
]

REPLACEMENTS = [
    ("python_symbols_vec", "symbols_vec"),
    ("add_python_symbol", "add_symbol"),
    ("idx_python_symbols", "idx_symbols"),
    ("python_symbols", "symbols"),
]

root = pathlib.Path(__file__).resolve().parents[2]
for rel in FILES:
    p = root / rel
    if not p.exists():
        print("MISSING", rel)
        continue
    t = p.read_text()
    before = t
    for old, new in REPLACEMENTS:
        t = t.replace(old, new)
    if t != before:
        p.write_text(t)
        # count residual (should be 0)
        resid = t.count("python_symbols")
        print(f"{rel}: rewritten, residual python_symbols={resid}")
    else:
        print(f"{rel}: no change")
