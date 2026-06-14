"""Smoke test chunker on a real goose Rust file."""
import sys
sys.path.insert(0, '')
from pathlib import Path
from kb.code_ingest.chunker import chunk_file

f = Path('/home/mcelrath/Projects/ai/goose/crates/goose-cli/src/cli.rs')
root = Path('/home/mcelrath/Projects/ai/goose')
chunks = chunk_file(f, 'rust', root=root)
full = [c for c in chunks if not c.extra.get('is_signature_only')]
sig = [c for c in chunks if c.extra.get('is_signature_only')]
print(f'Total chunks: {len(chunks)}  (full={len(full)}, sig-only={len(sig)})')
for c in full[:15]:
    impl_info = f" (impl {c.extra['parent_impl']})" if c.extra.get('parent_impl') else ''
    print(f"  [{c.kind:8}] {c.name}{impl_info} line={c.line} vis={c.extra['visibility']!r}")
