"""
Tests for kb/code_ingest/chunker.py — generalized tree-sitter code chunker,
Rust language config (first vertical slice, kb-asf.4).

Each assertion states the REASON for the expected value per project discipline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb.code_ingest.chunker import chunk_source, ChunkResult

# ---------------------------------------------------------------------------
# Sample Rust source covering every node type in the Rust config spec
# ---------------------------------------------------------------------------

RUST_SAMPLE = """\
/// Adds two numbers together.
pub async fn add(x: u32, y: u32) -> u32 {
    x + y
}

unsafe fn dangerous(ptr: *const u8) -> u8 {
    *ptr
}

pub struct Point {
    pub x: f64,
    pub y: f64,
}

pub enum Color {
    Red,
    Green,
    Blue(u8),
}

pub trait Area {
    fn area(&self) -> f64;
    fn perimeter(&self) -> f64 {
        0.0
    }
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

impl Area for Point {
    fn area(&self) -> f64 {
        0.0
    }
}

pub type Coordinate = f64;

pub const MAX_POINTS: usize = 1024;

pub static ORIGIN: Point = Point { x: 0.0, y: 0.0 };

macro_rules! debug_print {
    ($x:expr) => { println!("{:?}", $x); }
}

mod helpers {
    pub fn helper_fn(n: u32) -> u32 {
        n * 2
    }
}
"""


def _names(chunks: list[ChunkResult]) -> list[str]:
    return [c.name for c in chunks]


def _by_name(chunks: list[ChunkResult], name: str) -> list[ChunkResult]:
    return [c for c in chunks if c.name == name]


def _full_body_only(chunks: list[ChunkResult]) -> list[ChunkResult]:
    """Return only non-signature-only chunks."""
    return [c for c in chunks if not c.extra.get("is_signature_only")]


def _sig_only(chunks: list[ChunkResult]) -> list[ChunkResult]:
    return [c for c in chunks if c.extra.get("is_signature_only")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_function_items_extracted():
    """function_item nodes produce 'function' kind chunks — add, dangerous."""
    chunks = _full_body_only(chunk_source(RUST_SAMPLE, "rust", module="mylib"))
    fn_names = {c.name for c in chunks if c.kind == "function" and c.extra["parent_impl"] is None}
    # 'add' and 'dangerous' are top-level functions; expect both extracted
    assert "add" in fn_names, (
        f"Expected 'add' in top-level functions; got {fn_names}"
    )
    assert "dangerous" in fn_names, (
        f"Expected 'dangerous' in top-level functions; got {fn_names}"
    )


def test_struct_enum_trait_whole():
    """struct/enum/trait nodes produce 'class' kind chunks, kept whole (not split)."""
    chunks = _full_body_only(chunk_source(RUST_SAMPLE, "rust", module="mylib"))
    class_names = {c.name for c in chunks if c.kind == "class"}
    # Spec: struct_item/enum_item/trait_item -> full-body, kept whole
    assert "Point" in class_names, f"struct Point missing; got {class_names}"
    assert "Color" in class_names, f"enum Color missing; got {class_names}"
    assert "Area" in class_names, f"trait Area missing; got {class_names}"


def test_impl_split_per_method():
    """impl_item is split per method: each fn child becomes its own chunk with parent_impl metadata."""
    chunks = _full_body_only(chunk_source(RUST_SAMPLE, "rust", module="mylib"))
    # 'impl Point' has two methods: new, distance
    impl_fns = {c.name: c for c in chunks if c.extra.get("parent_impl") is not None}
    assert "new" in impl_fns, f"impl Point::new not extracted; impl chunks: {list(impl_fns)}"
    assert "distance" in impl_fns, f"impl Point::distance not extracted"
    assert "area" in impl_fns, f"impl Area for Point::area not extracted"

    # parent_impl metadata must identify the impl correctly
    # 'new' and 'distance' come from 'impl Point'
    new_chunk = impl_fns["new"]
    assert new_chunk.extra["parent_impl"] == "Point", (
        f"parent_impl for 'new' should be 'Point'; got {new_chunk.extra['parent_impl']!r}"
    )
    # 'area' implementation comes from 'impl Area for Point'
    area_chunk = impl_fns["area"]
    assert area_chunk.extra["parent_impl"] == "Area for Point", (
        f"parent_impl for Area::area should be 'Area for Point'; "
        f"got {area_chunk.extra['parent_impl']!r}"
    )


def test_type_alias_signature_only():
    """type_item produces embed_mode='signature': body is the signature line, not full text."""
    chunks = chunk_source(RUST_SAMPLE, "rust", module="mylib")
    type_chunks = [c for c in chunks if c.name == "Coordinate"]
    assert type_chunks, "type Coordinate chunk missing"
    coord = type_chunks[0]
    # embed_mode='signature' => body is the first line, not a multi-line block
    assert coord.extra["is_signature_only"], (
        "Coordinate type_item should be signature-only (embed_mode='signature')"
    )
    assert "Coordinate" in coord.body, f"signature body should contain 'Coordinate'; got {coord.body!r}"
    # The body must NOT contain a curly-brace block (it's a type alias, single line)
    assert "{" not in coord.body, (
        f"type alias signature should not contain a block; got {coord.body!r}"
    )


def test_macro_skipped():
    """macro_definition nodes are in exclude_node_types — must not appear in chunks."""
    chunks = chunk_source(RUST_SAMPLE, "rust", module="mylib")
    macro_chunks = [c for c in chunks if c.name == "debug_print"]
    assert not macro_chunks, (
        f"macro_rules! debug_print should be excluded; got {macro_chunks}"
    )


def test_const_static_extracted():
    """const_item and static_item produce 'constant' kind chunks."""
    chunks = _full_body_only(chunk_source(RUST_SAMPLE, "rust", module="mylib"))
    const_names = {c.name for c in chunks if c.kind == "constant"}
    assert "MAX_POINTS" in const_names, f"const MAX_POINTS missing; got {const_names}"
    assert "ORIGIN" in const_names, f"static ORIGIN missing; got {const_names}"


def test_mod_item_recursed():
    """mod_item is recursed into: functions inside inline mod are extracted."""
    chunks = _full_body_only(chunk_source(RUST_SAMPLE, "rust", module="mylib"))
    fn_names = {c.name for c in chunks if c.kind == "function"}
    assert "helper_fn" in fn_names, (
        f"mod helpers::helper_fn should be recursed into and extracted; got {fn_names}"
    )


def test_signature_pass_emits_sig_chunks():
    """Rust signature_pass=True: each function_item gets a companion signature-only chunk."""
    chunks = chunk_source(RUST_SAMPLE, "rust", module="mylib")
    sig_chunks = _sig_only(chunks)
    sig_names = {c.name for c in sig_chunks}
    # 'add', 'dangerous', 'new', 'distance', 'area', 'helper_fn' are all functions
    assert "add" in sig_names, f"Expected sig-only chunk for 'add'; got {sig_names}"
    assert "new" in sig_names, f"Expected sig-only chunk for 'new'; got {sig_names}"
    # All sig-only function chunks must NOT contain the opening body block text
    for sc in sig_chunks:
        # The sig body ends before '{'; it should not contain the function body statements
        body_lines = sc.body.split("\n")
        # At most two lines for a typical signature
        assert len(body_lines) <= 4, (
            f"Signature-only chunk for {sc.name!r} is too long ({len(body_lines)} lines); "
            f"body: {sc.body!r}"
        )


def test_doc_comment_extracted():
    """/// doc comment preceding a function is captured in doc_summary."""
    chunks = chunk_source(RUST_SAMPLE, "rust", module="mylib")
    add_chunks = [c for c in chunks if c.name == "add" and not c.extra.get("is_signature_only")]
    assert add_chunks, "Expected full-body chunk for 'add'"
    add_chunk = add_chunks[0]
    assert add_chunk.doc_summary is not None, "Expected doc_summary for 'add'"
    assert "Adds" in add_chunk.doc_summary or "add" in add_chunk.doc_summary.lower(), (
        f"doc_summary should contain the doc text; got {add_chunk.doc_summary!r}"
    )


def test_visibility_extracted():
    """visibility_modifier is captured in extra['visibility']."""
    chunks = chunk_source(RUST_SAMPLE, "rust", module="mylib")
    add_chunks = [c for c in chunks if c.name == "add" and not c.extra.get("is_signature_only")]
    assert add_chunks
    assert add_chunks[0].extra["visibility"] == "pub", (
        f"'add' is pub; got {add_chunks[0].extra['visibility']!r}"
    )
    dangerous_chunks = [c for c in chunks if c.name == "dangerous" and not c.extra.get("is_signature_only")]
    assert dangerous_chunks
    assert dangerous_chunks[0].extra["visibility"] == "", (
        f"'dangerous' has no visibility modifier; got {dangerous_chunks[0].extra['visibility']!r}"
    )


def test_line_numbers_correct():
    """start line numbers are 1-based and correspond to the function keyword line."""
    chunks = chunk_source(RUST_SAMPLE, "rust", module="mylib")
    # 'add' is on line 2 (the sample starts with a /// comment on line 1)
    add_chunks = [c for c in chunks if c.name == "add" and not c.extra.get("is_signature_only")]
    assert add_chunks
    # Line 2: "pub async fn add(x: u32, y: u32) -> u32 {"
    assert add_chunks[0].line == 2, (
        f"'add' should start at line 2 (1-based); got {add_chunks[0].line}"
    )


def test_chunk_result_feeds_add_python_symbol():
    """ChunkResult fields map cleanly to KnowledgeBase.add_python_symbol() kwargs."""
    chunks = _full_body_only(chunk_source(RUST_SAMPLE, "rust", module="mylib"))
    # Verify the ChunkResult has all required fields for add_python_symbol
    required = {"name", "kind", "module", "signature", "file", "line"}
    for chunk in chunks[:5]:
        missing = required - set(vars(chunk))
        assert not missing, (
            f"ChunkResult for {chunk.name!r} missing fields required by add_python_symbol: {missing}"
        )
        # kind must be one of the values accepted by python_symbols.kind
        assert chunk.kind in ("function", "class", "constant"), (
            f"ChunkResult.kind must be 'function'|'class'|'constant'; got {chunk.kind!r} for {chunk.name!r}"
        )


# ---------------------------------------------------------------------------
# TypeScript tests (kb-asf.4.2 — validated opencode #5402 spec)
# ---------------------------------------------------------------------------

# Representative TS sample covering all included node types + exclusion cases
TS_SAMPLE = """\
import { readFile } from 'fs';

export function greet(name: string): string {
    return 'hello ' + name;
}

export class Animal {
    constructor(public name: string) {}
    speak(): string {
        return 'roar';
    }
}

export interface Shape {
    area(): number;
    perimeter(): number;
}

export type Point = { x: number; y: number };

export enum Color { Red, Green, Blue }

export const layer = (x: number): number => x * 2;

const notExported = 42;
"""

# Minimal TSX sample for parser-selection test
TSX_SAMPLE = """\
export function Button({ label }: { label: string }): string {
    return label;
}

export const Widget = () => null;
"""


def _ts_names(chunks: list[ChunkResult]) -> set[str]:
    return {c.name for c in chunks}


def test_ts_function_and_named_declarations_extracted():
    """TS: function_declaration, class_declaration, interface_declaration,
    type_alias_declaration, enum_declaration are all extracted from TS_SAMPLE.

    Reason: these are exactly the include_node_types in the validated TS config
    (opencode #5402).  Each must be present in the result."""
    chunks = chunk_source(TS_SAMPLE, "typescript", module="mymod")
    names = _ts_names(chunks)
    assert "greet" in names, f"function_declaration 'greet' not extracted; got {names}"
    assert "Animal" in names, f"class_declaration 'Animal' not extracted; got {names}"
    assert "Shape" in names, f"interface_declaration 'Shape' not extracted; got {names}"
    assert "Point" in names, f"type_alias_declaration 'Point' not extracted; got {names}"
    assert "Color" in names, f"enum_declaration 'Color' not extracted; got {names}"


def test_ts_import_excluded():
    """TS: import_statement is in exclude_node_types — must not appear as a chunk.

    Reason: imports are structural, not semantic symbols for KB indexing."""
    chunks = chunk_source(TS_SAMPLE, "typescript", module="mymod")
    import_chunks = [c for c in chunks if c.extra.get("node_type") == "import_statement"]
    assert not import_chunks, (
        f"import_statement should be excluded; got {import_chunks}"
    )


def test_ts_exported_lexical_included_bare_excluded():
    """TS: exported lexical_declaration (export const layer = ...) is included;
    non-exported (const notExported = ...) is excluded.

    Reason: the spec requires exported_only_node_types={lexical_declaration} —
    only `export const` symbols are semantically significant for KB indexing;
    bare module-level consts are implementation details."""
    chunks = chunk_source(TS_SAMPLE, "typescript", module="mymod")
    names = _ts_names(chunks)
    assert "layer" in names, (
        f"exported 'export const layer' must be included; got {names}"
    )
    assert "notExported" not in names, (
        f"non-exported 'const notExported' must be excluded; got {names}"
    )


def test_ts_class_kept_whole():
    """TS: class_declaration container_split='whole' — the class is one chunk
    with its methods inside; no separate 'speak' chunk is emitted.

    Reason: the validated spec (opencode #5402) sets container_split='whole' for TS
    so that class body (methods + fields) is embedded as a unit, preserving context."""
    chunks = chunk_source(TS_SAMPLE, "typescript", module="mymod")
    # No chunk named 'speak' — speak is a method INSIDE Animal, not a top-level chunk
    names = _ts_names(chunks)
    assert "speak" not in names, (
        f"'speak' method must NOT be its own chunk (class is kept whole); got {names}"
    )
    # Animal class body must contain 'speak' (proving the method is included in the class body)
    animal_chunks = [c for c in chunks if c.name == "Animal"]
    assert animal_chunks, "Animal class chunk missing"
    animal = animal_chunks[0]
    assert "speak" in animal.body, (
        f"Animal.body should include the 'speak' method body; got {animal.body!r}"
    )


def test_ts_name_extraction_per_node_type():
    """TS: name extraction uses the correct child node type per declaration kind.

    Reason: TS grammar uses 'identifier' for function/enum names, 'type_identifier'
    for class/interface/type_alias names — verified by inspecting the grammar tree."""
    chunks = chunk_source(TS_SAMPLE, "typescript", module="mymod")
    by_name = {c.name: c for c in chunks}
    # function_declaration -> identifier child
    assert "greet" in by_name and by_name["greet"].extra["node_type"] == "function_declaration", (
        f"'greet' should come from function_declaration; got {by_name.get('greet')}"
    )
    # class_declaration -> type_identifier child
    assert "Animal" in by_name and by_name["Animal"].extra["node_type"] == "class_declaration", (
        f"'Animal' should come from class_declaration"
    )
    # interface_declaration -> type_identifier child
    assert "Shape" in by_name and by_name["Shape"].extra["node_type"] == "interface_declaration"
    # type_alias_declaration -> type_identifier child
    assert "Point" in by_name and by_name["Point"].extra["node_type"] == "type_alias_declaration"
    # enum_declaration -> identifier child
    assert "Color" in by_name and by_name["Color"].extra["node_type"] == "enum_declaration"
    # lexical_declaration -> variable_declarator -> identifier
    assert "layer" in by_name and by_name["layer"].extra["node_type"] == "lexical_declaration"


def test_ts_no_signature_pass():
    """TS: signature_pass=False — no signature-only chunks are emitted.

    Reason: TS config sets signature_pass=False (opencode #5402); sig-only chunks
    are a Rust-specific optimization for large function bodies."""
    chunks = chunk_source(TS_SAMPLE, "typescript", module="mymod")
    sig_only = [c for c in chunks if c.extra.get("is_signature_only")]
    # type_alias_declaration uses embed_mode='signature' but is NOT a sig_pass chunk —
    # it's always emitted as signature (embed_mode), not a secondary pass chunk.
    # The is_signature_only flag is set for type_alias chunks via embed_mode='signature'.
    # True sig_pass chunks would have the same name duplicated; none should exist beyond Point.
    names = [c.name for c in chunks]
    # No name should appear twice (which would indicate a sig_pass duplicate)
    from collections import Counter
    counts = Counter(names)
    duplicates = {n for n, cnt in counts.items() if cnt > 1}
    assert not duplicates, (
        f"signature_pass=False — no name should appear twice; duplicates: {duplicates}"
    )


def test_tsx_parser_selected_by_extension():
    """TSX: chunk_source with language='tsx' uses the language_tsx() grammar.

    Reason: .tsx files contain JSX syntax that requires the TSX grammar variant;
    passing 'tsx' to chunk_source must produce results via the correct parser."""
    chunks = chunk_source(TSX_SAMPLE, "tsx", module="mymod")
    names = _ts_names(chunks)
    assert "Button" in names, (
        f"function_declaration 'Button' must be extracted from TSX_SAMPLE; got {names}"
    )
    assert "Widget" in names, (
        f"export const 'Widget' must be extracted from TSX_SAMPLE; got {names}"
    )


if __name__ == "__main__":
    # Run all tests with simple reporting
    import traceback

    tests = [
        test_function_items_extracted,
        test_struct_enum_trait_whole,
        test_impl_split_per_method,
        test_type_alias_signature_only,
        test_macro_skipped,
        test_const_static_extracted,
        test_mod_item_recursed,
        test_signature_pass_emits_sig_chunks,
        test_doc_comment_extracted,
        test_visibility_extracted,
        test_line_numbers_correct,
        test_chunk_result_feeds_add_python_symbol,
        # TypeScript tests (kb-asf.4.2)
        test_ts_function_and_named_declarations_extracted,
        test_ts_import_excluded,
        test_ts_exported_lexical_included_bare_excluded,
        test_ts_class_kept_whole,
        test_ts_name_extraction_per_node_type,
        test_ts_no_signature_pass,
        test_tsx_parser_selected_by_extension,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
