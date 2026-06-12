"""
kb/code_ingest/chunker.py — Generalized tree-sitter code chunker.

A language-config table keyed by grammar maps each grammar node type to a
ChunkSpec describing how to extract chunks from that node.  The per-language
config supplies knobs that DIVERGE across languages:

  - container_split: "split" | "whole"
      "split"  — an impl/class container is split per method (each child fn = own chunk)
      "whole"  — the container is emitted as one chunk with all children included

  - signature_pass: bool
      True  — a second pass emits a signature-only chunk for each function
              (body text up to and including the first '{')
      False — no signature pass

  - include_node_types: set[str] | None
      When set, only nodes whose type is in this set are considered.
      None means "all named children".

  - exclude_node_types: set[str]
      Node types whose top-level occurrences are SKIPPED (not recursed into).
      Children of excluded nodes are also not extracted.

  - exported_only_node_types: set[str]
      Node types that are only emitted when their immediate tree-sitter parent is
      an export_statement.  Used for TypeScript lexical_declaration (export const).

  - grammar_language_fn: str
      Name of the function to call on the grammar module to get the Language
      object.  Defaults to "language".  TypeScript uses "language_typescript";
      TSX uses "language_tsx".

  - name_fn: callable | None
      Called as name_fn(node, src) to extract a symbol name.  If None, falls
      back to _rust_name (first identifier/type_identifier child).

  - sig_fn: callable | None
      Called as sig_fn(node, src) to extract a signature string.  If None,
      falls back to _rust_signature.

ChunkResult mirrors the dict shape returned by parse_python_file() in
ingest_python.py so chunks feed directly into KnowledgeBase.add_python_symbol().
The 'kind' field maps: fn -> 'function', struct/enum/trait/type -> 'class',
const/static -> 'constant'.

Integration point
-----------------
Call chunk_file(path, language_name) → list[ChunkResult].
Each ChunkResult can be forwarded to KnowledgeBase.add_python_symbol() with:
    kb.add_python_symbol(
        name=r.name,
        kind=r.kind,
        module=r.module,
        signature=r.signature,
        file=r.file,
        line=r.line,
        status="public",
        docstring_summary=r.doc_summary,
        project=project,
    )

The 'parent_impl' and 'visibility' metadata in ChunkResult.extra are not part of
the current python_symbols schema; a schema extension (add columns
parent_container TEXT, visibility TEXT, language TEXT) is the follow-up tracked
in kb-asf.4.  Until then, extra metadata is available on the result dict for
callers that query it directly.

chunk_file() auto-selects the TypeScript vs TSX grammar by file extension:
  .ts  -> LANG_CONFIGS["typescript"] (language_typescript())
  .tsx -> LANG_CONFIGS["tsx"]        (language_tsx())
Both share identical chunk specs; only the grammar differs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    """One extracted code chunk, ready for KB ingestion."""
    name: str           # symbol name (fn name, struct name, etc.)
    kind: str           # 'function' | 'class' | 'constant'
    module: str         # dotted module path (file-relative to repo root)
    signature: str      # first line / signature text
    file: str           # absolute file path
    line: int           # 1-based start line
    end_line: int       # 1-based end line
    body: str           # full chunk source text
    doc_summary: str | None   # leading doc comment (first /// block)
    extra: dict[str, Any] = field(default_factory=dict)
    # extra keys used by Rust config:
    #   parent_impl: str | None  — "Foo" or "Greet for Foo"
    #   visibility: str          — "pub" | "pub(crate)" | "" etc.
    #   is_signature_only: bool  — True for the signature-only pass chunk


# ---------------------------------------------------------------------------
# Language config
# ---------------------------------------------------------------------------

@dataclass
class ChunkSpec:
    """How to handle one grammar node type."""
    kind: str                          # KB kind: 'function' | 'class' | 'constant'
    embed_mode: str                    # 'full' | 'signature'
    container_split: str = "whole"     # 'whole' | 'split' — for container nodes
    # 'split' is only meaningful when the node type is a container (impl_item, class).


@dataclass
class LangConfig:
    """Per-language chunker configuration."""
    grammar_module: str            # importable Python module name, e.g. 'tree_sitter_rust'
    file_extensions: tuple[str, ...]
    chunk_specs: dict[str, ChunkSpec]    # node_type -> ChunkSpec
    exclude_node_types: set[str]         # top-level nodes to skip (not recurse into)
    signature_pass: bool                 # emit a signature-only chunk for each function
    doc_node_types: set[str]             # node types that carry doc comments
    # Optional knobs (language-specific extensions)
    exported_only_node_types: set[str] = field(default_factory=set)
    # Node types only emitted when immediate parent is export_statement (TypeScript lexical_declaration)
    grammar_language_fn: str = "language"
    # Function name on grammar_module that returns the Language object.
    # Defaults to "language"; TypeScript uses "language_typescript", TSX uses "language_tsx".
    name_fn: Callable[[Any, bytes], str] | None = None
    # Override for symbol name extraction.  None -> falls back to _rust_name.
    sig_fn: Callable[[Any, bytes], str] | None = None
    # Override for signature extraction.  None -> falls back to _rust_signature.


# Rust language config (verified spec from goose #5372, bd kb-asf.4 notes)
_RUST_SPECS: dict[str, ChunkSpec] = {
    # Functions — full body, including async/unsafe variants
    "function_item": ChunkSpec(kind="function", embed_mode="full"),

    # Structs, enums, traits — full body, kept whole
    "struct_item": ChunkSpec(kind="class", embed_mode="full"),
    "enum_item":   ChunkSpec(kind="class", embed_mode="full"),
    "trait_item":  ChunkSpec(kind="class", embed_mode="full"),

    # impl block — split per method: each function_item child = own chunk
    "impl_item": ChunkSpec(kind="function", embed_mode="full", container_split="split"),

    # Type aliases — signature only (drop = rhs_type; just name + alias target)
    "type_item": ChunkSpec(kind="class", embed_mode="signature"),

    # Constants and statics — full
    "const_item":  ChunkSpec(kind="constant", embed_mode="full"),
    "static_item": ChunkSpec(kind="constant", embed_mode="full"),
}

RUST_CONFIG = LangConfig(
    grammar_module="tree_sitter_rust",
    file_extensions=(".rs",),
    chunk_specs=_RUST_SPECS,
    exclude_node_types={
        "macro_definition",     # skip macro_rules! definitions
        # inline mod_item: skip (recurse into its children instead)
        # handled in _extract_chunks by special-casing mod_item
    },
    signature_pass=True,
    doc_node_types={"line_comment"},   # /// line_comment nodes preceding the item
)

# ---------------------------------------------------------------------------
# TypeScript / TSX language config (kb-asf.4.2 — validated by opencode #5402,
# 2503 files / 13221 symbols)
# ---------------------------------------------------------------------------
# Spec:
#   container_split="whole"  — class kept whole with all methods
#   signature_pass=False
#   include_node_types: function_declaration, class_declaration,
#                       interface_declaration, type_alias_declaration,
#                       enum_declaration, lexical_declaration
#   exported_only_node_types: {lexical_declaration}
#       — lexical_declaration is only emitted when parent == export_statement
#         (i.e. `export const layer = ...`; bare `const x = ...` is skipped)
#   exclude_node_types: {import_statement}
#   name extraction per node type:
#       function_declaration / class_declaration / enum_declaration -> identifier
#       interface_declaration / type_alias_declaration -> type_identifier
#       lexical_declaration -> variable_declarator child -> identifier grandchild
#   .ts  -> language_typescript()   .tsx -> language_tsx()
# ---------------------------------------------------------------------------

def _ts_name(node: Any, src: bytes) -> str:
    """Extract symbol name from a TypeScript AST node.

    Node-type-specific name extraction per validated opencode spec (#5402):
      - function_declaration / class_declaration / enum_declaration:
            first 'identifier' child
      - interface_declaration / type_alias_declaration:
            first 'type_identifier' child
      - lexical_declaration (export const ...):
            first 'variable_declarator' child, then its 'identifier' child
    """
    ntype = node.type
    if ntype in ("function_declaration", "enum_declaration"):
        # function and enum names are `identifier` nodes
        for child in node.named_children:
            if child.type == "identifier":
                return src[child.start_byte:child.end_byte].decode(errors="replace")
    elif ntype in ("class_declaration", "interface_declaration", "type_alias_declaration"):
        for child in node.named_children:
            if child.type == "type_identifier":
                return src[child.start_byte:child.end_byte].decode(errors="replace")
    elif ntype == "lexical_declaration":
        # `export const layer = ...`
        # lexical_declaration -> variable_declarator -> identifier (the name)
        for child in node.named_children:
            if child.type == "variable_declarator":
                for gc in child.named_children:
                    if gc.type == "identifier":
                        return src[gc.start_byte:gc.end_byte].decode(errors="replace")
    return f"<unnamed_{ntype}>"


def _ts_signature(node: Any, src: bytes) -> str:
    """Extract a TypeScript signature (first line of the node source)."""
    full_text = src[node.start_byte:node.end_byte].decode(errors="replace")
    return full_text.split("\n")[0]


# Shared chunk_specs for both .ts and .tsx (container_split="whole" for all)
_TS_SPECS: dict[str, ChunkSpec] = {
    "function_declaration":    ChunkSpec(kind="function", embed_mode="full", container_split="whole"),
    "class_declaration":       ChunkSpec(kind="class",    embed_mode="full", container_split="whole"),
    "interface_declaration":   ChunkSpec(kind="class",    embed_mode="full", container_split="whole"),
    "type_alias_declaration":  ChunkSpec(kind="class",    embed_mode="signature", container_split="whole"),
    "enum_declaration":        ChunkSpec(kind="class",    embed_mode="full", container_split="whole"),
    "lexical_declaration":     ChunkSpec(kind="constant", embed_mode="full", container_split="whole"),
}

TYPESCRIPT_CONFIG = LangConfig(
    grammar_module="tree_sitter_typescript",
    file_extensions=(".ts",),
    chunk_specs=_TS_SPECS,
    exclude_node_types={"import_statement"},
    signature_pass=False,
    doc_node_types={"comment"},          # // and /* */ comments in TS
    exported_only_node_types={"lexical_declaration"},
    grammar_language_fn="language_typescript",
    name_fn=_ts_name,
    sig_fn=_ts_signature,
)

TSX_CONFIG = LangConfig(
    grammar_module="tree_sitter_typescript",
    file_extensions=(".tsx",),
    chunk_specs=_TS_SPECS,
    exclude_node_types={"import_statement"},
    signature_pass=False,
    doc_node_types={"comment"},
    exported_only_node_types={"lexical_declaration"},
    grammar_language_fn="language_tsx",
    name_fn=_ts_name,
    sig_fn=_ts_signature,
)

# Registry: language name -> LangConfig
LANG_CONFIGS: dict[str, LangConfig] = {
    "rust": RUST_CONFIG,
    "typescript": TYPESCRIPT_CONFIG,
    "tsx": TSX_CONFIG,
}


# ---------------------------------------------------------------------------
# Tree-sitter parser cache (one parser per language, lazy-init)
# ---------------------------------------------------------------------------

_parser_cache: dict[str, Any] = {}   # language_name -> tree_sitter.Parser


def _get_parser(lang_name: str) -> Any:
    if lang_name in _parser_cache:
        return _parser_cache[lang_name]

    cfg = LANG_CONFIGS.get(lang_name)
    if cfg is None:
        raise ValueError(f"No language config for {lang_name!r}. Known: {list(LANG_CONFIGS)}")

    import importlib
    from tree_sitter import Language, Parser

    grammar_mod = importlib.import_module(cfg.grammar_module)
    lang_fn = getattr(grammar_mod, cfg.grammar_language_fn)
    lang = Language(lang_fn())
    parser = Parser(lang)
    _parser_cache[lang_name] = parser
    return parser


# ---------------------------------------------------------------------------
# Doc-comment extraction
# ---------------------------------------------------------------------------

def _collect_doc_before(node: Any, src: bytes, doc_node_types: set[str]) -> str | None:
    """Collect leading doc-comment lines immediately preceding *node*.

    Walks *node*'s preceding named siblings that are doc/line_comment nodes
    and whose end_point.row is contiguous with the next sibling/node.
    Returns the concatenated comment text (stripped of /// prefix), or None.
    """
    lines: list[str] = []
    sib = node.prev_named_sibling
    # Collect contiguous preceding line_comment / doc_comment siblings
    while sib is not None and sib.type in doc_node_types:
        text = src[sib.start_byte:sib.end_byte].decode(errors="replace").strip()
        # Strip leading /// or //!
        text = text.lstrip("/").strip()
        lines.insert(0, text)
        sib = sib.prev_named_sibling
    if not lines:
        return None
    combined = " ".join(l for l in lines if l)
    return combined[:300] if combined else None


# ---------------------------------------------------------------------------
# Signature extraction
# ---------------------------------------------------------------------------

def _rust_signature(node: Any, src: bytes) -> str:
    """Extract a Rust function/type signature (up to but not including the body block).

    For function_item: returns everything from the start through the return type,
    stopping before the opening '{' of the body.
    For type_item: returns the full line (e.g. "pub type Alias = u32;").
    For struct/enum/trait: returns the first line (declaration header).
    """
    full_text = src[node.start_byte:node.end_byte].decode(errors="replace")
    if node.type == "function_item":
        # Find the block child — signature is everything before it
        for child in node.named_children:
            if child.type == "block":
                sig_bytes = src[node.start_byte:child.start_byte]
                return sig_bytes.decode(errors="replace").rstrip()
        # No block found (e.g. function declaration in trait) — return full
        return full_text.split("\n")[0]
    if node.type in ("type_item", "const_item", "static_item"):
        return full_text.split("\n")[0]
    # struct/enum/trait: first line
    return full_text.split("\n")[0]


def _rust_name(node: Any, src: bytes) -> str:
    """Extract the primary identifier/name from a Rust node."""
    # Most nodes carry an 'identifier' or 'type_identifier' child
    for child in node.named_children:
        if child.type in ("identifier", "type_identifier"):
            return src[child.start_byte:child.end_byte].decode(errors="replace")
    return f"<unnamed_{node.type}>"


def _rust_visibility(node: Any, src: bytes) -> str:
    """Extract visibility modifier text, e.g. 'pub', 'pub(crate)', or ''."""
    for child in node.named_children:
        if child.type == "visibility_modifier":
            return src[child.start_byte:child.end_byte].decode(errors="replace")
    return ""


def _rust_impl_label(node: Any, src: bytes) -> str:
    """Produce a human-readable label for an impl_item, e.g. 'Foo' or 'Greet for Foo'."""
    # impl_item named children: [generic_type_arguments?] type_identifier [type_identifier]
    # "impl Trait for Type" => two type_identifiers
    type_ids = [
        src[c.start_byte:c.end_byte].decode(errors="replace")
        for c in node.named_children
        if c.type in ("type_identifier", "generic_type", "scoped_type_identifier")
    ]
    if len(type_ids) >= 2:
        return f"{type_ids[0]} for {type_ids[-1]}"
    if len(type_ids) == 1:
        return type_ids[0]
    return "<impl>"


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def _file_to_module(file_path: Path, root: Path | None) -> str:
    """Convert file path to dotted module name relative to root."""
    if root is not None:
        try:
            rel = file_path.relative_to(root)
            parts = list(rel.parts)
            if parts and parts[-1].endswith(".rs"):
                parts[-1] = parts[-1][:-3]
            if parts and parts[-1] == "mod":
                parts.pop()
            return "::".join(parts)
        except ValueError:
            pass
    return file_path.stem


def _extract_chunks(
    node: Any,
    src: bytes,
    cfg: LangConfig,
    file_path: Path,
    module: str,
    parent_impl: str | None = None,
) -> list[ChunkResult]:
    """Recursively extract chunks from *node* per language config."""
    results: list[ChunkResult] = []

    for child in node.named_children:
        ntype = child.type

        # Skip explicitly excluded top-level node types
        if ntype in cfg.exclude_node_types:
            continue

        # Recurse into mod_item (inline module) — don't emit the mod itself
        if ntype == "mod_item":
            decl_list = child.child_by_field_name("body")
            if decl_list is not None:
                results.extend(_extract_chunks(decl_list, src, cfg, file_path, module, parent_impl))
            continue

        # Recurse into export_statement (TypeScript) — the declaration is a child
        # e.g. `export function f() {}` -> export_statement -> function_declaration
        # The declaration child will be processed in the recursive call with its
        # parent correctly set to the export_statement node.
        if ntype == "export_statement":
            results.extend(_extract_chunks(child, src, cfg, file_path, module, parent_impl))
            continue

        spec = cfg.chunk_specs.get(ntype)
        if spec is None:
            continue

        # impl_item with container_split="split": emit one chunk per method
        if ntype == "impl_item" and spec.container_split == "split":
            impl_label = _rust_impl_label(child, src)
            decl_list = child.child_by_field_name("body")
            if decl_list is None:
                # Try named_children fallback
                for gc in child.named_children:
                    if gc.type == "declaration_list":
                        decl_list = gc
                        break
            if decl_list is not None:
                results.extend(_extract_chunks(decl_list, src, cfg, file_path, module, impl_label))
            continue

        # exported_only filter: skip this node unless its parent is export_statement
        if ntype in cfg.exported_only_node_types:
            if child.parent is None or child.parent.type != "export_statement":
                continue

        # Normal chunk emission
        _name_fn = cfg.name_fn if cfg.name_fn is not None else _rust_name
        _sig_fn = cfg.sig_fn if cfg.sig_fn is not None else _rust_signature
        name = _name_fn(child, src)
        visibility = _rust_visibility(child, src)
        doc_summary = _collect_doc_before(child, src, cfg.doc_node_types)
        start_line = child.start_point[0] + 1  # 1-based
        end_line = child.end_point[0] + 1
        body = src[child.start_byte:child.end_byte].decode(errors="replace")

        if spec.embed_mode == "signature":
            sig = _sig_fn(child, src)
            emit_body = sig
        else:
            sig = _sig_fn(child, src)
            emit_body = body

        chunk = ChunkResult(
            name=name,
            kind=spec.kind,
            module=module,
            signature=sig,
            file=str(file_path),
            line=start_line,
            end_line=end_line,
            body=emit_body,
            doc_summary=doc_summary,
            extra={
                "parent_impl": parent_impl,
                "visibility": visibility,
                "is_signature_only": spec.embed_mode == "signature",
                "node_type": ntype,
            },
        )
        results.append(chunk)

        # Signature-only pass for functions (second pass, separate chunk)
        if cfg.signature_pass and spec.kind == "function" and spec.embed_mode != "signature":
            sig_text = _sig_fn(child, src)
            sig_chunk = ChunkResult(
                name=name,
                kind="function",
                module=module,
                signature=sig_text,
                file=str(file_path),
                line=start_line,
                end_line=start_line,
                body=sig_text,
                doc_summary=doc_summary,
                extra={
                    "parent_impl": parent_impl,
                    "visibility": visibility,
                    "is_signature_only": True,
                    "node_type": ntype,
                },
            )
            results.append(sig_chunk)

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _language_for_file(file_path: Path, language: str) -> str:
    """Resolve the language key for *file_path*, honouring .tsx vs .ts disambiguation.

    If *language* is 'typescript' and the file has a .tsx extension, returns
    'tsx' so the TSX grammar is used.  Otherwise returns *language* unchanged.
    """
    if language == "typescript" and file_path.suffix == ".tsx":
        return "tsx"
    return language


def chunk_file(
    file_path: Path | str,
    language: str,
    root: Path | str | None = None,
) -> list[ChunkResult]:
    """Parse *file_path* with tree-sitter and return all code chunks per language config.

    Args:
        file_path: Path to the source file.
        language:  Language key, e.g. 'rust' or 'typescript'.
                   For TypeScript, pass 'typescript'; .tsx files are auto-detected
                   and switched to the TSX grammar.
        root:      Optional repo root for module path computation.

    Returns:
        List of ChunkResult, one per extracted symbol (functions, structs, etc.).
        impl_item nodes are split per method (for Rust config).
        Signature-only chunks are appended after their full-body counterpart.
    """
    file_path = Path(file_path)
    if root is not None:
        root = Path(root)
    language = _language_for_file(file_path, language)

    cfg = LANG_CONFIGS[language]
    parser = _get_parser(language)
    module = _file_to_module(file_path, root)

    src = file_path.read_bytes()
    tree = parser.parse(src)

    return _extract_chunks(tree.root_node, src, cfg, file_path, module)


def chunk_source(
    source: str | bytes,
    language: str,
    file_path: str = "<source>",
    module: str = "<source>",
) -> list[ChunkResult]:
    """Parse *source* text and return chunks.  Useful for testing without a real file."""
    cfg = LANG_CONFIGS[language]
    parser = _get_parser(language)

    if isinstance(source, str):
        src = source.encode()
    else:
        src = source

    tree = parser.parse(src)
    fp = Path(file_path)
    return _extract_chunks(tree.root_node, src, cfg, fp, module)
