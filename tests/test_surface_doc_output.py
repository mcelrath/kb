"""Tests for color + truncation in kb surface (run_surface) and kb doc list/toc."""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from kb.cli import output
from kb.cli.commands.surface import run_surface
from kb.cli.commands.doc import run_doc


# ---------------------------------------------------------------------------
# Helpers — minimal fake kb and args
# ---------------------------------------------------------------------------

def _fake_kb(symbols=(), findings=(), bridge_hits=()):
    kb = MagicMock()
    kb.search_symbols.return_value = list(symbols)
    kb.search.return_value = list(findings)
    kb._bridge.search.return_value = list(bridge_hits)
    return kb


def _surface_args(query="test", limit=5, project=None, min_sim=0.0,
                  sources="code,findings,bridge", as_json=False):
    args = types.SimpleNamespace(
        query=query, limit=limit, project=project, min_sim=min_sim,
        sources=sources, json=as_json,
        # producer modes all None so legacy path is taken
        prompt=None, analysis=None, file=None, issues=None, bridge=None, all_input=None,
    )
    return args


def _doc_list_args(project=None, doc_type=None, as_json=False):
    return types.SimpleNamespace(doc_cmd="list", project=project, type=doc_type, json=as_json)


def _doc_toc_args(doc_id="doc-1", as_json=False):
    return types.SimpleNamespace(doc_cmd="toc", doc_id=doc_id, json=as_json)


_SYMBOL = {"similarity": 0.82, "name": "my_func", "module": "mymod", "kind": "function",
           "file": "mymod/foo.py", "line": 42}
_FINDING = {"similarity": 0.75, "id": "kb-20260101-abc", "project": "kb", "summary": "A test finding"}
_BRIDGE_HIT = {"similarity": 0.55, "id": 7, "sender": "archie", "subject": "hello"}


# ---------------------------------------------------------------------------
# surface — agent mode: plain, no ANSI, no truncation
# ---------------------------------------------------------------------------

def test_surface_agent_mode_no_ansi(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    kb = _fake_kb(symbols=[_SYMBOL], findings=[_FINDING], bridge_hits=[_BRIDGE_HIT])
    run_surface(kb, _surface_args())
    out = capsys.readouterr().out
    assert "\033[" not in out


def test_surface_agent_mode_no_truncation(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    long_name = "x" * 300
    sym = dict(_SYMBOL, name=long_name)
    kb = _fake_kb(symbols=[sym])
    run_surface(kb, _surface_args(sources="code"))
    out = capsys.readouterr().out
    assert long_name in out


# ---------------------------------------------------------------------------
# surface — user mode: colorized, rows fit width
# ---------------------------------------------------------------------------

def test_surface_user_mode_colorized(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 200)
    kb = _fake_kb(symbols=[_SYMBOL], findings=[_FINDING])
    run_surface(kb, _surface_args(sources="code,findings"))
    out = capsys.readouterr().out
    assert "\033[" in out


def test_surface_user_mode_rows_truncated(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 30)
    long_name = "y" * 200
    sym = dict(_SYMBOL, name=long_name)
    kb = _fake_kb(symbols=[sym])
    run_surface(kb, _surface_args(sources="code"))
    out = capsys.readouterr().out
    # each printed row (non-empty, non-header) must be <= 30 visible chars
    for line in out.splitlines():
        if line.strip():
            assert output.visible_len(line) <= 30, f"line too long: {output.visible_len(line)!r}"


# ---------------------------------------------------------------------------
# doc list — agent mode
# ---------------------------------------------------------------------------

def _make_docs_repo(docs):
    repo = MagicMock()
    repo.list.return_value = docs
    return repo


def _kb_with_conn():
    kb = MagicMock()
    return kb


_DOC = {"id": "doc-abc", "project": "kb", "doc_type": "reference",
        "title": "My Document Title", "summary": "A short summary"}


def test_doc_list_agent_no_ansi(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)

    import kb.cli.commands.doc as doc_mod
    from kb.entities.documents import DocumentsRepository

    monkeypatch.setattr(DocumentsRepository, "__init__", lambda self, conn: None)
    monkeypatch.setattr(DocumentsRepository, "list", lambda self, **kw: [_DOC])

    kb = _kb_with_conn()
    run_doc(kb, _doc_list_args(), MagicMock())
    out = capsys.readouterr().out
    assert "\033[" not in out


def test_doc_list_agent_no_truncation(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)

    from kb.entities.documents import DocumentsRepository
    long_title = "T" * 300
    doc = dict(_DOC, title=long_title)
    monkeypatch.setattr(DocumentsRepository, "__init__", lambda self, conn: None)
    monkeypatch.setattr(DocumentsRepository, "list", lambda self, **kw: [doc])

    kb = _kb_with_conn()
    run_doc(kb, _doc_list_args(), MagicMock())
    out = capsys.readouterr().out
    assert long_title in out


def test_doc_list_user_colorized(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 200)

    from kb.entities.documents import DocumentsRepository
    monkeypatch.setattr(DocumentsRepository, "__init__", lambda self, conn: None)
    monkeypatch.setattr(DocumentsRepository, "list", lambda self, **kw: [_DOC])

    kb = _kb_with_conn()
    run_doc(kb, _doc_list_args(), MagicMock())
    out = capsys.readouterr().out
    assert "\033[" in out


def test_doc_list_user_rows_truncated(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 30)

    from kb.entities.documents import DocumentsRepository
    long_title = "Z" * 300
    doc = dict(_DOC, title=long_title)
    monkeypatch.setattr(DocumentsRepository, "__init__", lambda self, conn: None)
    monkeypatch.setattr(DocumentsRepository, "list", lambda self, **kw: [doc])

    kb = _kb_with_conn()
    run_doc(kb, _doc_list_args(), MagicMock())
    out = capsys.readouterr().out
    for line in out.splitlines():
        if line.strip():
            assert output.visible_len(line) <= 30


# ---------------------------------------------------------------------------
# doc toc — agent mode
# ---------------------------------------------------------------------------

_SECTION = {"id": "sec-1", "level": 2, "path": "1.2", "heading": "Introduction",
            "kind": "heading", "content": None, "asset_path": None}


def test_doc_toc_agent_no_ansi(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", True)

    from kb.entities.document_sections import DocumentSectionsRepository
    monkeypatch.setattr(DocumentSectionsRepository, "__init__", lambda self, conn: None)
    monkeypatch.setattr(DocumentSectionsRepository, "list_by_document", lambda self, doc_id: [_SECTION])

    kb = _kb_with_conn()
    run_doc(kb, _doc_toc_args(), MagicMock())
    out = capsys.readouterr().out
    assert "\033[" not in out


def test_doc_toc_user_colorized(monkeypatch, capsys):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 200)

    from kb.entities.document_sections import DocumentSectionsRepository
    monkeypatch.setattr(DocumentSectionsRepository, "__init__", lambda self, conn: None)
    monkeypatch.setattr(DocumentSectionsRepository, "list_by_document", lambda self, doc_id: [_SECTION])

    kb = _kb_with_conn()
    run_doc(kb, _doc_toc_args(), MagicMock())
    out = capsys.readouterr().out
    assert "\033[" in out
