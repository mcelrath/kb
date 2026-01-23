"""
Database Schema

Schema initialization and migrations for the knowledge base.
"""

import sqlite3


SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS findings (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL CHECK(type IN ('success', 'failure', 'experiment', 'discovery', 'correction')),
        status TEXT DEFAULT 'current' CHECK(status IN ('current', 'superseded')),
        supersedes_id TEXT REFERENCES findings(id),
        project TEXT,
        sprint TEXT,
        tags TEXT,  -- JSON array
        content TEXT NOT NULL,
        summary TEXT,  -- LLM-generated one-line summary for search results
        evidence TEXT,  -- Supporting evidence (log snippets, test output)
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_findings_status ON findings(status);
    CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(type);
    CREATE INDEX IF NOT EXISTS idx_findings_project ON findings(project);
    CREATE INDEX IF NOT EXISTS idx_findings_supersedes ON findings(supersedes_id);
    CREATE INDEX IF NOT EXISTS idx_findings_created_at ON findings(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_findings_project_status ON findings(project, status);

    CREATE VIRTUAL TABLE IF NOT EXISTS findings_fts USING fts5(
        content, evidence, tags,
        content='findings',
        content_rowid='rowid'
    );

    -- Notation tracking tables
    CREATE TABLE IF NOT EXISTS notations (
        id TEXT PRIMARY KEY,
        current_symbol TEXT NOT NULL,
        meaning TEXT NOT NULL,
        project TEXT,
        domain TEXT CHECK(domain IN ('physics', 'math', 'cs', 'general')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS notation_history (
        id TEXT PRIMARY KEY,
        notation_id TEXT NOT NULL REFERENCES notations(id),
        old_symbol TEXT NOT NULL,
        new_symbol TEXT NOT NULL,
        reason TEXT,
        changed_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_notations_project ON notations(project);
    CREATE INDEX IF NOT EXISTS idx_notations_symbol ON notations(current_symbol);
    CREATE INDEX IF NOT EXISTS idx_notations_project_symbol ON notations(project, current_symbol);
    CREATE INDEX IF NOT EXISTS idx_notation_history_notation ON notation_history(notation_id);

    -- Error tracking and solution linking
    CREATE TABLE IF NOT EXISTS errors (
        id TEXT PRIMARY KEY,
        signature TEXT NOT NULL,  -- Error message or pattern
        error_type TEXT,  -- build, runtime, test, etc.
        project TEXT,
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        occurrence_count INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS error_solutions (
        error_id TEXT NOT NULL REFERENCES errors(id),
        finding_id TEXT NOT NULL REFERENCES findings(id),
        linked_at TEXT NOT NULL,
        verified INTEGER DEFAULT 0,  -- 1 if solution was confirmed to work
        PRIMARY KEY (error_id, finding_id)
    );

    -- Authoritative documents (specs, papers, standards)
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        url TEXT,  -- URL or file path
        doc_type TEXT NOT NULL CHECK(doc_type IN ('spec', 'paper', 'standard', 'internal', 'reference')),
        project TEXT,
        status TEXT DEFAULT 'active' CHECK(status IN ('active', 'superseded', 'deprecated')),
        summary TEXT,  -- Brief description of the document
        created_at TEXT NOT NULL,
        superseded_by TEXT REFERENCES documents(id)
    );

    -- Links between findings and documents they cite
    CREATE TABLE IF NOT EXISTS document_citations (
        finding_id TEXT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        citation_type TEXT DEFAULT 'references' CHECK(citation_type IN ('references', 'implements', 'contradicts', 'extends')),
        notes TEXT,
        cited_at TEXT NOT NULL,
        PRIMARY KEY (finding_id, document_id)
    );

    CREATE INDEX IF NOT EXISTS idx_errors_project ON errors(project);
    CREATE INDEX IF NOT EXISTS idx_errors_signature ON errors(signature);
    CREATE INDEX IF NOT EXISTS idx_error_solutions_error ON error_solutions(error_id);
    CREATE INDEX IF NOT EXISTS idx_error_solutions_finding ON error_solutions(finding_id);
    CREATE INDEX IF NOT EXISTS idx_error_solutions_verified ON error_solutions(verified);
    CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project);
    CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(doc_type);
    CREATE INDEX IF NOT EXISTS idx_document_citations_doc ON document_citations(document_id);
    CREATE INDEX IF NOT EXISTS idx_document_citations_finding ON document_citations(finding_id);

    -- Script registry for tracking hypothesis-testing scripts
    CREATE TABLE IF NOT EXISTS scripts (
        id TEXT PRIMARY KEY,
        path TEXT NOT NULL,  -- Original file path
        filename TEXT NOT NULL,  -- Just the filename
        content_hash TEXT NOT NULL,  -- SHA256 of content for deduplication
        content TEXT,  -- Full script content (optional, for small scripts)
        purpose TEXT NOT NULL,  -- What hypothesis/question this script tests
        project TEXT,
        language TEXT DEFAULT 'python' CHECK(language IN ('python', 'sage', 'bash', 'other')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    -- Links between findings and scripts that generated them
    CREATE TABLE IF NOT EXISTS finding_scripts (
        finding_id TEXT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
        script_id TEXT NOT NULL REFERENCES scripts(id) ON DELETE CASCADE,
        relationship TEXT DEFAULT 'generated_by' CHECK(relationship IN ('generated_by', 'validates', 'contradicts')),
        linked_at TEXT NOT NULL,
        PRIMARY KEY (finding_id, script_id)
    );

    CREATE INDEX IF NOT EXISTS idx_scripts_project ON scripts(project);
    CREATE INDEX IF NOT EXISTS idx_scripts_hash ON scripts(content_hash);
    CREATE INDEX IF NOT EXISTS idx_scripts_filename ON scripts(filename);
    CREATE INDEX IF NOT EXISTS idx_finding_scripts_script ON finding_scripts(script_id);
    CREATE INDEX IF NOT EXISTS idx_finding_scripts_finding ON finding_scripts(finding_id);

    CREATE TRIGGER IF NOT EXISTS findings_ai AFTER INSERT ON findings BEGIN
        INSERT INTO findings_fts(rowid, content, evidence, tags)
        VALUES (new.rowid, new.content, new.evidence, new.tags);
    END;

    CREATE TRIGGER IF NOT EXISTS findings_ad AFTER DELETE ON findings BEGIN
        INSERT INTO findings_fts(findings_fts, rowid, content, evidence, tags)
        VALUES ('delete', old.rowid, old.content, old.evidence, old.tags);
    END;

    CREATE TRIGGER IF NOT EXISTS findings_au AFTER UPDATE ON findings BEGIN
        INSERT INTO findings_fts(findings_fts, rowid, content, evidence, tags)
        VALUES ('delete', old.rowid, old.content, old.evidence, old.tags);
        INSERT INTO findings_fts(rowid, content, evidence, tags)
        VALUES (new.rowid, new.content, new.evidence, new.tags);
    END;
"""


def init_schema(conn: sqlite3.Connection, embedding_dim: int) -> None:
    """Initialize database schema."""
    _ = conn.executescript(SCHEMA_SQL)

    # Create vector table for embeddings
    _ = conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS findings_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
    """)

    # Create vector table for script purpose embeddings
    _ = conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS scripts_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
    """)

    # Schema migration: add summary column if not exists
    try:
        _ = conn.execute("SELECT summary FROM findings LIMIT 1")
    except sqlite3.OperationalError:
        _ = conn.execute("ALTER TABLE findings ADD COLUMN summary TEXT")

    conn.commit()
