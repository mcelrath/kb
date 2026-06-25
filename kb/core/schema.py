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

    -- Lean theorem index
    CREATE TABLE IF NOT EXISTS lean_theorems (
        id TEXT PRIMARY KEY,
        lean_name TEXT NOT NULL,
        name TEXT NOT NULL,
        statement TEXT NOT NULL,
        statement_pure TEXT,
        declaration TEXT NOT NULL,
        module TEXT,
        file TEXT NOT NULL,
        line INTEGER,
        tex_source TEXT,
        project TEXT,
        tags TEXT,
        finding_id TEXT,  -- FK to findings.id, populated at ingest when a finding cites this theorem
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_lean_theorems_project ON lean_theorems(project);
    CREATE INDEX IF NOT EXISTS idx_lean_theorems_module ON lean_theorems(module);
    CREATE INDEX IF NOT EXISTS idx_lean_theorems_lean_name ON lean_theorems(lean_name);

    CREATE VIRTUAL TABLE IF NOT EXISTS lean_theorems_fts USING fts5(
        name, statement, statement_pure, lean_name,
        content='lean_theorems',
        content_rowid='rowid'
    );

    CREATE TRIGGER IF NOT EXISTS lean_theorems_ai AFTER INSERT ON lean_theorems BEGIN
        INSERT INTO lean_theorems_fts(rowid, name, statement, statement_pure, lean_name)
        VALUES (new.rowid, new.name, new.statement, new.statement_pure, new.lean_name);
    END;

    CREATE TRIGGER IF NOT EXISTS lean_theorems_ad AFTER DELETE ON lean_theorems BEGIN
        INSERT INTO lean_theorems_fts(lean_theorems_fts, rowid, name, statement, statement_pure, lean_name)
        VALUES ('delete', old.rowid, old.name, old.statement, old.statement_pure, old.lean_name);
    END;

    CREATE TRIGGER IF NOT EXISTS lean_theorems_au AFTER UPDATE ON lean_theorems BEGIN
        INSERT INTO lean_theorems_fts(lean_theorems_fts, rowid, name, statement, statement_pure, lean_name)
        VALUES ('delete', old.rowid, old.name, old.statement, old.statement_pure, old.lean_name);
        INSERT INTO lean_theorems_fts(rowid, name, statement, statement_pure, lean_name)
        VALUES (new.rowid, new.name, new.statement, new.statement_pure, new.lean_name);
    END;

    CREATE TABLE IF NOT EXISTS theorem_dependencies (
        theorem_id TEXT REFERENCES lean_theorems(id) ON DELETE CASCADE,
        depends_on_id TEXT REFERENCES lean_theorems(id) ON DELETE CASCADE,
        PRIMARY KEY (theorem_id, depends_on_id)
    );

    -- Python symbol index
    CREATE TABLE IF NOT EXISTS symbols (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,        -- 'function' | 'class'
        module TEXT NOT NULL,
        signature TEXT NOT NULL,
        status TEXT NOT NULL,      -- 'canonical' | 'public' | 'scratch' | 'archived' | 'retired'
        is_lru_cached INTEGER DEFAULT 0,
        frame_hint TEXT,
        redirect_to TEXT,
        docstring_summary TEXT,
        lean_citations TEXT,       -- JSON array
        kb_refs TEXT,              -- JSON array
        also_in_modules TEXT,      -- JSON array
        file TEXT NOT NULL,
        line INTEGER NOT NULL,
        project TEXT,
        created_at TEXT,
        updated_at TEXT,
        embedding BLOB,
        parent_impl TEXT,          -- Rust: impl block label, e.g. 'Foo' or 'Greet for Foo'
        visibility TEXT,           -- Rust/TS: 'pub' | 'pub(crate)' | 'export' | '' etc.
        is_signature_only INTEGER DEFAULT 0,  -- 1 for signature-only pass chunks
        node_type TEXT,            -- tree-sitter node type, e.g. 'function_item', 'class_declaration'
        language TEXT              -- 'python' | 'typescript' | 'rust' | 'go' | ... (derived at ingest from file ext)
    );

    CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
    CREATE INDEX IF NOT EXISTS idx_symbols_status ON symbols(status);
    CREATE INDEX IF NOT EXISTS idx_symbols_module ON symbols(module);
    CREATE INDEX IF NOT EXISTS idx_symbols_project ON symbols(project);

    -- TeX annotation index
    CREATE TABLE IF NOT EXISTS tex_annotations (
        id TEXT PRIMARY KEY,
        section_label TEXT,
        section_title TEXT,
        python_refs TEXT,           -- JSON array of "cl44/module.py::function"
        lean_refs TEXT,             -- JSON array of "File.lean" or "File.lean::Name"
        epic_refs TEXT,             -- JSON array of "project-XXXX" beads IDs
        kb_refs TEXT,               -- JSON array of "kb-YYYYMMDD-..." IDs
        context TEXT,               -- 2-3 lines of TeX text following annotation block
        file TEXT NOT NULL,
        line INTEGER NOT NULL,
        created_at TEXT,
        updated_at TEXT,
        embedding BLOB
    );

    CREATE INDEX IF NOT EXISTS idx_tex_annotations_section ON tex_annotations(section_label);
    CREATE INDEX IF NOT EXISTS idx_tex_annotations_file ON tex_annotations(file);

    -- Structural algebraic facts (commutators, eigenvalues, spectra, identities)
    -- Single source = cl44.certified_data; this table holds pointers + lookup keys.
    -- NOT semantically embedded: queries are exact-match on operator names.
    CREATE TABLE IF NOT EXISTS structural_facts (
        id TEXT PRIMARY KEY,
        relation_type TEXT NOT NULL
            CHECK(relation_type IN ('commutator','anticommutator','eigenvalue',
                                    'trace','charpoly','identity','negative')),
        lhs_operator TEXT NOT NULL,   -- e.g. 'C', 'K', 'M_odd', 'Q_EM'
        rhs_operator TEXT,            -- NULL for unary relations (trace, eigenvalue, charpoly)
        result_exact TEXT NOT NULL,   -- human-readable exact result, e.g. '0', '3/4 * I', 'MIXED'
        negative INTEGER DEFAULT 0,   -- 1 if result asserts the relation does NOT hold
        certified_data_key TEXT,      -- import path, e.g. 'cl44.certified_data.ALGEBRA_RELATIONS["C_Modd"]'
        lean_thm TEXT,                -- qualified Lean theorem name or NULL
        project TEXT,  -- caller-supplied; no hardcoded default (kb-4mi: was 'algebraic-genesis', silently mis-tagged other users' rows)
        notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_structural_facts_lhs ON structural_facts(lhs_operator);
    CREATE INDEX IF NOT EXISTS idx_structural_facts_rhs ON structural_facts(rhs_operator);
    CREATE INDEX IF NOT EXISTS idx_structural_facts_type ON structural_facts(relation_type);
    CREATE INDEX IF NOT EXISTS idx_structural_facts_lhs_rhs ON structural_facts(lhs_operator, rhs_operator);

    -- Issue tracker (kb-native replacement for beads/bd)
    CREATE TABLE IF NOT EXISTS issues (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL CHECK(type IN ('task','bug','feature','epic','chore','spike','decision')),
        status TEXT NOT NULL DEFAULT 'open'
            CHECK(status IN ('open','in_progress','blocked','closed','deferred')),
        priority INTEGER DEFAULT 2,
        parent_id TEXT REFERENCES issues(id),
        title TEXT NOT NULL,
        description TEXT,
        design_file TEXT,
        assignee TEXT,
        close_reason TEXT,
        project TEXT,
        tags TEXT,  -- JSON array
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        started_at TEXT,
        closed_at TEXT,
        closed_by_session TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_issues_project ON issues(project);
    CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status);
    CREATE INDEX IF NOT EXISTS idx_issues_parent_id ON issues(parent_id);
    CREATE INDEX IF NOT EXISTS idx_issues_type ON issues(type);
    CREATE INDEX IF NOT EXISTS idx_issues_project_status ON issues(project, status);

    CREATE VIRTUAL TABLE IF NOT EXISTS issues_fts USING fts5(
        title, description,
        content='issues',
        content_rowid='rowid'
    );

    CREATE TRIGGER IF NOT EXISTS issues_ai AFTER INSERT ON issues BEGIN
        INSERT INTO issues_fts(rowid, title, description)
        VALUES (new.rowid, new.title, new.description);
    END;

    CREATE TRIGGER IF NOT EXISTS issues_ad AFTER DELETE ON issues BEGIN
        INSERT INTO issues_fts(issues_fts, rowid, title, description)
        VALUES ('delete', old.rowid, old.title, old.description);
    END;

    CREATE TRIGGER IF NOT EXISTS issues_au AFTER UPDATE ON issues BEGIN
        INSERT INTO issues_fts(issues_fts, rowid, title, description)
        VALUES ('delete', old.rowid, old.title, old.description);
        INSERT INTO issues_fts(rowid, title, description)
        VALUES (new.rowid, new.title, new.description);
    END;

    CREATE TABLE IF NOT EXISTS issue_deps (
        issue_id TEXT NOT NULL REFERENCES issues(id) ON DELETE CASCADE,
        depends_on_id TEXT NOT NULL REFERENCES issues(id) ON DELETE CASCADE,
        type TEXT NOT NULL CHECK(type IN ('blocks','parent-child','discovered-from','related','supersedes')),
        created_at TEXT NOT NULL,
        created_by TEXT,
        PRIMARY KEY (issue_id, depends_on_id, type)
    );

    CREATE INDEX IF NOT EXISTS idx_issue_deps_issue ON issue_deps(issue_id);
    CREATE INDEX IF NOT EXISTS idx_issue_deps_depends_on ON issue_deps(depends_on_id);

    CREATE TABLE IF NOT EXISTS issue_comments (
        id TEXT PRIMARY KEY,
        issue_id TEXT NOT NULL REFERENCES issues(id) ON DELETE CASCADE,
        body TEXT NOT NULL,
        author TEXT,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_issue_comments_issue ON issue_comments(issue_id);

    CREATE TABLE IF NOT EXISTS child_counters (
        parent_id TEXT PRIMARY KEY,
        counter INTEGER NOT NULL
    );

    -- Bridge message memory (agent-to-agent communication archive)
    -- All messages stored for thread reconstruction; substantive subset embedded.
    CREATE TABLE IF NOT EXISTS bridge_messages (
        id INTEGER PRIMARY KEY,       -- jsonl msg id (natural key)
        ts TEXT NOT NULL,             -- ISO timestamp from jsonl
        sender TEXT NOT NULL,
        recipients TEXT,              -- JSON array (from 'to' field if present)
        subject TEXT,
        body TEXT,
        event_type TEXT,              -- announce/message/reply/ack/recv-noop/watch-noop/etc
        reply_to INTEGER,             -- thread link (FK to bridge_messages.id, nullable)
        is_substantive INTEGER NOT NULL DEFAULT 0,  -- 1 if worth embedding/surfacing
        embedded INTEGER NOT NULL DEFAULT 0,        -- 1 once written to bridge_messages_vec
        created_at TEXT NOT NULL      -- ingest timestamp
    );

    CREATE INDEX IF NOT EXISTS idx_bridge_ts ON bridge_messages(ts DESC);
    CREATE INDEX IF NOT EXISTS idx_bridge_sender ON bridge_messages(sender);
    CREATE INDEX IF NOT EXISTS idx_bridge_reply_to ON bridge_messages(reply_to);
    CREATE INDEX IF NOT EXISTS idx_bridge_substantive ON bridge_messages(is_substantive);

    -- Bridge LIVENESS overlay (kb-jij Phase-6 additive: agents.json stays canonical for
    -- identity; this table holds the SERVER-OWNED liveness the jsonl/agents.json never had).
    -- last_seen = max(SSE heartbeat, message poll, write); online = a live /bridge/watch SSE
    -- connection is currently held. /bridge/agents classifies online/idle/stale/offline from
    -- these vs. TTLs (online: SSE held + heartbeat<=20s; idle<=2m; stale<=10m; offline>10m).
    CREATE TABLE IF NOT EXISTS bridge_agents (
        id TEXT PRIMARY KEY,          -- bridge handle (e.g. kb-dev)
        session_id TEXT,              -- harness session id (for kb resume <agent>, kb-4c3ff4)
        last_seen REAL,               -- epoch seconds; refreshed by SSE heartbeat / poll / write
        online INTEGER NOT NULL DEFAULT 0,  -- 1 while a live /bridge/watch SSE connection is held
        updated_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_bridge_agents_last_seen ON bridge_agents(last_seen DESC);

    CREATE VIRTUAL TABLE IF NOT EXISTS bridge_messages_fts USING fts5(
        subject, body,
        content='bridge_messages',
        content_rowid='id'
    );

    CREATE TRIGGER IF NOT EXISTS bridge_messages_ai AFTER INSERT ON bridge_messages BEGIN
        INSERT INTO bridge_messages_fts(rowid, subject, body)
        VALUES (new.id, new.subject, new.body);
    END;

    CREATE TRIGGER IF NOT EXISTS bridge_messages_ad AFTER DELETE ON bridge_messages BEGIN
        INSERT INTO bridge_messages_fts(bridge_messages_fts, rowid, subject, body)
        VALUES ('delete', old.id, old.subject, old.body);
    END;

    CREATE TRIGGER IF NOT EXISTS bridge_messages_au AFTER UPDATE ON bridge_messages BEGIN
        INSERT INTO bridge_messages_fts(bridge_messages_fts, rowid, subject, body)
        VALUES ('delete', old.id, old.subject, old.body);
        INSERT INTO bridge_messages_fts(rowid, subject, body)
        VALUES (new.id, new.subject, new.body);
    END;

    -- Persona index: structural metadata for staleness checks (content lives in findings)
    -- Markdown is the source of truth; this is a derived index. Re-ingest on file change.
    CREATE TABLE IF NOT EXISTS persona_index (
        id TEXT PRIMARY KEY,         -- persona-<name>-<project>
        name TEXT NOT NULL,          -- frontmatter name: == bridge id
        project TEXT NOT NULL,       -- git-root basename of the project owning the file
        role TEXT,                   -- frontmatter role:
        file_path TEXT NOT NULL,     -- absolute path to the .md file
        file_mtime REAL NOT NULL,    -- mtime of the .md at last ingest (epoch float)
        content_hash TEXT NOT NULL,  -- SHA256 of file content (change detection)
        reviewers_yaml_path TEXT,    -- absolute path to reviewers.yaml if referenced
        reviewers_yaml_mtime REAL,   -- mtime of reviewers.yaml at last ingest
        top_level_dir_count INTEGER, -- count of top-level dirs in the project root at ingest
        finding_id TEXT,             -- FK to findings.id (the searchable entry)
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_persona_index_project ON persona_index(project);
    CREATE INDEX IF NOT EXISTS idx_persona_index_name ON persona_index(name);
    CREATE INDEX IF NOT EXISTS idx_persona_index_finding ON persona_index(finding_id);

    -- Hierarchical document sections (chunks of ingested documents)
    CREATE TABLE IF NOT EXISTS document_sections (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        parent_section_id TEXT REFERENCES document_sections(id),
        level INT NOT NULL,
        ordinal INT NOT NULL,
        heading TEXT,
        path TEXT NOT NULL,
        content TEXT,
        kind TEXT NOT NULL CHECK(kind IN ('prose', 'table', 'figure')),
        table_repr TEXT,         -- HTML table repr for kind=table; NULL otherwise
        embed_text TEXT,         -- linearized shadow used for embedding
        summary TEXT,
        token_count INT,
        content_hash TEXT,
        asset_path TEXT,         -- filesystem path to a stored figure image (NULL for non-figure leaves)
        status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'superseded')),
        superseded_by TEXT REFERENCES document_sections(id),
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_document_sections_document ON document_sections(document_id);
    CREATE INDEX IF NOT EXISTS idx_document_sections_parent ON document_sections(parent_section_id);
    CREATE INDEX IF NOT EXISTS idx_document_sections_path ON document_sections(document_id, path);
    CREATE INDEX IF NOT EXISTS idx_document_sections_status ON document_sections(status);
    CREATE INDEX IF NOT EXISTS idx_document_sections_ordinal ON document_sections(document_id, ordinal);

    -- Embedding model metadata (single row; tracks configured model for reembed detection)
    CREATE TABLE IF NOT EXISTS embedding_meta (
        id INTEGER PRIMARY KEY CHECK(id=1),
        format TEXT,
        url TEXT,
        model TEXT,
        dim INTEGER,
        signature TEXT,
        updated_at TEXT
    );

    -- Proof work queue: tracks unblocked Lean tasks so tip cannot stop silently.
    -- Rows inserted by: bd_close_reingest (cleared-contract), ingest_lean_work_queue.py
    -- (bulk bootstrap), compose_time_check (routing-deposit), and manually.
    -- divergence_flag=1: row's spec contradicts current certified_data/registry state —
    -- auto-forced to DESIGN-NEEDED, must be reported to archie before touching.
    CREATE TABLE IF NOT EXISTS lean_work_queue (
        id TEXT PRIMARY KEY,
        file TEXT NOT NULL,
        decl_name TEXT,
        class TEXT NOT NULL CHECK(class IN (
            'cleared-contract','docstring-pass','discharge-pad',
            'statement-suspect','routing-deposit','agent-returns-verify','review-class'
        )),
        readiness TEXT NOT NULL DEFAULT 'EXECUTE-READY'
            CHECK(readiness IN ('EXECUTE-READY','DESIGN-NEEDED')),
        bd_id TEXT,
        defer_reason TEXT,
        defer_detail TEXT,
        provenance_grade TEXT,
        agent_id TEXT,
        bead_date TEXT,
        divergence_flag INTEGER NOT NULL DEFAULT 0,
        project TEXT,  -- caller-supplied; no hardcoded default (kb-4mi: was 'algebraic-genesis', silently mis-tagged other users' rows)
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_lwq_file ON lean_work_queue(file);
    CREATE INDEX IF NOT EXISTS idx_lwq_class ON lean_work_queue(class);
    CREATE INDEX IF NOT EXISTS idx_lwq_defer ON lean_work_queue(defer_reason);
    CREATE INDEX IF NOT EXISTS idx_lwq_readiness ON lean_work_queue(readiness);
    CREATE INDEX IF NOT EXISTS idx_lwq_divergence ON lean_work_queue(divergence_flag);
"""


def init_schema(conn: sqlite3.Connection, embedding_dim: int) -> None:
    """Initialize database schema."""
    # Migration (read-index epic kb-3d347c / A1): rename python_symbols -> symbols.
    # MUST run BEFORE executescript — otherwise CREATE TABLE IF NOT EXISTS symbols would
    # create an empty table first and strand the legacy data. Main table renames in place
    # (cheap, metadata-only); the vec0 copy happens after symbols_vec is created (below).
    _names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "python_symbols" in _names and "symbols" not in _names:
        conn.execute("ALTER TABLE python_symbols RENAME TO symbols")
        for _old in ("idx_python_symbols_name", "idx_python_symbols_status",
                     "idx_python_symbols_module", "idx_python_symbols_project"):
            conn.execute(f"DROP INDEX IF EXISTS {_old}")

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

    # Create vector table for theorems
    _ = conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS lean_theorems_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
    """)
    _ = conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS issues_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
    """)

    # Create vector table for document sections
    _ = conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS document_sections_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
    """)

    # Schema migration: add source_path and source_hash to documents
    try:
        _ = conn.execute("SELECT source_path FROM documents LIMIT 1")
    except sqlite3.OperationalError:
        _ = conn.execute("ALTER TABLE documents ADD COLUMN source_path TEXT")
    try:
        _ = conn.execute("SELECT source_hash FROM documents LIMIT 1")
    except sqlite3.OperationalError:
        _ = conn.execute("ALTER TABLE documents ADD COLUMN source_hash TEXT")

    # Schema migration: add asset_path column to document_sections
    try:
        _ = conn.execute("SELECT asset_path FROM document_sections LIMIT 1")
    except sqlite3.OperationalError:
        _ = conn.execute("ALTER TABLE document_sections ADD COLUMN asset_path TEXT")

    # Schema migration: add summary column if not exists
    try:
        _ = conn.execute("SELECT summary FROM findings LIMIT 1")
    except sqlite3.OperationalError:
        _ = conn.execute("ALTER TABLE findings ADD COLUMN summary TEXT")

    # Create vector table for symbols
    _ = conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS symbols_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
    """)

    # Migration (read-index epic kb-3d347c / A1): copy the legacy python_symbols_vec rows
    # into the freshly-created symbols_vec (vec0 cannot be renamed), then drop the old one.
    # Copy — NOT re-embed (keys on id, same dim). Idempotent: only while the legacy vec exists.
    _allnames = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    if "python_symbols_vec" in _allnames:
        conn.execute("INSERT OR IGNORE INTO symbols_vec(id, embedding) "
                     "SELECT id, embedding FROM python_symbols_vec")
        conn.execute("DROP TABLE python_symbols_vec")

    # Create vector table for tex annotations
    _ = conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS tex_annotations_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
    """)

    # Create vector table for bridge messages (substantive subset only)
    _ = conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS bridge_messages_vec USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
    """)

    # Schema migration: add finding_id column to lean_theorems if not exists
    try:
        _ = conn.execute("SELECT finding_id FROM lean_theorems LIMIT 1")
    except sqlite3.OperationalError:
        _ = conn.execute("ALTER TABLE lean_theorems ADD COLUMN finding_id TEXT")

    # Schema migration: add project column to symbols if not exists
    try:
        _ = conn.execute("SELECT project FROM symbols LIMIT 1")
    except sqlite3.OperationalError:
        _ = conn.execute("ALTER TABLE symbols ADD COLUMN project TEXT")

    # Schema migration: add chunker metadata columns to symbols (kb-asf.4.1)
    _ps_cols = {row[1] for row in conn.execute("PRAGMA table_info(symbols)").fetchall()}
    if "parent_impl" not in _ps_cols:
        conn.execute("ALTER TABLE symbols ADD COLUMN parent_impl TEXT")
    if "visibility" not in _ps_cols:
        conn.execute("ALTER TABLE symbols ADD COLUMN visibility TEXT")
    if "is_signature_only" not in _ps_cols:
        conn.execute("ALTER TABLE symbols ADD COLUMN is_signature_only INTEGER DEFAULT 0")
    if "node_type" not in _ps_cols:
        conn.execute("ALTER TABLE symbols ADD COLUMN node_type TEXT")
    if "language" not in _ps_cols:
        conn.execute("ALTER TABLE symbols ADD COLUMN language TEXT")

    # NOTE: structural_facts, lean_work_queue, persona_index, and bridge_messages are
    # all created unconditionally by SCHEMA_SQL (CREATE TABLE IF NOT EXISTS) above, so
    # the old try/except-CREATE "migration" blocks for them were dead (the SELECT always
    # succeeded; the except never fired) and were removed (R6 / kb-b3e837).

    conn.commit()
