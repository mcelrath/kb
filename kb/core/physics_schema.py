"""Physics/Lean/TeX schema extension for kb.

Manages the physics-specific tables that belong to kb-ag, not the generic kb plugin.
Called by:
  - kb ensure-physics-schema (CLI, invoked by kb-ag SessionStart hook)
  - kb-ag hooks that need these tables to exist before querying them

DO NOT call init_physics_schema() from init_schema() — generic kb databases
must not create these tables. Only physics-aware consumers should call this.
"""

import sqlite3


PHYSICS_SCHEMA_SQL = """
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
        lhs_operator TEXT NOT NULL,
        rhs_operator TEXT,
        result_exact TEXT NOT NULL,
        negative INTEGER DEFAULT 0,
        certified_data_key TEXT,
        lean_thm TEXT,
        project TEXT,
        notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_structural_facts_lhs ON structural_facts(lhs_operator);
    CREATE INDEX IF NOT EXISTS idx_structural_facts_rhs ON structural_facts(rhs_operator);
    CREATE INDEX IF NOT EXISTS idx_structural_facts_type ON structural_facts(relation_type);
    CREATE INDEX IF NOT EXISTS idx_structural_facts_lhs_rhs ON structural_facts(lhs_operator, rhs_operator);

    -- Proof work queue: tracks unblocked Lean tasks so tip cannot stop silently.
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
        project TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_lwq_file ON lean_work_queue(file);
    CREATE INDEX IF NOT EXISTS idx_lwq_class ON lean_work_queue(class);
    CREATE INDEX IF NOT EXISTS idx_lwq_defer ON lean_work_queue(defer_reason);
    CREATE INDEX IF NOT EXISTS idx_lwq_readiness ON lean_work_queue(readiness);
    CREATE INDEX IF NOT EXISTS idx_lwq_divergence ON lean_work_queue(divergence_flag);
"""


def init_physics_schema(conn: sqlite3.Connection, embedding_dim: int) -> None:
    """Create physics/Lean/TeX tables in an existing kb database.

    Idempotent (CREATE TABLE IF NOT EXISTS). Safe to call on every SessionStart
    from a kb-ag hook — it is a no-op if tables already exist.

    Does NOT call conn.commit() — caller is responsible.
    """
    conn.executescript(PHYSICS_SCHEMA_SQL)

    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS lean_theorems_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
    """)

    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS tex_annotations_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{embedding_dim}]
        )
    """)

    # Migration: add finding_id column to lean_theorems if not exists
    try:
        conn.execute("SELECT finding_id FROM lean_theorems LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE lean_theorems ADD COLUMN finding_id TEXT")

    conn.commit()
