"""
Test each kb module for annotation failures under eager evaluation (3.11-3.13).
We try to compile+exec each module in isolation, stubbing out problematic imports.
"""
import sys
import importlib
import importlib.util
import types
from pathlib import Path
from unittest.mock import MagicMock

# Stub out modules that aren't available
stubs = ['sqlite_vec', 'sqlite3']
for name in stubs:
    pass  # don't stub - just try to import naturally

# Add kb to path
sys.path.insert(0, '/home/mcelrath/Projects/ai/kb')

# Try importing each submodule
modules_to_try = [
    'kb.constants',
    'kb.validation',
    'kb.core.embedding',
    'kb.core.schema',
    'kb.core.connection',
    'kb.entities.base',
    'kb.entities.scripts',
    'kb.entities.concepts',
    'kb.entities.documents',
    'kb.entities.issues',
    'kb.entities.theorems',
    'kb.llm.client',
    'kb.llm.analysis',
    'kb.llm.extractive',
    'kb.llm.summary_sdk',
    'kb.search.hybrid',
    'kb.code_lineage.structural',
    'kb.code_lineage.lineage',
    'kb.code_ingest.chunker',
    'kb.clone_detect.shingle',
    'kb.hooks.check_symbols',
    'kb.hooks.lean_write_guard',
    'kb.hooks.lean_py_surface',
    'kb.hooks.py_lean_surface',
    'kb.hooks.bd_close_reingest',
    'kb.hooks.lake_error_surface',
    'kb.hooks.python_tex_backref',
    'kb.hooks.tex_section_context',
    'kb.hooks.tex_stale_surface',
    'kb.bd_import',
    'kb.issue_cli',
    'kb.config',
    'kb.configure',
    'kb.facade',
]

for mod_name in modules_to_try:
    try:
        importlib.import_module(mod_name)
        print(f"OK  {mod_name}")
    except ModuleNotFoundError as e:
        print(f"MISSING_DEP  {mod_name}: {e}")
    except (NameError, UnboundLocalError) as e:
        print(f"ANNOTATION_ERROR  {mod_name}: {e}")
    except Exception as e:
        print(f"OTHER_ERROR  {mod_name}: {type(e).__name__}: {e}")
