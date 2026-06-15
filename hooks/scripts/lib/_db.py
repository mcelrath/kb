"""Shared kb database-path resolver for hooks.

Hooks must honor KB_DB (a user with a custom / per-project database). Reading the
hardcoded ~/.cache/kb/knowledge.db silently queries the WRONG db. This mirrors
open_issues_surface._db_path and kb/config.py's KB_DB precedence so every hook
agrees on the active database. (kb-05n)
"""
import os


def kb_db_path() -> str:
    return os.path.expanduser(os.environ.get("KB_DB", "~/.cache/kb/knowledge.db"))
