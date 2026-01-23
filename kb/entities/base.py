"""
Base Entity Repository

Abstract base class for entity repositories.
"""

import sqlite3
from abc import ABC


class EntityRepository(ABC):
    """Base class for entity repositories."""

    conn: sqlite3.Connection

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
