"""
Entity Repositories

Repository classes for each entity type: scripts, documents, theorems, concepts.
"""

from .base import EntityRepository
from .scripts import ScriptsRepository
from .documents import DocumentsRepository

__all__ = [
    "EntityRepository",
    "ScriptsRepository",
    "DocumentsRepository",
]
