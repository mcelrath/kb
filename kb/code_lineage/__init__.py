"""code_lineage — Tier-A structural-hash lineage for Python functions.

Embedding-free, git-metadata-independent identity via normalized AST.
"""
from .structural import structural_hash
from .lineage import FunctionVersion, cluster_lineages

__all__ = ["structural_hash", "FunctionVersion", "cluster_lineages"]
