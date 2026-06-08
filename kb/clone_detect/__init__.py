# kb/clone_detect — embedding-free sub-function clone detection (Tier C)
# Uses AST-normalized statement shingles + MinHash containment (|A∩B|/|A|).
from .shingle import shingles, minhash_signature, containment, minhash_containment_estimate, IdfIndex

__all__ = [
    "shingles",
    "minhash_signature",
    "containment",
    "minhash_containment_estimate",
    "IdfIndex",
]
