"""
kb/clone_detect/shingle.py — Tier C: sub-function shingle + MinHash CONTAINMENT

Design goal: detect that a code fragment already exists even when function
boundaries changed (extract = helper.shingles ⊆ original; inline = fragment
inside bigger fn), regardless of git or naming.

Containment (asymmetric), NOT Jaccard
--------------------------------------
containment(A, B) = |A ∩ B| / |A|

This is deliberately asymmetric:
  - helper extracted from original: containment(helper_shingles, original_shingles) ≈ 1
  - original vs helper:             containment(original_shingles, helper_shingles) ≈ small
  - Jaccard would be intermediate for both — useless for this task.

MinHash natively estimates Jaccard.  We correct to containment via:
  minhash_containment_estimate(A_sig, B_sig, |A|, |B|) = jaccard_estimate * (|A| + |B| - |A|*jaccard_estimate) / |A|
  (standard identity: |A∩B| = J*|A∪B| = J*(|A|+|B|-|A∩B|)  →  |A∩B|/|A| = J*(|A|+|B|)/(|A|*(1+J)) approx)
  Simpler exact form used here: containment = J_hat * |A∪B_hat| / |A|
  where |A∪B_hat| = (|A| + |B|) / (1 + J_hat)   [from |A∩B| = J|A∪B| and |A∪B|=|A|+|B|-|A∩B|]

Shingle parameters
------------------
k = 3  (trigrams over normalized statement tokens)

Chosen as a balance: k=2 produces too many boilerplate overlaps (assignment,
return patterns); k=4 is too sparse for short helpers (< 4 statements).  k=3
hits both extract (helpers are typically 3-8 statements) and inline detection.

In addition to k-gram shingles we emit single-statement hashes (k=1 equivalent,
order-insensitive) so short fragments (1-2 statements) still register.

num_perm = 128  (standard; gives ~2% std-dev on Jaccard, ≈ same on containment).

IDF / document-frequency boilerplate suppression
-------------------------------------------------
IdfIndex tracks how many indexed functions contain each shingle.  Shingles
with df > threshold (default: appearing in > 30% of corpus) are excluded from
containment queries.  This suppresses "return None", "pass", "raise ValueError"
patterns that would otherwise inflate containment for unrelated functions.
"""

from __future__ import annotations

import ast
import hashlib
import struct
from collections import Counter
from typing import Sequence


# ---------------------------------------------------------------------------
# AST normalization
# ---------------------------------------------------------------------------

def _collect_locals(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Collect all locally-defined names in a function: parameters + assigned names."""
    local_names: set[str] = set()
    # Parameters
    for arg in func_node.args.args + func_node.args.posonlyargs + func_node.args.kwonlyargs:
        local_names.add(arg.arg)
    if func_node.args.vararg:
        local_names.add(func_node.args.vararg.arg)
    if func_node.args.kwarg:
        local_names.add(func_node.args.kwarg.arg)
    # Assigned names in body
    for node in ast.walk(func_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            local_names.add(node.id)
    return local_names


class _Normalizer(ast.NodeTransformer):
    """
    Normalize a function for shingle comparison:
    - All locally-scoped names (parameters + assigned vars) are replaced by
      a single canonical placeholder ``_v`` (not numbered, so the same pattern
      of calls/literals matches regardless of which local holds which value).
    - External names (builtins, module refs, global calls like ``sum``, ``len``,
      ``logger.info``) are kept as-is — they carry semantic signal.
    - Docstrings are stripped.
    - Comments are stripped by the AST parser.

    Rationale for flat ``_v`` vs numbered ``_v0``, ``_v1``, ...:
    Numbered renaming assigns counters in first-appearance order, which differs
    between the extracted helper (where ``result`` is parameter→``_v0``) and the
    original (where ``result`` is a local→``_v1``).  Using a flat ``_v`` makes
    ``sum(_v)`` match ``sum(_v)`` regardless of which counter the local got in
    each function.  The cost: we lose intra-statement variable distinctness, but
    for statement-level shingles the call structure (``sum``, ``len``, ``append``)
    already differentiates statements; the local-variable values are noise.
    """

    def __init__(self, local_names: set[str] | None = None) -> None:
        self._locals: set[str] = local_names or set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Collect all locals defined in THIS function (not nested ones)
        self._locals = _collect_locals(node)
        # Rename parameter annotations in signature
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            arg.arg = "_v"
        if node.args.vararg:
            node.args.vararg.arg = "_v"
        if node.args.kwarg:
            node.args.kwarg.arg = "_v"
        # Strip docstring
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self._locals:
            node.id = "_v"
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        # Already handled in visit_FunctionDef for parameter names
        if node.arg in self._locals:
            node.arg = "_v"
        return node


def _stmt_tokens(stmt: ast.stmt) -> str:
    """Convert one statement to a stable token string (comments stripped by AST parse)."""
    try:
        return ast.unparse(stmt)
    except Exception:
        return repr(stmt)


def _normalize_function(source: str) -> list[str]:
    """
    Parse *source* as a single function definition and return a list of
    normalized statement strings (one per top-level statement in the body).

    If source is not parseable as a function, try parsing as a module and
    take all top-level statements.

    Locals are identified before visiting so that Load references to local
    names are also replaced with the canonical placeholder ``_v``.
    """
    try:
        tree = ast.parse(source.strip(), mode="exec")
    except SyntaxError:
        return []

    # Find function node; collect its locals before transformation
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_node = node
            break

    if func_node is not None:
        local_names = _collect_locals(func_node)
    else:
        # Fragment mode: collect all assigned names as "locals"
        local_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                local_names.add(node.id)

    normalizer = _Normalizer(local_names=local_names)
    tree = normalizer.visit(tree)
    ast.fix_missing_locations(tree)

    stmts: list[ast.stmt] = []
    if func_node is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stmts = node.body
                break
    else:
        stmts = tree.body  # type: ignore[assignment]

    return [_stmt_tokens(s) for s in stmts if not isinstance(s, ast.Pass)]


# ---------------------------------------------------------------------------
# Shingles
# ---------------------------------------------------------------------------

SHINGLE_K = 3  # k-gram size over normalized statement stream


def _hash_str(s: str) -> str:
    """Short hex hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def shingles(source: str, k: int = SHINGLE_K) -> set[str]:
    """
    Return the set of shingles for *source* (one function's source code).

    Emits two kinds of shingle:
    1. k-gram shingles over the normalized statement stream (order-sensitive):
       ``"s0|s1|s2"`` where s_i is the sha256[:16] of the normalized statement.
    2. Single-statement hashes (order-insensitive), prefixed ``"1g:"`` so they
       don't collide with k-gram entries.

    k defaults to SHINGLE_K=3 (trigrams).  Single-statement hashes are always
    included regardless of k, so short fragments (1-2 stmts) still produce
    non-empty shingle sets.
    """
    stmts = _normalize_function(source)
    if not stmts:
        return set()

    hashes = [_hash_str(s) for s in stmts]

    result: set[str] = set()

    # Single-statement hashes (order-insensitive component)
    for h in hashes:
        result.add(f"1g:{h}")

    # k-gram shingles
    if len(hashes) >= k:
        for i in range(len(hashes) - k + 1):
            gram = "|".join(hashes[i : i + k])
            result.add(gram)

    return result


# ---------------------------------------------------------------------------
# MinHash signature
# ---------------------------------------------------------------------------

# 64-bit multipliers for universal hashing (Tabulation / Knuth multiplicative)
_LARGE_PRIME = (1 << 61) - 1  # Mersenne prime
_MOD = (1 << 32)


def _make_hash_params(num_perm: int, seed: int = 0xDEADBEEF) -> list[tuple[int, int]]:
    """Generate (a, b) pairs for num_perm independent hash functions h(x) = (ax+b) mod p."""
    rng_state = seed
    params: list[tuple[int, int]] = []
    for _ in range(num_perm):
        rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        a = (rng_state >> 17) | 1  # ensure odd
        rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        b = rng_state >> 32
        params.append((a, b))
    return params


_DEFAULT_PARAMS: list[tuple[int, int]] | None = None


def _get_params(num_perm: int) -> list[tuple[int, int]]:
    global _DEFAULT_PARAMS
    if _DEFAULT_PARAMS is None or len(_DEFAULT_PARAMS) < num_perm:
        _DEFAULT_PARAMS = _make_hash_params(max(num_perm, 128))
    return _DEFAULT_PARAMS[:num_perm]


def _hash_shingle(shingle: str, a: int, b: int) -> int:
    """Hash a shingle string to a 32-bit integer using (a*sha_int + b) mod 2^32."""
    raw = int(hashlib.sha256(shingle.encode()).hexdigest()[:8], 16)  # 32-bit from sha256
    return ((a * raw + b) & 0xFFFFFFFF)


def minhash_signature(shingles_set: set[str], num_perm: int = 128) -> list[int]:
    """
    Compute a MinHash signature of *shingles_set* using *num_perm* permutations.

    Returns a list of *num_perm* minimum hash values (one per hash function).
    An empty shingle set returns all-max values (2^32 - 1).
    """
    if not shingles_set:
        return [0xFFFFFFFF] * num_perm

    params = _get_params(num_perm)
    sig = [0xFFFFFFFF] * num_perm

    for shingle in shingles_set:
        raw = int(hashlib.sha256(shingle.encode()).hexdigest()[:8], 16)
        for i, (a, b) in enumerate(params):
            h = (a * raw + b) & 0xFFFFFFFF
            if h < sig[i]:
                sig[i] = h

    return sig


# ---------------------------------------------------------------------------
# Containment (exact + MinHash estimate)
# ---------------------------------------------------------------------------

def containment(query_shingles: set[str], target_shingles: set[str]) -> float:
    """
    Exact asymmetric containment: |query ∩ target| / |query|.

    Interpretation:
      - high value (~1) means query is mostly a SUBSET of target
        (query was extracted from target, or query is inlined inside target).
      - containment(big, small) will be LOW even when containment(small, big) is HIGH.
      - Jaccard would be intermediate for both — incorrect for extract/inline detection.

    Returns 0.0 if query_shingles is empty.
    """
    if not query_shingles:
        return 0.0
    return len(query_shingles & target_shingles) / len(query_shingles)


def minhash_containment_estimate(
    query_sig: list[int],
    target_sig: list[int],
    query_size: int,
    target_size: int,
) -> float:
    """
    Estimate containment(query, target) = |A∩B|/|A| from MinHash signatures.

    Derivation:
      MinHash natively estimates Jaccard J = |A∩B| / |A∪B|.
      Jaccard estimate from sigs: J_hat = (# positions where sig_A[i] == sig_B[i]) / num_perm.
      |A∪B| = |A| + |B| - |A∩B|  →  |A∩B| = J * |A∪B| = J * (|A| + |B|) / (1 + J)
      containment = |A∩B| / |A| = J_hat * (|A| + |B|) / (|A| * (1 + J_hat))

    This is strictly NOT Jaccard — it's asymmetric and query-size-normalized.

    Falls back to 0.0 if query_size == 0 or sigs are empty.
    """
    if query_size == 0 or not query_sig:
        return 0.0

    num_perm = len(query_sig)
    matches = sum(1 for a, b in zip(query_sig, target_sig) if a == b)
    j_hat = matches / num_perm

    if j_hat == 0.0:
        return 0.0
    if j_hat >= 1.0:
        return 1.0

    # |A∩B| estimate
    union_size = query_size + target_size  # approximate |A∪B| before dedup correction
    # More accurate: |A∪B| = (|A| + |B|) / (1 + J_hat)
    union_size_hat = (query_size + target_size) / (1 + j_hat)
    intersection_hat = j_hat * union_size_hat
    return min(1.0, intersection_hat / query_size)


# ---------------------------------------------------------------------------
# IDF / document-frequency boilerplate suppression
# ---------------------------------------------------------------------------

class IdfIndex:
    """
    Tracks per-shingle document frequency across indexed functions and
    provides a filtered shingle set that suppresses boilerplate.

    Usage:
        idx = IdfIndex()
        idx.add("fn_id_1", shingles(source1))
        idx.add("fn_id_2", shingles(source2))
        # Query: remove high-df shingles before containment check
        filtered = idx.filter(query_shingles, df_threshold=0.3)
        containment(filtered, target_shingles_filtered)

    df_threshold: shingles appearing in more than this fraction of all
    indexed functions are excluded (default 0.3 = 30%).  Set to 1.0 to
    disable suppression.
    """

    def __init__(self, df_threshold: float = 0.3) -> None:
        self.df_threshold = df_threshold
        self._df: Counter[str] = Counter()      # shingle -> document count
        self._n_docs: int = 0                   # total indexed functions
        self._doc_shingles: dict[str, set[str]] = {}  # fn_id -> shingle set (for re-index)

    def add(self, fn_id: str, fn_shingles: set[str]) -> None:
        """Index a function's shingles.  Idempotent if fn_id already present."""
        if fn_id in self._doc_shingles:
            return
        self._doc_shingles[fn_id] = fn_shingles
        self._n_docs += 1
        for s in fn_shingles:
            self._df[s] += 1

    def df(self, shingle: str) -> float:
        """Fraction of indexed functions containing *shingle* (0 if unseen)."""
        if self._n_docs == 0:
            return 0.0
        return self._df[shingle] / self._n_docs

    def is_boilerplate(self, shingle: str) -> bool:
        return self.df(shingle) > self.df_threshold

    def filter(self, fn_shingles: set[str], df_threshold: float | None = None) -> set[str]:
        """Return *fn_shingles* with boilerplate shingles removed."""
        threshold = df_threshold if df_threshold is not None else self.df_threshold
        if self._n_docs == 0:
            return fn_shingles
        cutoff = threshold * self._n_docs
        return {s for s in fn_shingles if self._df.get(s, 0) <= cutoff}

    @property
    def corpus_size(self) -> int:
        return self._n_docs
