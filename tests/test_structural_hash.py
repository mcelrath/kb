"""Unit tests for kb.code_lineage.structural_hash.

Invariants under test (spec from PLAN-semantic-prior-art-surfacing.md Tier A):

(a) function rename          -> SAME hash
(b) in-file move / reformat / comment change -> SAME hash
(c) local-var rename         -> SAME hash
(d) return-type change       -> DIFFERENT hash
(e) decorator add/remove     -> DIFFERENT hash
(f) real logic change        -> DIFFERENT hash
(g) delete-then-readd        -> same hash -> same lineage
"""

import sqlite3
import textwrap
import pytest

from kb.code_lineage.structural import structural_hash
from kb.code_lineage.lineage import (
    FunctionVersion,
    cluster_lineages,
    migrate,
    insert_version,
    load_lineages,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def h(src: str) -> str:
    """Dedent and hash."""
    return structural_hash(textwrap.dedent(src))


# ---------------------------------------------------------------------------
# (a) Function rename -> SAME hash
# ---------------------------------------------------------------------------

def test_a_function_rename_same_hash():
    foo = """
        def foo(x):
            return x + 1
    """
    bar = """
        def bar(x):
            return x + 1
    """
    assert h(foo) == h(bar), "Rename of function name must not change hash"


# ---------------------------------------------------------------------------
# (b) In-file move / reformat / comment change -> SAME hash
# ---------------------------------------------------------------------------

def test_b_whitespace_change_same_hash():
    compact = "def f(x):\n    return x + 1\n"
    spaced = "def f(x):\n\n    return x + 1\n\n"
    assert h(compact) == h(spaced), "Extra blank lines must not change hash"


def test_b_comment_stripped_same_hash():
    with_comment = """
        def f(x):
            # add one
            return x + 1
    """
    no_comment = """
        def f(x):
            return x + 1
    """
    assert h(with_comment) == h(no_comment), "Comment removal must not change hash"


def test_b_docstring_stripped_same_hash():
    with_doc = '''
        def f(x):
            """Add one."""
            return x + 1
    '''
    no_doc = """
        def f(x):
            return x + 1
    """
    assert h(with_doc) == h(no_doc), "Docstring removal must not change hash"


# ---------------------------------------------------------------------------
# (c) Local-var rename -> SAME hash
# ---------------------------------------------------------------------------

def test_c_param_rename_same_hash():
    fa = """
        def f(a):
            return a * 2
    """
    fz = """
        def f(z):
            return z * 2
    """
    assert h(fa) == h(fz), "Parameter rename must not change hash"


def test_c_local_var_rename_same_hash():
    v1 = """
        def compute(x):
            result = x * 2
            return result
    """
    v2 = """
        def compute(x):
            answer = x * 2
            return answer
    """
    assert h(v1) == h(v2), "Local variable rename must not change hash"


def test_c_multiple_param_rename_same_hash():
    a = """
        def f(a, b):
            return a + b
    """
    b = """
        def f(p, q):
            return p + q
    """
    assert h(a) == h(b), "Multi-param rename must not change hash"


# ---------------------------------------------------------------------------
# (d) Return-type change -> DIFFERENT hash
# ---------------------------------------------------------------------------

def test_d_return_type_change_different_hash():
    int_ann = """
        def f(x) -> int:
            return x
    """
    str_ann = """
        def f(x) -> str:
            return x
    """
    assert h(int_ann) != h(str_ann), "Different return annotation must produce different hash"


def test_d_no_annotation_vs_annotated_different_hash():
    no_ann = """
        def f(x):
            return x
    """
    ann = """
        def f(x) -> int:
            return x
    """
    assert h(no_ann) != h(ann), "Adding return annotation must change hash"


# ---------------------------------------------------------------------------
# (e) Decorator add/remove -> DIFFERENT hash
# ---------------------------------------------------------------------------

def test_e_decorator_add_different_hash():
    plain = """
        def f(x):
            return x
    """
    decorated = """
        from functools import lru_cache

        @lru_cache
        def f(x):
            return x
    """
    assert h(plain) != h(decorated), "Adding @lru_cache must change hash"


def test_e_different_decorators_different_hash():
    a = """
        @staticmethod
        def f(x):
            return x
    """
    b = """
        @classmethod
        def f(x):
            return x
    """
    assert h(a) != h(b), "Different decorators must produce different hashes"


# ---------------------------------------------------------------------------
# (f) Real logic change -> DIFFERENT hash
# ---------------------------------------------------------------------------

def test_f_logic_change_different_hash():
    add = """
        def f(x):
            return x + 1
    """
    sub = """
        def f(x):
            return x - 1
    """
    assert h(add) != h(sub), "Different logic must produce different hash"


def test_f_called_function_name_matters():
    uses_foo = """
        def f(x):
            return foo(x)
    """
    uses_bar = """
        def f(x):
            return bar(x)
    """
    assert h(uses_foo) != h(uses_bar), "Different called-function name must produce different hash"


def test_f_literal_value_matters():
    lit1 = """
        def f(x):
            return x + 42
    """
    lit2 = """
        def f(x):
            return x + 99
    """
    assert h(lit1) != h(lit2), "Different literal must produce different hash"


# ---------------------------------------------------------------------------
# (g) Delete-then-readd -> same hash -> same lineage
# ---------------------------------------------------------------------------

def test_g_delete_readd_same_lineage():
    source = """
        def compute(x):
            return x * 2
    """
    h1 = h(source)

    # Simulate delete: old commit has the version, new commit re-adds it
    versions = [
        FunctionVersion(
            structural_hash=h1,
            name="compute",
            file="old_module.py",
            provenance="abc123",
            line=10,
        ),
        # gap — version not present in some commits
        FunctionVersion(
            structural_hash=h1,
            name="compute",  # same name, same content
            file="new_module.py",  # might have moved files
            provenance="def456",
            line=5,
        ),
    ]

    lineages = cluster_lineages(versions)
    assert len(lineages) == 1, "Delete-then-readd must produce one lineage, not two"
    lin = lineages[h1]
    assert lin.commits_seen == ["abc123", "def456"]
    assert set(lin.paths_seen) == {"old_module.py", "new_module.py"}


def test_g_renamed_and_moved_same_lineage():
    """Rename + file move with same body -> same lineage."""
    source_original = """
        def old_name(a, b):
            return a + b + 1
    """
    source_renamed = """
        def new_name(x, y):
            return x + y + 1
    """
    h_orig = h(source_original)
    h_renamed = h(source_renamed)
    assert h_orig == h_renamed, "Rename + param-rename with same body should yield same hash"

    versions = [
        FunctionVersion(
            structural_hash=h_orig,
            name="old_name",
            file="utils/old.py",
            provenance="commit_v1",
        ),
        FunctionVersion(
            structural_hash=h_renamed,
            name="new_name",
            file="utils/new.py",
            provenance="commit_v2",
        ),
    ]
    lineages = cluster_lineages(versions)
    assert len(lineages) == 1, "Rename+move with same body must cluster into one lineage"


# ---------------------------------------------------------------------------
# DB round-trip test
# ---------------------------------------------------------------------------

def test_db_roundtrip():
    """Insert versions, load them back, cluster: identical to in-memory result."""
    source_a = """
        def helper(n):
            return n * 3
    """
    source_b = """
        def renamed_helper(m):
            return m * 3
    """
    ha = h(source_a)
    hb = h(source_b)
    assert ha == hb, "Pre-condition: same body, different name -> same hash"

    conn = sqlite3.connect(":memory:")
    migrate(conn)

    insert_version(conn, FunctionVersion(
        structural_hash=ha, name="helper", file="a.py", provenance="c1"
    ))
    insert_version(conn, FunctionVersion(
        structural_hash=hb, name="renamed_helper", file="b.py", provenance="c2"
    ))

    lineages = load_lineages(conn)
    assert len(lineages) == 1
    lin = lineages[ha]
    assert set(lin.names_seen) == {"helper", "renamed_helper"}
    assert set(lin.paths_seen) == {"a.py", "b.py"}
    assert set(lin.commits_seen) == {"c1", "c2"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_function_body():
    """A no-op function hashes without error."""
    src = "def noop(x): pass"
    result = h(src)
    assert isinstance(result, str) and len(result) == 64


def test_recursive_function():
    """Recursive self-call: own name replaced but call still normalized."""
    recursive = """
        def factorial(n):
            if n <= 1:
                return 1
            return factorial(n - 1) * n
    """
    renamed_recursive = """
        def fact(m):
            if m <= 1:
                return 1
            return fact(m - 1) * m
    """
    assert h(recursive) == h(renamed_recursive), "Recursive rename must yield same hash"


def test_async_function():
    """Async functions hash without error."""
    src = """
        async def fetch(url):
            return await get(url)
    """
    src2 = """
        async def download(link):
            return await get(link)
    """
    assert h(src) == h(src2), "Async function param rename must yield same hash"
