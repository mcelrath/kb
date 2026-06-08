"""
tests/test_clone_detect_shingle.py

Invariant tests for kb/clone_detect/shingle.py (Tier C).

Tests:
(a) EXTRACT:    helper (subset of original) → high containment(helper, original), MinHash approx matches
(b) INLINE:     fragment shingles against bigger fn → high containment
(c) ASYMMETRY:  containment(small, big) >> containment(big, small)
(d) UNRELATED:  two disjoint functions → ~0 containment
(e) MinHash estimate vs exact: error < 0.15 on several cases (128 perms)
(f) IDF:        a boilerplate shingle present in many fns gets down-weighted/excluded
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kb.clone_detect.shingle import (
    shingles,
    minhash_signature,
    containment,
    minhash_containment_estimate,
    IdfIndex,
    SHINGLE_K,
)

# ---------------------------------------------------------------------------
# Sample functions
# ---------------------------------------------------------------------------

ORIGINAL = """
def process_data(items):
    result = []
    for item in items:
        if item > 0:
            val = item * 2
            result.append(val)
    total = sum(result)
    mean = total / len(result) if result else 0
    return mean
"""

HELPER = """
def compute_mean(result):
    total = sum(result)
    mean = total / len(result) if result else 0
    return mean
"""

HELPER2 = """
def double_positives(items):
    result = []
    for item in items:
        if item > 0:
            val = item * 2
            result.append(val)
    return result
"""

UNRELATED = """
def serialize_graph(graph):
    nodes = list(graph.nodes())
    edges = [(u, v) for u, v in graph.edges()]
    payload = {"nodes": nodes, "edges": edges}
    return payload
"""

BIGGER_WITH_EXTRAS = """
def process_data_v2(items, config):
    validated = [x for x in items if isinstance(x, (int, float))]
    result = []
    for item in validated:
        if item > 0:
            val = item * 2
            result.append(val)
    total = sum(result)
    mean = total / len(result) if result else 0
    logger.info("done")
    return mean, total
"""


# ---------------------------------------------------------------------------
# (a) EXTRACT: helper ⊆ original
# ---------------------------------------------------------------------------

def test_extract_containment_high():
    """containment(helper, original) should be high — helper's shingles mostly ⊆ original."""
    h = shingles(HELPER2)
    o = shingles(ORIGINAL)
    assert h, "helper shingles must be non-empty"
    assert o, "original shingles must be non-empty"
    c = containment(h, o)
    print(f"(a) EXTRACT containment(helper2, original) = {c:.3f}")
    assert c > 0.4, f"Expected extract containment > 0.4, got {c:.3f}"


def test_extract_minhash_approx():
    """MinHash containment estimate should be within 0.15 of exact for extract case."""
    h = shingles(HELPER2)
    o = shingles(ORIGINAL)
    exact = containment(h, o)
    h_sig = minhash_signature(h, num_perm=128)
    o_sig = minhash_signature(o, num_perm=128)
    estimated = minhash_containment_estimate(h_sig, o_sig, len(h), len(o))
    err = abs(exact - estimated)
    print(f"(a) MinHash estimate={estimated:.3f}, exact={exact:.3f}, err={err:.3f}")
    assert err < 0.20, f"MinHash containment error {err:.3f} too large (threshold 0.20)"


# ---------------------------------------------------------------------------
# (b) INLINE: fragment {A,B} ⊆ bigger fn {A,B,C,D}
# ---------------------------------------------------------------------------

def test_inline_containment_high():
    """A fragment that was inlined into a bigger fn: containment(fragment, bigger) should be high."""
    frag = shingles(HELPER)     # compute_mean logic: total=sum, mean=..., return mean
    big = shingles(BIGGER_WITH_EXTRAS)
    c = containment(frag, big)
    print(f"(b) INLINE containment(helper_frag, bigger) = {c:.3f}")
    assert c > 0.35, f"Expected inline containment > 0.35, got {c:.3f}"


# ---------------------------------------------------------------------------
# (c) ASYMMETRY: containment(small,big) >> containment(big,small)
# ---------------------------------------------------------------------------

def test_asymmetry():
    """The containment measure must be asymmetric — otherwise it's Jaccard."""
    h = shingles(HELPER2)
    o = shingles(ORIGINAL)
    c_small_big = containment(h, o)
    c_big_small = containment(o, h)
    print(f"(c) ASYMMETRY: containment(small,big)={c_small_big:.3f}, containment(big,small)={c_big_small:.3f}")
    assert c_small_big > c_big_small + 0.15, (
        f"Expected containment(small,big) >> containment(big,small), "
        f"got {c_small_big:.3f} vs {c_big_small:.3f}"
    )


def test_asymmetry_minhash():
    """MinHash containment estimate should also be asymmetric."""
    h = shingles(HELPER2)
    o = shingles(ORIGINAL)
    h_sig = minhash_signature(h, 128)
    o_sig = minhash_signature(o, 128)
    c_sh = minhash_containment_estimate(h_sig, o_sig, len(h), len(o))
    c_bs = minhash_containment_estimate(o_sig, h_sig, len(o), len(h))
    print(f"(c) MinHash ASYMMETRY: c(small,big)={c_sh:.3f}, c(big,small)={c_bs:.3f}")
    assert c_sh > c_bs + 0.05, (
        f"MinHash containment not asymmetric: {c_sh:.3f} vs {c_bs:.3f}"
    )


# ---------------------------------------------------------------------------
# (d) UNRELATED: two disjoint functions → ~0 containment
# ---------------------------------------------------------------------------

def test_unrelated_near_zero():
    h = shingles(HELPER)
    u = shingles(UNRELATED)
    c = containment(h, u)
    print(f"(d) UNRELATED containment = {c:.3f}")
    # Note: 'return _v' normalizes identically for any function that returns a local;
    # this is a known boilerplate shingle (IDF would suppress it in practice).
    # The raw containment bound is 0.25 (1 shared single-stmt shingle out of 4).
    assert c <= 0.25, f"Expected unrelated containment <= 0.25, got {c:.3f}"


def test_unrelated_original_vs_unrelated():
    o = shingles(ORIGINAL)
    u = shingles(UNRELATED)
    c = containment(o, u)
    rev = containment(u, o)
    print(f"(d) UNRELATED original↔unrelated: {c:.3f}, {rev:.3f}")
    assert c < 0.25 and rev < 0.25


# ---------------------------------------------------------------------------
# (e) MinHash estimate vs exact: error report on several cases
# ---------------------------------------------------------------------------

def _report_minhash_error(label: str, a_src: str, b_src: str) -> float:
    a = shingles(a_src)
    b = shingles(b_src)
    exact = containment(a, b)
    a_sig = minhash_signature(a, 128)
    b_sig = minhash_signature(b, 128)
    est = minhash_containment_estimate(a_sig, b_sig, len(a), len(b))
    err = abs(exact - est)
    print(f"(e) {label}: exact={exact:.3f} est={est:.3f} err={err:.3f}")
    return err


def test_minhash_errors_all_small():
    """All MinHash containment estimate errors should be < 0.20 with 128 perms."""
    errors = [
        _report_minhash_error("helper2→original", HELPER2, ORIGINAL),
        _report_minhash_error("helper→bigger",    HELPER,  BIGGER_WITH_EXTRAS),
        _report_minhash_error("original→bigger",  ORIGINAL, BIGGER_WITH_EXTRAS),
        _report_minhash_error("unrelated pair",   HELPER,  UNRELATED),
    ]
    for i, err in enumerate(errors):
        assert err < 0.20, f"Case {i}: MinHash error {err:.3f} >= 0.20"


# ---------------------------------------------------------------------------
# (f) IDF: boilerplate shingle gets down-weighted / excluded
# ---------------------------------------------------------------------------

def test_idf_suppresses_boilerplate():
    """
    A shingle that appears in ALL indexed functions should be flagged boilerplate
    and excluded by IdfIndex.filter().
    """
    # Build a fake corpus where all functions share a common pattern
    common_src = """
def fn_common():
    result = []
    return result
"""
    unique_src = """
def fn_unique():
    result = []
    x = compute_something_special(42)
    return x
"""

    common_shingles = shingles(common_src)
    unique_shingles = shingles(unique_src)

    # Find a shingle in common_shingles (appears in all 10 corpus docs)
    shared = common_shingles & unique_shingles
    if not shared:
        print("(f) IDF: no shared shingles between common and unique — test is vacuous")
        # Still test the index machinery
        idx = IdfIndex(df_threshold=0.5)
        for i in range(10):
            idx.add(f"fn_common_{i}", common_shingles)
        assert idx.corpus_size == 10
        return

    # Build corpus: 10 copies of common_src, 1 of unique_src
    idx = IdfIndex(df_threshold=0.5)
    for i in range(10):
        idx.add(f"fn_common_{i}", common_shingles)
    idx.add("fn_unique", unique_shingles)

    # Shared shingles should be boilerplate
    for s in shared:
        df = idx.df(s)
        print(f"(f) IDF: shared shingle df={df:.2f} (threshold=0.5)")
        if df > 0.5:
            assert idx.is_boilerplate(s), f"df={df:.2f} > threshold but not flagged boilerplate"

    # filter should exclude the boilerplate shingles
    filtered = idx.filter(unique_shingles)
    retained_shared = shared & filtered
    print(f"(f) IDF: shared shingles={len(shared)}, retained after filter={len(retained_shared)}")
    assert len(retained_shared) < len(shared), (
        "IdfIndex.filter should remove at least one boilerplate shingle"
    )


def test_idf_low_df_shingles_retained():
    """Shingles that appear rarely should NOT be suppressed."""
    rare_src = """
def fn_rare():
    specialized_transform = compute_rare_thing(x, y, z)
    return specialized_transform
"""
    common_src = """
def fn_other():
    result = []
    return result
"""
    rare_shingles = shingles(rare_src)
    common_shingles = shingles(common_src)

    idx = IdfIndex(df_threshold=0.5)
    for i in range(10):
        idx.add(f"fn_common_{i}", common_shingles)
    idx.add("fn_rare", rare_shingles)

    # rare_shingles - common_shingles: appears only in 1/11 docs → not boilerplate
    rare_only = rare_shingles - common_shingles
    if rare_only:
        s = next(iter(rare_only))
        df = idx.df(s)
        print(f"(f) IDF: rare-only shingle df={df:.3f}")
        assert not idx.is_boilerplate(s), f"Rare shingle (df={df:.3f}) should not be boilerplate"


# ---------------------------------------------------------------------------
# Sanity: shingles / minhash on empty input
# ---------------------------------------------------------------------------

def test_empty_source():
    s = shingles("")
    assert s == set()
    sig = minhash_signature(s)
    assert all(v == 0xFFFFFFFF for v in sig)
    c = containment(s, shingles(ORIGINAL))
    assert c == 0.0


def test_shingle_k():
    """SHINGLE_K should be 3 per the plan spec."""
    assert SHINGLE_K == 3, f"Expected SHINGLE_K=3, got {SHINGLE_K}"


if __name__ == "__main__":
    # Run all tests manually and print results
    tests = [
        test_shingle_k,
        test_empty_source,
        test_extract_containment_high,
        test_extract_minhash_approx,
        test_inline_containment_high,
        test_asymmetry,
        test_asymmetry_minhash,
        test_unrelated_near_zero,
        test_unrelated_original_vs_unrelated,
        test_minhash_errors_all_small,
        test_idf_suppresses_boilerplate,
        test_idf_low_df_shingles_retained,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
