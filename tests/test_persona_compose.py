"""Tests for the persona archetype composer (hooks/scripts/lib/persona_compose.py).

Layer order: L1 archetype -> L2 augmentation -> persona instance body.
"""

import importlib.util
from pathlib import Path

_COMPOSER = Path(__file__).resolve().parents[1] / "hooks" / "scripts" / "lib" / "persona_compose.py"
_spec = importlib.util.spec_from_file_location("persona_compose", _COMPOSER)
assert _spec and _spec.loader
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)


def _mk(tmp_path):
    """Build a fake plugin root + persona dir; return (plugin_root, persona_dir)."""
    plugin_root = tmp_path / "plugin"
    arch_dir = plugin_root / "skills" / "persona" / "archetypes"
    arch_dir.mkdir(parents=True)
    (arch_dir / "architect.md").write_text(
        "---\nname: architect\narchetype: architect\n---\n\nL1-MARKER\n", encoding="utf-8"
    )
    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    (persona_dir / "physics-augmentation.md").write_text(
        "---\nname: physics-augmentation\n---\n\nL2-MARKER\n", encoding="utf-8"
    )
    return str(plugin_root), str(persona_dir)


def test_archetype_plus_augmentation_order(tmp_path):
    plugin_root, persona_dir = _mk(tmp_path)
    pf = Path(persona_dir) / "archie.md"
    pf.write_text(
        "---\nname: archie\narchetype: architect\naugmentation: physics-augmentation\n---\n\nINSTANCE-MARKER\n",
        encoding="utf-8",
    )
    assert pc.compose(str(pf), plugin_root, persona_dir) == "L1-MARKER\n\nL2-MARKER\n\nINSTANCE-MARKER"


def test_archetype_only_body_is_l2_no_dup(tmp_path):
    plugin_root, persona_dir = _mk(tmp_path)
    pf = Path(persona_dir) / "archie.md"
    pf.write_text("---\nname: archie\narchetype: architect\n---\n\nBODY-AS-L2\n", encoding="utf-8")
    assert pc.compose(str(pf), plugin_root, persona_dir) == "L1-MARKER\n\nBODY-AS-L2"


def test_legacy_no_archetype_flat_body_preserves_leading_dash(tmp_path):
    # A body starting with a markdown list dash MUST survive (the lstrip('---\\n') bug).
    plugin_root, persona_dir = _mk(tmp_path)
    pf = Path(persona_dir) / "tip.md"
    pf.write_text("---\nname: tip\nrole: prover\n---\n\n- first item\n- second item\n", encoding="utf-8")
    assert pc.compose(str(pf), plugin_root, persona_dir) == "- first item\n- second item"


def test_missing_archetype_file_emits_instance_body(tmp_path):
    plugin_root, persona_dir = _mk(tmp_path)
    pf = Path(persona_dir) / "archie.md"
    pf.write_text("---\nname: archie\narchetype: nonesuch\n---\n\nSTILL-HERE\n", encoding="utf-8")
    # archetype file missing -> skipped with a stderr warning; instance body still emitted.
    assert pc.compose(str(pf), plugin_root, persona_dir) == "STILL-HERE"


def test_no_frontmatter_returns_whole_text(tmp_path):
    plugin_root, persona_dir = _mk(tmp_path)
    pf = Path(persona_dir) / "raw.md"
    pf.write_text("NO-FM-CONTENT\nsecond line\n", encoding="utf-8")
    assert pc.compose(str(pf), plugin_root, persona_dir) == "NO-FM-CONTENT\nsecond line"


def test_archetype_only_empty_body_is_just_l1(tmp_path):
    # Thin persona: frontmatter declares the archetype, no instance body. L1 alone, no dangle.
    plugin_root, persona_dir = _mk(tmp_path)
    pf = Path(persona_dir) / "archie.md"
    pf.write_text("---\nname: archie\narchetype: architect\n---\n", encoding="utf-8")
    assert pc.compose(str(pf), plugin_root, persona_dir) == "L1-MARKER"


def test_module_is_pure_stdlib_no_kb_import():
    # SessionStart hot path: persona_compose must NOT pull the kb package (facade/core/numpy).
    # Fresh interpreter so suite-mates that import kb don't pollute sys.modules.
    import subprocess
    import sys as _sys
    code = (
        "import importlib.util,sys;"
        f"spec=importlib.util.spec_from_file_location('pc',{str(_COMPOSER)!r});"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "bad=[x for x in sys.modules if x=='kb' or x.startswith('kb.') or x=='numpy'];"
        "print('BAD:'+','.join(bad)) if bad else print('CLEAN')"
    )
    out = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "CLEAN", f"persona_compose dragged in heavy deps: {out.stdout} {out.stderr}"
