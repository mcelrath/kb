"""
Tests for kb/configure.py — Phase 5 (kb-2c3.5).

All tests use TEMP dirs as config/project targets; never touch real ~/.claude.
Run: python3 tmp/configure_test/test_configure.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure the kb package is importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kb.configure import (
    _is_gitignored,
    _merge_env,
    _load_json_safe,
    _verify_and_ensure_gitignored,
    _write_secret_key,
    run_global_configure,
    run_project_configure,
    _merge_beads_config,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        msg = f"{name}: {detail}" if detail else name
        print(f"  {FAIL}  {msg}")
        _failures.append(msg)


# ---------------------------------------------------------------------------
# Helper: init a real git repo in a temp dir
# ---------------------------------------------------------------------------


def _make_git_repo(d: Path) -> None:
    subprocess.run(["git", "init", str(d)], capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        capture_output=True, check=True, cwd=str(d),
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        capture_output=True, check=True, cwd=str(d),
    )


# ---------------------------------------------------------------------------
# Test 1: global non-interactive configure — env merge
# ---------------------------------------------------------------------------


def test_global_configure_env_merge() -> None:
    print("\n[1] Global non-interactive configure — env merge")
    with tempfile.TemporaryDirectory() as td:
        config_dir = Path(td)

        # Pre-seed settings.json with an unrelated key + existing env key
        pre = {"env": {"FOO": "bar", "KB_EMBEDDING_FORMAT": "OLD"}, "otherKey": 1}
        settings_path = config_dir / "settings.json"
        settings_path.write_text(json.dumps(pre))

        rc = run_global_configure(
            provider="ollama-local",
            model="nomic-embed-text",
            dim=768,
            fmt="openai",
            url="http://localhost:11434/v1/embeddings",
            summary_mode="local-llm",
            key=None,
            reembed=False,
            config_dir=config_dir,
            db_path=None,
            interactive=False,
        )
        check("returns 0", rc == 0, f"rc={rc}")

        data = _load_json_safe(settings_path)
        env = data.get("env", {})

        check("KB_EMBEDDING_FORMAT set", env.get("KB_EMBEDDING_FORMAT") == "openai",
              repr(env.get("KB_EMBEDDING_FORMAT")))
        check("KB_EMBEDDING_URL set", env.get("KB_EMBEDDING_URL") == "http://localhost:11434/v1/embeddings",
              repr(env.get("KB_EMBEDDING_URL")))
        check("KB_EMBEDDING_MODEL set", env.get("KB_EMBEDDING_MODEL") == "nomic-embed-text",
              repr(env.get("KB_EMBEDDING_MODEL")))
        check("KB_EMBEDDING_DIM set", env.get("KB_EMBEDDING_DIM") == "768",
              repr(env.get("KB_EMBEDDING_DIM")))
        check("KB_SUMMARY_MODE set", env.get("KB_SUMMARY_MODE") == "local-llm",
              repr(env.get("KB_SUMMARY_MODE")))

        # Merge: pre-existing env keys survive
        check("FOO survives", env.get("FOO") == "bar", repr(env.get("FOO")))
        # Merge: top-level non-env key survives
        check("otherKey survives", data.get("otherKey") == 1, repr(data.get("otherKey")))


# ---------------------------------------------------------------------------
# Test 2: secret guard — write path (git repo, settings.local.json not yet ignored)
# ---------------------------------------------------------------------------


def test_secret_guard_write() -> None:
    print("\n[2] Secret guard — write path (git repo, auto-adds to .gitignore)")
    with tempfile.TemporaryDirectory() as td:
        config_dir = Path(td)
        _make_git_repo(config_dir)

        # No .gitignore yet — configure should add settings.local.json to it
        ok, msg = _write_secret_key(config_dir, "SEKRET")
        check("write succeeds", ok, msg)
        check("settings.local.json gitignored after write",
              _is_gitignored(config_dir, "settings.local.json"),
              "not confirmed by git check-ignore")
        local_path = config_dir / "settings.local.json"
        check("settings.local.json exists", local_path.exists(), "file missing")
        if local_path.exists():
            data = _load_json_safe(local_path)
            check("key written correctly", data.get("env", {}).get("KB_EMBEDDING_KEY") == "SEKRET",
                  repr(data))


# ---------------------------------------------------------------------------
# Test 3: secret guard — REFUSE path (dir that is not a git repo AND cannot
# git check-ignore, but we fake the "cannot confirm" path by monkeypatching)
# ---------------------------------------------------------------------------


def test_secret_guard_refuse() -> None:
    print("\n[3] Secret guard — refuse when gitignore cannot be confirmed")
    import unittest.mock as mock

    with tempfile.TemporaryDirectory() as td:
        config_dir = Path(td)
        _make_git_repo(config_dir)

        # Patch _is_gitignored to always return False (simulates broken git env)
        with mock.patch("kb.configure._is_gitignored", return_value=False):
            ok, msg = _write_secret_key(config_dir, "SHOULD_NOT_WRITE")

        check("write refused (ok=False)", ok is False, f"ok={ok}")
        check("error message mentions gitignore", "gitignore" in msg.lower() or "ignored" in msg.lower(),
              repr(msg))

        local_path = config_dir / "settings.local.json"
        check("key NOT written to settings.local.json",
              not local_path.exists() or "SHOULD_NOT_WRITE" not in local_path.read_text(),
              "key leaked")


# ---------------------------------------------------------------------------
# Test 4: full non-interactive configure with --key (write + verify gitignore)
# ---------------------------------------------------------------------------


def test_global_configure_with_key() -> None:
    print("\n[4] Global configure with --key — gitignore auto-added + key in local.json")
    with tempfile.TemporaryDirectory() as td:
        config_dir = Path(td)
        _make_git_repo(config_dir)

        rc = run_global_configure(
            provider="voyage",
            model="voyage-3-lite",
            dim=512,
            fmt="openai",
            url="https://api.voyageai.com/v1/embeddings",
            summary_mode="none",
            key="VYG_SECRET_KEY",
            reembed=False,
            config_dir=config_dir,
            db_path=None,
            interactive=False,
        )
        check("returns 0", rc == 0, f"rc={rc}")

        local_path = config_dir / "settings.local.json"
        check("settings.local.json created", local_path.exists())
        if local_path.exists():
            data = _load_json_safe(local_path)
            check("key in local.json", data.get("env", {}).get("KB_EMBEDDING_KEY") == "VYG_SECRET_KEY")

        # Key must NOT be in tracked settings.json
        settings_data = _load_json_safe(config_dir / "settings.json")
        tracked_env = settings_data.get("env", {})
        check("key NOT in settings.json", "KB_EMBEDDING_KEY" not in tracked_env,
              f"LEAKED: {tracked_env}")

        check("settings.local.json gitignored",
              _is_gitignored(config_dir, "settings.local.json"))


# ---------------------------------------------------------------------------
# Test 5: --project configure — .beads/config.yaml + KB_DB
# ---------------------------------------------------------------------------


def test_project_configure() -> None:
    print("\n[5] Project configure — .beads/config.yaml backend:kb + KB_DB env")
    with tempfile.TemporaryDirectory() as td:
        proj_dir = Path(td)

        rc = run_project_configure(
            project_tag="myproj",
            enable_tracker=True,
            db_override="/tmp/x.db",
            key=None,
            project_dir=proj_dir,
        )
        check("returns 0", rc == 0, f"rc={rc}")

        beads_yaml = proj_dir / ".beads" / "config.yaml"
        check(".beads/config.yaml exists", beads_yaml.exists())
        if beads_yaml.exists():
            content = beads_yaml.read_text()
            check("backend: kb in config.yaml", "backend" in content and "kb" in content,
                  repr(content))

        proj_settings = proj_dir / ".claude" / "settings.json"
        check(".claude/settings.json exists", proj_settings.exists())
        if proj_settings.exists():
            data = _load_json_safe(proj_settings)
            check("KB_DB in project env", data.get("env", {}).get("KB_DB") == "/tmp/x.db",
                  repr(data))


# ---------------------------------------------------------------------------
# Test 6: merge-not-clobber direct unit test
# ---------------------------------------------------------------------------


def test_merge_not_clobber() -> None:
    print("\n[6] _merge_env — merge-not-clobber")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "settings.json"
        path.write_text(json.dumps({"env": {"FOO": "bar"}, "otherKey": 1}))

        _merge_env(path, {"KB_EMBEDDING_DIM": "4096"})

        data = _load_json_safe(path)
        check("FOO preserved", data["env"].get("FOO") == "bar")
        check("otherKey preserved", data.get("otherKey") == 1)
        check("KB_EMBEDDING_DIM written", data["env"].get("KB_EMBEDDING_DIM") == "4096")


# ---------------------------------------------------------------------------
# Test 7: _merge_beads_config
# ---------------------------------------------------------------------------


def test_merge_beads_config() -> None:
    print("\n[7] _merge_beads_config — merge-not-clobber for YAML")
    with tempfile.TemporaryDirectory() as td:
        yaml_path = Path(td) / "config.yaml"
        # Write existing content
        yaml_path.parent.mkdir(exist_ok=True)
        yaml_path.write_text("other_key: some_value\n")

        _merge_beads_config(yaml_path, {"backend": "kb"})
        content = yaml_path.read_text()
        check("backend: kb present", "backend" in content and "kb" in content, repr(content))
        check("other_key preserved", "other_key" in content, repr(content))


# ---------------------------------------------------------------------------
# Test 8: py_compile checks
# ---------------------------------------------------------------------------


def test_py_compile() -> None:
    print("\n[8] py_compile kb/configure.py and kb.py")
    project_root = Path(__file__).parent.parent.parent

    for fname in ["kb/configure.py", "kb.py"]:
        fpath = project_root / fname
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(fpath)],
            capture_output=True, text=True,
        )
        check(f"py_compile {fname}", result.returncode == 0,
              result.stderr.strip() or result.stdout.strip())


# ---------------------------------------------------------------------------
# Test 9: non-interactive via CLI (subprocess)
# ---------------------------------------------------------------------------


def test_cli_non_interactive() -> None:
    print("\n[9] CLI non-interactive: kb configure --provider ollama ...")
    with tempfile.TemporaryDirectory() as td:
        config_dir = Path(td)
        # Pre-seed with unrelated key
        pre = {"env": {"UNRELATED": "val"}, "anotherKey": 99}
        (config_dir / "settings.json").write_text(json.dumps(pre))

        kb_py = str(Path(__file__).parent.parent.parent / "kb.py")
        result = subprocess.run(
            [
                sys.executable, kb_py,
                "configure",
                "--provider", "ollama-local",
                "--model", "nomic-embed-text",
                "--dim", "768",
                "--format", "openai",
                "--url", "http://localhost:11434/v1/embeddings",
                "--summary-mode", "local-llm",
                "--config-dir", str(config_dir),
            ],
            capture_output=True, text=True,
        )
        check("exits 0", result.returncode == 0,
              result.stderr[:300] or result.stdout[:300])

        data = _load_json_safe(config_dir / "settings.json")
        env = data.get("env", {})
        check("UNRELATED survives", env.get("UNRELATED") == "val", repr(env))
        check("anotherKey survives", data.get("anotherKey") == 99, repr(data))
        check("KB_EMBEDDING_MODEL set", env.get("KB_EMBEDDING_MODEL") == "nomic-embed-text",
              repr(env))
        check("KB_EMBEDDING_DIM=768", env.get("KB_EMBEDDING_DIM") == "768", repr(env))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    test_global_configure_env_merge()
    test_secret_guard_write()
    test_secret_guard_refuse()
    test_global_configure_with_key()
    test_project_configure()
    test_merge_not_clobber()
    test_merge_beads_config()
    test_py_compile()
    test_cli_non_interactive()

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} FAILURE(S):\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\033[32mAll tests passed.\033[0m")
        sys.exit(0)
