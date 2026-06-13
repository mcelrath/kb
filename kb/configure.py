"""
kb configure — two-layer config UX (Phase 5, epic kb-2c3).

Layer A: GLOBAL `kb configure` (interactive + non-interactive flags)
  - Writes non-secret KB_* env vars into GLOBAL settings.json env block (merge).
  - Writes secret KB_EMBEDDING_KEY to settings.local.json ONLY after verifying
    settings.local.json is gitignored; refuses otherwise (blocker-2 fix).
  - Seeds embedding_meta in the configured db; prints reembed instruction if
    model/dim changed vs an existing index.

Layer B: `kb configure --project <tag>` (non-interactive)
  - Sets project tag; with --enable-tracker writes .beads/config.yaml backend:kb.
  - Writes optional per-project KB_DB to the project's .claude/settings.json (merge).
  - Applies the same gitignore+check-ignore guard if --key is supplied.

Do NOT import from kb.core.embedding, kb.facade, kb.core.schema, kb.issue_cli,
or kb.llm (other phases own them); uses only stdlib + kb.constants.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Provider catalogue
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict[str, Any]] = {
    "ollama-local": {
        "label": "Ollama local (recommended; free, private)",
        "format": "openai",
        "url": "http://localhost:11434/v1/embeddings",
        "models": [
            ("qwen3-embedding:0.6b", 1024, "CPU-capable ~0.6B; strong on code+science (MTEB-Code 75) — recommended default"),
            ("qwen3-embedding:8b", 4096, "GPU (~16GB); top code+science quality"),
            ("nomic-embed-text", 768, "tiny/fast but WEAK on code+science — only if RAM-starved"),
        ],
        "default_model": "qwen3-embedding:0.6b",
        "default_dim": 1024,
        "needs_key": False,
    },
    "voyage": {
        "label": "Voyage AI (free hosted, code-specialized; 200M tokens free, no GPU)",
        "format": "openai",
        "url": "https://api.voyageai.com/v1/embeddings",
        "models": [
            ("voyage-code-3", 1024, "code-specialized, top code retrieval — recommended hosted"),
            ("voyage-3.5", 1024, "strong general"),
        ],
        "default_model": "voyage-code-3",
        "default_dim": 1024,
        "needs_key": True,
    },
    "openai": {
        "label": "OpenAI embeddings",
        "format": "openai",
        "url": "https://api.openai.com/v1/embeddings",
        "models": [
            ("text-embedding-3-small", 1536, "fast, cheap"),
            ("text-embedding-3-large", 3072, "higher quality"),
        ],
        "default_model": "text-embedding-3-small",
        "default_dim": 1536,
        "needs_key": True,
    },
    "gemini": {
        "label": "Google Gemini embeddings (OpenAI-compatible path)",
        "format": "openai",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/embeddings",
        "models": [
            ("text-embedding-004", 768, "Google's latest"),
        ],
        "default_model": "text-embedding-004",
        "default_dim": 768,
        "needs_key": True,
    },
    "jina": {
        "label": "Jina AI embeddings",
        "format": "openai",
        "url": "https://api.jina.ai/v1/embeddings",
        "models": [
            ("jina-embeddings-v3", 1024, "current generation"),
        ],
        "default_model": "jina-embeddings-v3",
        "default_dim": 1024,
        "needs_key": True,
    },
    "local-llamacpp": {
        "label": "Local llama.cpp server (legacy default, ash:8081)",
        "format": "llamacpp",
        "url": "http://ash:8081/embedding",
        "models": [
            ("", 4096, "model identity set on server"),
        ],
        "default_model": "",
        "default_dim": 4096,
        "needs_key": False,
    },
}

SUMMARY_MODES = ["extractive", "none", "local-llm", "subscription-sdk", "api"]
SUMMARY_MODE_LABELS = {
    "extractive": "Extractive, no LLM (default — first sentence, zero VRAM/cost)",
    "none": "No summaries (raw content shown)",
    "local-llm": "Local LLM server (needs a second model)",
    "subscription-sdk": "Claude Haiku via Agent SDK (subscription credits)",
    "api": "Claude API (requires ANTHROPIC_API_KEY)",
}

# ---------------------------------------------------------------------------
# JSON merge helpers (NEVER clobber existing keys)
# ---------------------------------------------------------------------------


def _load_json_safe(path: Path) -> dict[str, Any]:
    """Load JSON from path; return {} if file missing or unparseable."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _merge_env(settings_path: Path, env_updates: dict[str, str]) -> None:
    """Merge env_updates into settings_path["env"], preserving all other keys.

    Load-modify-dump: never replaces top-level keys outside "env".
    """
    data = _load_json_safe(settings_path)
    env = data.get("env", {})
    env.update(env_updates)
    data["env"] = env
    _save_json(settings_path, data)


# ---------------------------------------------------------------------------
# Secret guard: ensure settings.local.json is gitignored before writing key
# ---------------------------------------------------------------------------


def _gitignore_path(config_dir: Path) -> Path:
    return config_dir / ".gitignore"


def _add_to_gitignore(config_dir: Path, filename: str) -> None:
    """Append filename to config_dir/.gitignore if not already present."""
    gi = _gitignore_path(config_dir)
    existing = gi.read_text() if gi.exists() else ""
    lines = existing.splitlines()
    if filename not in lines:
        with gi.open("a") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(f"{filename}\n")


def _is_gitignored(config_dir: Path, filename: str) -> bool:
    """Return True iff git check-ignore confirms the file is ignored."""
    target = config_dir / filename
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(target)],
            capture_output=True,
            cwd=str(config_dir),
        )
        return result.returncode == 0
    except FileNotFoundError:
        # git not available; fall back to scanning .gitignore lines
        gi = _gitignore_path(config_dir)
        if not gi.exists():
            return False
        lines = [ln.strip() for ln in gi.read_text().splitlines()]
        return filename in lines


def _verify_and_ensure_gitignored(config_dir: Path, filename: str) -> tuple[bool, str]:
    """Ensure `filename` is gitignored in `config_dir`.

    Returns (ok: bool, message: str).
    - If already ignored: (True, "already gitignored")
    - If .gitignore exists (or config_dir is a git repo): adds entry, re-checks.
    - If still not confirmed: (False, reason)
    """
    # Fast path: already ignored
    if _is_gitignored(config_dir, filename):
        return True, "already gitignored"

    # Try to add to .gitignore and re-confirm
    gi = _gitignore_path(config_dir)
    # Check whether config_dir is inside a git repo at all
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            check=True,
            cwd=str(config_dir),
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repo; writing the key here would not leak via git.
        # This is still safe, so allow it.
        return True, "config dir is not a git repo; key write is safe"

    _add_to_gitignore(config_dir, filename)

    if _is_gitignored(config_dir, filename):
        return True, f"added {filename} to {gi}"

    return False, (
        f"Cannot confirm {filename} is gitignored in {config_dir}. "
        "Refusing to write secret. Add it to .gitignore manually and retry."
    )


def _write_secret_key(
    config_dir: Path, key: str, env_var: str = "KB_EMBEDDING_KEY"
) -> tuple[bool, str]:
    """Write secret to settings.local.json after verifying gitignore.

    Returns (ok, message).
    """
    ok, msg = _verify_and_ensure_gitignored(config_dir, "settings.local.json")
    if not ok:
        return False, msg

    local_path = config_dir / "settings.local.json"
    _merge_env(local_path, {env_var: key})
    return True, f"Secret {env_var} written to {local_path}"


# ---------------------------------------------------------------------------
# config.toml writer (source of truth for non-secret embedding config)
# ---------------------------------------------------------------------------


def _write_config_toml(
    fmt: str,
    url: str,
    model: str,
    dim: int,
    summary_mode: str,
    toml_path: Path | None = None,
) -> None:
    """Write non-secret embedding + LLM config to ~/.config/kb/config.toml.

    Secret KB_EMBEDDING_KEY is intentionally excluded — it stays in
    settings.local.json (gitignore-guarded).

    The toml is the authoritative persistent config read by kb.config.load_config().
    settings.json env stays in sync so the Claude Code harness also sees the values.
    """
    if toml_path is None:
        toml_path = Path.home() / ".config" / "kb" / "config.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)

    content = (
        "# kb configuration — written by `kb configure`.\n"
        "# Non-secret values only; KB_EMBEDDING_KEY lives in settings.local.json.\n\n"
        "[embedding]\n"
        f'url    = "{url}"\n'
        f"dim    = {dim}\n"
        f'format = "{fmt}"\n'
        f'model  = "{model}"\n\n'
        "[llm]\n"
        f'summary_mode = "{summary_mode}"\n'
    )
    toml_path.write_text(content)
    print(f"Written config.toml to {toml_path}")


# ---------------------------------------------------------------------------
# embedding_meta seed / reembed-prompt helper
# ---------------------------------------------------------------------------


def _seed_embedding_meta_if_needed(
    db_path: Path,
    fmt: str,
    url: str,
    model: str,
    dim: int,
) -> dict[str, Any]:
    """Open the target DB and upsert embedding_meta.

    Returns dict with keys: seeded (bool), model_changed (bool), dim_changed (bool),
    prior_model, prior_dim.

    This function only imports stdlib + sqlite3. It does NOT import kb.facade.
    The embedding_meta table is guaranteed to exist (created by init_schema in kb-2c3.2).
    If the table doesn't exist yet we skip silently (DB not initialized).
    """
    import sqlite3

    if not db_path.exists():
        return {"seeded": False, "reason": "db not found"}

    sig = hashlib.sha256(f"{fmt}|{url}|{model}|{dim}".encode()).hexdigest()
    now_str = __import__("datetime").datetime.now().isoformat()

    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT format, url, model, dim, signature FROM embedding_meta WHERE id = 1"
        ).fetchone()
    except sqlite3.OperationalError:
        # Table doesn't exist yet (pre-migration DB or brand-new)
        return {"seeded": False, "reason": "embedding_meta table missing"}

    if row is None:
        conn.execute(
            "INSERT INTO embedding_meta (id, format, url, model, dim, signature, updated_at) "
            "VALUES (1, ?, ?, ?, ?, ?, ?)",
            (fmt, url, model, dim, sig, now_str),
        )
        conn.commit()
        conn.close()
        return {
            "seeded": True,
            "model_changed": False,
            "dim_changed": False,
            "prior_model": None,
            "prior_dim": None,
        }

    prior_model = row["model"]
    prior_dim = row["dim"]
    model_changed = row["signature"] != sig
    dim_changed = int(row["dim"]) != int(dim)

    conn.execute(
        "UPDATE embedding_meta SET format=?, url=?, model=?, dim=?, signature=?, updated_at=? "
        "WHERE id = 1",
        (fmt, url, model, dim, sig, now_str),
    )
    conn.commit()
    conn.close()
    return {
        "seeded": True,
        "model_changed": model_changed,
        "dim_changed": dim_changed,
        "prior_model": prior_model,
        "prior_dim": prior_dim,
    }


# ---------------------------------------------------------------------------
# Interactive prompt helpers (only used in tty / non-agent mode)
# ---------------------------------------------------------------------------


def _prompt(label: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        val = input(f"{label}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val if val else default


def _choose(label: str, options: list[tuple[str, str]], default_key: str) -> str:
    """Present numbered menu; return chosen key."""
    print(f"\n{label}")
    for i, (key, desc) in enumerate(options, 1):
        marker = " *" if key == default_key else ""
        print(f"  {i}. {desc}{marker}")
    while True:
        raw = _prompt(f"Choice (1-{len(options)}, Enter={default_key})", default_key)
        if raw == default_key:
            return default_key
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        except ValueError:
            pass
        # Maybe user typed the key directly
        if raw in dict(options):
            return raw
        print(f"  Invalid choice; enter a number 1-{len(options)}.")


# ---------------------------------------------------------------------------
# Core configure logic
# ---------------------------------------------------------------------------


def _check_ollama(url: str, model: str) -> None:
    """If the embedding URL is an ollama endpoint, verify ollama is reachable on
    its port and the model is pulled; advise the user otherwise. Advisory only —
    never fails configure (the user may start ollama / pull the model later)."""
    if "11434" not in url and "ollama" not in url.lower():
        return
    import urllib.request

    base = url.split("/v1/")[0].split("/api/")[0].rstrip("/")
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=3) as r:
            tags = json.loads(r.read())
    except Exception:
        print(
            f"\n  ⚠ Ollama is NOT reachable at {base}.\n"
            f"    Start it:  ollama serve   (or `systemctl --user start ollama`)\n"
            f"    Then pull the embedding model:  ollama pull {model}"
        )
        return
    have = {m.get("name", "") for m in tags.get("models", [])}
    short = model.split(":")[0]
    if model in have or any(n.split(":")[0] == short for n in have):
        print(f"  ✓ Ollama running at {base}; embedding model '{model}' is available.")
    else:
        installed = ", ".join(sorted(have)) or "(none)"
        print(
            f"\n  ⚠ Ollama is running at {base} but model '{model}' is NOT pulled.\n"
            f"    Installed: {installed}\n"
            f"    Download it:  ollama pull {model}"
        )


def run_global_configure(
    provider: str | None,
    model: str | None,
    dim: int | None,
    fmt: str | None,
    url: str | None,
    summary_mode: str | None,
    key: str | None,
    reembed: bool,
    config_dir: Path,
    db_path: Path | None,
    interactive: bool,
) -> int:
    """Write global config. Returns 0 on success, 1 on error."""
    from kb.constants import DEFAULT_DB_PATH

    # ------------------------------------------------------------------
    # 1. Resolve provider + model + dim + format + url
    # ------------------------------------------------------------------
    if interactive and provider is None:
        # Show interactive menu
        options = [(k, v["label"]) for k, v in PROVIDERS.items()]
        provider = _choose("Embedding provider", options, "ollama-local")

    if provider is None:
        provider = "ollama-local"

    pdata = PROVIDERS.get(provider)
    if pdata is None:
        print(f"Error: unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
        return 1

    resolved_fmt = fmt if fmt is not None else pdata["format"]
    resolved_url = url if url is not None else pdata["url"]
    resolved_model = model
    resolved_dim = dim

    if interactive and resolved_model is None and len(pdata["models"]) > 1:
        model_options = [(m, f"{m} (dim={d}) — {note}") for m, d, note in pdata["models"]]
        resolved_model = _choose("Embedding model", model_options, pdata["default_model"])

    if resolved_model is None:
        resolved_model = pdata["default_model"]

    if resolved_dim is None:
        # Pick dim matching the chosen model from the provider catalogue
        for m, d, _ in pdata["models"]:
            if m == resolved_model:
                resolved_dim = d
                break
        if resolved_dim is None:
            resolved_dim = pdata["default_dim"]

    # Ollama reachability + model-presence check (advisory; prompts to pull a model)
    _check_ollama(resolved_url, resolved_model)

    # ------------------------------------------------------------------
    # 2. Summary mode
    # ------------------------------------------------------------------
    resolved_summary = summary_mode
    if interactive and resolved_summary is None:
        sm_options = [(k, f"{k}: {SUMMARY_MODE_LABELS[k]}") for k in SUMMARY_MODES]
        resolved_summary = _choose("Summary mode", sm_options, "extractive")
    if resolved_summary is None:
        resolved_summary = "extractive"

    # ------------------------------------------------------------------
    # 3. Key (secret)
    # ------------------------------------------------------------------
    resolved_key = key
    if interactive and pdata["needs_key"] and resolved_key is None:
        raw = _prompt(f"API key for {provider} (leave blank to skip)")
        if raw:
            resolved_key = raw

    # ------------------------------------------------------------------
    # 4a. Write non-secret config to ~/.config/kb/config.toml (source of truth)
    # ------------------------------------------------------------------
    _write_config_toml(resolved_fmt, resolved_url, resolved_model, resolved_dim, resolved_summary)

    # ------------------------------------------------------------------
    # 4b. Write non-secret env to settings.json (MERGE) — kept so the
    #     harness env block stays in sync for processes that don't load
    #     the toml directly.
    # ------------------------------------------------------------------
    settings_path = config_dir / "settings.json"
    env_updates: dict[str, str] = {
        "KB_EMBEDDING_FORMAT": resolved_fmt,
        "KB_EMBEDDING_URL": resolved_url,
        "KB_EMBEDDING_MODEL": resolved_model,
        "KB_EMBEDDING_DIM": str(resolved_dim),
        "KB_SUMMARY_MODE": resolved_summary,
    }
    _merge_env(settings_path, env_updates)
    print(f"Written non-secret config to {settings_path}")

    # ------------------------------------------------------------------
    # 5. Write secret to settings.local.json (gitignore-verified)
    # ------------------------------------------------------------------
    if resolved_key:
        ok, msg = _write_secret_key(config_dir, resolved_key)
        if not ok:
            print(f"Error: {msg}", file=sys.stderr)
            return 1
        print(msg)

    # ------------------------------------------------------------------
    # 6. Seed embedding_meta + reembed prompt
    # ------------------------------------------------------------------
    effective_db = db_path or DEFAULT_DB_PATH
    meta = _seed_embedding_meta_if_needed(
        effective_db, resolved_fmt, resolved_url, resolved_model, resolved_dim
    )

    if meta.get("seeded"):
        if meta.get("model_changed") or meta.get("dim_changed"):
            prior_model = meta.get("prior_model") or "(unknown)"
            prior_dim = meta.get("prior_dim") or "(unknown)"
            print(
                f"\nEmbedding model changed:"
                f"\n  was:  model={prior_model}  dim={prior_dim}"
                f"\n  now:  model={resolved_model}  dim={resolved_dim}"
            )
            if meta.get("dim_changed"):
                print(
                    "  Dim changed — all _vec tables must be recreated.\n"
                    "  Run: kb reembed --force"
                )
            else:
                print("  Model changed (same dim) — run: kb reembed --force")

            if reembed:
                print("  Running reembed (--reembed flag set)...")
                _run_reembed(effective_db)
        else:
            print("  Embedding config matches existing index — no reembed needed.")

    print(f"\nGlobal configure done (provider={provider}, model={resolved_model}, dim={resolved_dim}, summary={resolved_summary})")
    return 0


def _run_reembed(db_path: Path) -> None:
    """Invoke kb reembed --force as a subprocess (avoids circular imports)."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "kb.py"),
         "--db", str(db_path), "reembed", "--force"],
    )
    if result.returncode != 0:
        print(f"Warning: reembed exited with code {result.returncode}", file=sys.stderr)


def run_project_configure(
    project_tag: str,
    enable_tracker: bool,
    db_override: str | None,
    key: str | None,
    project_dir: Path,
) -> int:
    """Write per-project config. Returns 0 on success, 1 on error."""
    project_dir = project_dir.resolve()

    # ------------------------------------------------------------------
    # A. .beads/config.yaml (backend: kb) — merge, don't clobber
    # ------------------------------------------------------------------
    if enable_tracker:
        beads_yaml = project_dir / ".beads" / "config.yaml"
        _merge_beads_config(beads_yaml, {"backend": "kb"})
        print(f"Written .beads/config.yaml backend:kb to {beads_yaml}")

    # ------------------------------------------------------------------
    # B. Per-project KB_DB to .claude/settings.json (merge)
    # ------------------------------------------------------------------
    project_settings = project_dir / ".claude" / "settings.json"
    env_updates: dict[str, str] = {}
    if db_override:
        env_updates["KB_DB"] = db_override
    if env_updates:
        _merge_env(project_settings, env_updates)
        print(f"Written per-project env to {project_settings}")

    # ------------------------------------------------------------------
    # C. Per-project secret (same guard)
    # ------------------------------------------------------------------
    if key:
        project_claude_dir = project_dir / ".claude"
        project_claude_dir.mkdir(parents=True, exist_ok=True)
        ok, msg = _write_secret_key(project_claude_dir, key)
        if not ok:
            print(f"Error: {msg}", file=sys.stderr)
            return 1
        print(msg)

    print(f"Project configure done (project={project_tag}, tracker={enable_tracker})")
    return 0


def _merge_beads_config(beads_yaml: Path, updates: dict[str, Any]) -> None:
    """Merge key-value updates into a YAML file (simple key: value pairs only)."""
    beads_yaml.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if beads_yaml.exists():
        try:
            import yaml  # type: ignore[import-untyped]
            existing = yaml.safe_load(beads_yaml.read_text()) or {}
        except (ImportError, Exception):
            # Minimal YAML parser: read "key: value" lines
            for line in beads_yaml.read_text().splitlines():
                if ":" in line and not line.strip().startswith("#"):
                    k, _, v = line.partition(":")
                    existing[k.strip()] = v.strip()

    existing.update(updates)

    try:
        import yaml  # type: ignore[import-untyped]
        beads_yaml.write_text(yaml.dump(existing, default_flow_style=False))
    except ImportError:
        # Write minimal YAML without pyyaml
        lines = [f"{k}: {v}" for k, v in existing.items()]
        beads_yaml.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point (called from kb.py main())
# ---------------------------------------------------------------------------


def _install_systemd_service(port: int = 8765) -> int:
    """Install + enable the kb-server systemd --user service. Generates the unit
    with paths resolved to THIS checkout's venv + kb.py (run-from-source), writes
    ~/.config/systemd/user/kb-server.service, daemon-reloads, and enables --now.
    A reference copy lives in the repo at deploy/kb-server.service."""
    import shutil

    kb_py = Path(__file__).resolve().parent.parent / "kb.py"
    venv_py = kb_py.parent / ".venv" / "bin" / "python"
    python = str(venv_py) if venv_py.exists() else sys.executable

    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / "kb-server.service"
    unit_path.write_text(
        "[Unit]\n"
        "Description=kb HTTP/SSE server (bridge transport + kb/issues read endpoints)\n"
        "After=network.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={python} {kb_py} serve --port {port} --host 127.0.0.1\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "StartLimitIntervalSec=0\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    print(f"Wrote {unit_path}")

    if not shutil.which("systemctl"):
        print("systemctl not found — unit written; enable manually:\n"
              "  systemctl --user daemon-reload && systemctl --user enable --now kb-server")
        return 0
    for cmd in (
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", "kb-server.service"],
    ):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ! {' '.join(cmd)} failed: {r.stderr.strip()[:200]}")
    active = subprocess.run(
        ["systemctl", "--user", "is-active", "kb-server.service"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"kb-server service: {active}  (KB_SERVER_URL=http://localhost:{port})")
    return 0


def configure_main(args: Any) -> int:  # noqa: ANN401
    """Dispatch from kb.py argparse namespace. Returns exit code."""
    from kb.constants import DEFAULT_DB_PATH

    if getattr(args, "install_server", False):
        return _install_systemd_service(getattr(args, "server_port", 8765))

    config_dir = getattr(args, "config_dir", None)
    if config_dir is None:
        config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
    config_dir = Path(config_dir)

    project_tag = getattr(args, "project", None)

    if project_tag:
        # Layer B: per-project configure
        project_dir = getattr(args, "project_dir", None) or Path.cwd()
        return run_project_configure(
            project_tag=project_tag,
            enable_tracker=getattr(args, "enable_tracker", False),
            db_override=getattr(args, "db_path_override", None),
            key=getattr(args, "key", None),
            project_dir=Path(project_dir),
        )

    # Layer A: global configure
    interactive = sys.stdout.isatty() and not getattr(args, "provider", None)
    return run_global_configure(
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        dim=getattr(args, "dim", None),
        fmt=getattr(args, "format", None),
        url=getattr(args, "url", None),
        summary_mode=getattr(args, "summary_mode", None),
        key=getattr(args, "key", None),
        reembed=getattr(args, "reembed", False),
        config_dir=config_dir,
        db_path=getattr(args, "db", None),
        interactive=interactive,
    )
