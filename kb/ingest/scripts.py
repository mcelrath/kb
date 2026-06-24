#!/usr/bin/env python3
"""Auto-register scripts using local LLM to generate purpose descriptions."""

import json
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

# Allow standalone execution: ensure the package root is on sys.path.
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

LLM_URL = "http://tardis:9510/completion"

_VENV_PYTHON = _PKG_ROOT / ".venv" / "bin" / "python"


def _run_kb(args: list[str]):
    """Run `kb.py <args>` under the repo venv with the standard embedding env.
    Shared by the skip-registered listing and the per-script add (was duplicated)."""
    import os
    import subprocess
    return subprocess.run(
        [str(_VENV_PYTHON), "kb.py", *args],
        capture_output=True, text=True, cwd=_PKG_ROOT,
        env={**os.environ,
             "KB_EMBEDDING_URL": "http://ash:8081/embedding",
             "KB_EMBEDDING_DIM": "4096"},
    )


def generate_purpose(script_path: Path) -> str | None:
    """Use local LLM to generate a purpose description for a script."""
    content = script_path.read_text()
    snippet = content  # Full script - LLM has large context window

    prompt = f"""You are analyzing a scientific computation script. Given the filename and complete code, output a SINGLE LINE describing what hypothesis or computation this script tests/performs.

Rules:
1. Be specific about the mathematical/physical content
2. Use technical terms (e.g., "composition algebra", "gap equation", "signature obstruction")
3. Maximum 120 characters
4. NO quotes, NO explanations, just the purpose line

Examples:
File: dim4_complete.sage
Output: Complete 4D composition algebra classification: quaternions (4,0) and split-quaternions (2,2)

File: gap_equation_setup.sage
Output: Set up BCS gap equation integral in signature (2,2) spacetime

File: {script_path.name}
Code:
{snippet}

Output:"""

    req = Request(
        LLM_URL,
        data=json.dumps({
            "prompt": prompt,
            "n_predict": 150,
            "temperature": 0.3,
            "stop": ["\n", "\n\n"],
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            purpose = data.get("content", "").strip()
            purpose = purpose.replace('"', '').replace("'", "")
            if purpose and len(purpose) > 10:
                return purpose[:150]
    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  LLM error: {e}", file=sys.stderr)

    return None


def run(
    directory: Path,
    project: str = "hypercomplex",
    dry_run: bool = False,
    limit: int = 50,
    skip_registered: bool = False,
) -> int:
    """Auto-register scripts in-process.  Returns 0 on success."""
    directory = Path(directory)

    # Find scripts
    scripts = list(directory.rglob("*.py")) + list(directory.rglob("*.sage"))
    scripts = sorted(scripts)[:limit]

    print(f"Found {len(scripts)} scripts in {directory}")

    # Get already registered scripts if needed
    registered_hashes: set[str] = set()
    if skip_registered:
        result = _run_kb(["script", "list", "-n", "1000"])
        for line in result.stdout.split('\n'):
            if '.py:' in line or '.sage:' in line:
                parts = line.strip().split(':')
                if parts:
                    registered_hashes.add(parts[0].strip())

    for script_path in scripts:
        filename = script_path.name
        if skip_registered and filename in registered_hashes:
            print(f"  [skip] {filename} (already registered)")
            continue

        print(f"Processing: {script_path.name}...", end=" ", flush=True)
        purpose = generate_purpose(script_path)

        if purpose:
            print(f"OK")
            print(f"  Purpose: {purpose}")

            if not dry_run:
                result = _run_kb(["script", "add", str(script_path),
                                  "--purpose", purpose, "-p", project])
                if result.returncode == 0:
                    print(f"  Registered!")
                else:
                    print(f"  Error: {result.stderr[:100]}")
        else:
            print("FAILED (no purpose generated)")

    return 0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Auto-register scripts with LLM-generated purposes")
    parser.add_argument("directory", type=Path, help="Directory to scan for scripts")
    parser.add_argument("-p", "--project", default="hypercomplex", help="Project name")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be registered")
    parser.add_argument("-n", "--limit", type=int, default=50, help="Max scripts to process")
    parser.add_argument("--skip-registered", action="store_true", help="Skip already registered scripts")
    args = parser.parse_args()
    rc = run(
        directory=args.directory,
        project=args.project,
        dry_run=args.dry_run,
        limit=args.limit,
        skip_registered=args.skip_registered,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
