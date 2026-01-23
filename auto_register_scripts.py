#!/usr/bin/env python3
"""Auto-register scripts using local LLM to generate purpose descriptions."""

import json
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

LLM_URL = "http://tardis:9510/completion"

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
        with urlopen(req, timeout=120) as resp:  # Longer timeout for full scripts
            data = json.loads(resp.read().decode("utf-8"))
            purpose = data.get("content", "").strip()
            # Clean up
            purpose = purpose.replace('"', '').replace("'", "")
            if purpose and len(purpose) > 10:
                return purpose[:150]  # Truncate if too long
    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  LLM error: {e}", file=sys.stderr)

    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-register scripts with LLM-generated purposes")
    parser.add_argument("directory", type=Path, help="Directory to scan for scripts")
    parser.add_argument("-p", "--project", default="hypercomplex", help="Project name")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be registered")
    parser.add_argument("-n", "--limit", type=int, default=50, help="Max scripts to process")
    parser.add_argument("--skip-registered", action="store_true", help="Skip already registered scripts")
    args = parser.parse_args()

    # Find scripts
    scripts = list(args.directory.rglob("*.py")) + list(args.directory.rglob("*.sage"))
    scripts = sorted(scripts)[:args.limit]

    print(f"Found {len(scripts)} scripts in {args.directory}")

    # Get already registered scripts if needed
    registered_hashes = set()
    if args.skip_registered:
        import subprocess
        import os
        venv_python = Path(__file__).parent / ".venv" / "bin" / "python"
        result = subprocess.run(
            [str(venv_python), "kb.py", "script", "list", "-n", "1000"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
            env={**os.environ,
                 "KB_EMBEDDING_URL": "http://ash:8080/embedding",
                 "KB_EMBEDDING_DIM": "4096"}
        )
        # Parse output to get filenames (crude but works)
        for line in result.stdout.split('\n'):
            if '.py:' in line or '.sage:' in line:
                parts = line.strip().split(':')
                if parts:
                    registered_hashes.add(parts[0].strip())

    for script_path in scripts:
        filename = script_path.name
        if args.skip_registered and filename in registered_hashes:
            print(f"  [skip] {filename} (already registered)")
            continue

        print(f"Processing: {script_path.name}...", end=" ", flush=True)
        purpose = generate_purpose(script_path)

        if purpose:
            print(f"OK")
            print(f"  Purpose: {purpose}")

            if not args.dry_run:
                import subprocess
                import os
                venv_python = Path(__file__).parent / ".venv" / "bin" / "python"
                result = subprocess.run(
                    [str(venv_python), "kb.py", "script", "add", str(script_path),
                     "--purpose", purpose, "-p", args.project],
                    capture_output=True, text=True, cwd=Path(__file__).parent,
                    env={**os.environ,
                         "KB_EMBEDDING_URL": "http://ash:8080/embedding",
                         "KB_EMBEDDING_DIM": "4096"}
                )
                if result.returncode == 0:
                    print(f"  Registered!")
                else:
                    print(f"  Error: {result.stderr[:100]}")
        else:
            print("FAILED (no purpose generated)")


if __name__ == "__main__":
    main()
