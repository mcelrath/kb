"""Thin entry-point shim for console_scripts.

pyproject.toml [project.scripts] maps  kbt = "kbt_entry:main"
so that `pip install -e .` produces <venv>/bin/kbt (mirrors kb_entry.py).
All real logic lives in the top-level `kbt` script; run it as __main__.
"""

import runpy
from pathlib import Path


def main() -> None:
    kbt_script = Path(__file__).parent / "kbt"
    runpy.run_path(str(kbt_script), run_name="__main__")


if __name__ == "__main__":
    main()
