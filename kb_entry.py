"""Thin entry-point shim for console_scripts.

pyproject.toml [project.scripts] maps  kb = "kb_entry:main"
so that `pip install -e .` produces <venv>/bin/kb.
All real logic lives in kb.py; import and delegate.
"""

import runpy
from pathlib import Path


def main() -> None:
    # Execute kb.py as __main__ in the same process.
    # runpy.run_path honours the __name__ == "__main__" guard in kb.py.
    kb_py = Path(__file__).parent / "kb.py"
    runpy.run_path(str(kb_py), run_name="__main__")


if __name__ == "__main__":
    main()
