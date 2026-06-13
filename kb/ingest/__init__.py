"""kb.ingest — importable entry points for all ingest scripts.

Each sub-module exposes a ``run(...)`` function that performs the ingest
in-process (no subprocess).  A thin ``main()`` with argparse is also kept so
the modules remain directly executable for standalone / hook use.
"""
