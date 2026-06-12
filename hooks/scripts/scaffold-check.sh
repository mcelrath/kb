#!/usr/bin/env bash
# scaffold-check.sh — SessionStart hook: detect missing project scaffold and inject
# additionalContext directing the main loop to run project-setup.
#
# "Run on install" mechanism: hooks cannot spawn agents directly, so detection +
# additionalContext injection is the only way to auto-launch project-setup.
#
# Detection criteria (ALL must be true to trigger):
#   1. Current dir is a git repo
#   2. reviewers.yaml is absent from the repo root
#   3. agent-preamble.md is absent from the repo root AND .claude/agents/preamble.md absent
#   4. .kb-setup-done marker is absent
#
# Marker location: ${CLAUDE_PLUGIN_DATA}/markers/ OR ~/.cache/kb/markers/
# (same DATA/cache convention as venv — never in PLUGIN_ROOT which churns on update,
#  never in the user's repo which would pollute their working tree).
#
# The marker is keyed by repo root hash so one install can track multiple repos.

set -euo pipefail

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

# --- Resolve marker directory (DATA if available, else fixed cache) ---
if [ -n "${CLAUDE_PLUGIN_DATA:-}" ]; then
    MARKER_DIR="${CLAUDE_PLUGIN_DATA}/markers"
else
    MARKER_DIR="${HOME}/.cache/kb/markers"
fi

# --- Must be inside a git repo ---
if ! git -C "${PWD}" rev-parse --show-toplevel >/dev/null 2>&1; then
    exit 0
fi

REPO_ROOT=$(git -C "${PWD}" rev-parse --show-toplevel)

# --- Compute a stable, short hash of the repo root path for the marker filename ---
REPO_HASH=$(printf '%s' "$REPO_ROOT" | sha256sum | cut -c1-12)
MARKER_FILE="${MARKER_DIR}/.kb-setup-done-${REPO_HASH}"

# --- Already set up: be silent ---
if [ -f "$MARKER_FILE" ]; then
    exit 0
fi

# --- Check scaffold presence ---
HAS_REVIEWERS=false
HAS_PREAMBLE=false

[ -f "${REPO_ROOT}/reviewers.yaml" ] && HAS_REVIEWERS=true
[ -f "${REPO_ROOT}/.github/reviewers.yaml" ] && HAS_REVIEWERS=true

[ -f "${REPO_ROOT}/agent-preamble.md" ]            && HAS_PREAMBLE=true
[ -f "${REPO_ROOT}/.claude/agents/preamble.md" ]   && HAS_PREAMBLE=true

# --- If scaffold already present: write the marker and be silent ---
if $HAS_REVIEWERS && $HAS_PREAMBLE; then
    mkdir -p "$MARKER_DIR"
    touch "$MARKER_FILE"
    exit 0
fi

# --- Scaffold missing: emit additionalContext to trigger project-setup ---
# SessionStart hook stdout becomes additionalContext injected before the first turn.
# This tells the main loop to run project-setup as its FIRST action.
# After detection we write the marker so a second run in the same session is silent
# (the agent will run once, create the scaffold, and future sessions see the marker).
mkdir -p "$MARKER_DIR"
touch "$MARKER_FILE"

cat <<EOF

=== KB PLUGIN: PROJECT SCAFFOLD MISSING ===

Repository: ${REPO_ROOT}
Missing:
$(${HAS_REVIEWERS} || echo "  - reviewers.yaml (reviewer personas)")
$(${HAS_PREAMBLE}  || echo "  - agent-preamble.md / .claude/agents/preamble.md")

ACTION REQUIRED (first action this session):
Run the project-setup agent to scaffold this repository:

  Task(subagent_type="project-setup", model="sonnet", run_in_background=True,
       prompt="Setup project at: ${REPO_ROOT}")

This creates reviewers.yaml, agent-preamble.md, configures kb embedding for the
project, and sets up the code-exploration stack (ast-grep, LSP, tree-sitter).
It is idempotent — safe to re-run if anything is missing.

Do this BEFORE any other work in this session.
===========================================
EOF

exit 0
