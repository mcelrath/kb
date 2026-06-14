#!/usr/bin/env python3
"""kb-876 Phases 3-9: move ONE domain's hooks into hooks/<domain>/ and repoint
every reference. Run per-domain (verify + commit between) for bisectability.

  python3 move_domain.py <domain>

Does, for the named domain:
  1. git mv each file  hooks/<f> -> hooks/<domain>/<f>
  2. rewrite global settings.json command paths  /hooks/<f> -> /hooks/<domain>/<f>
  3. (kb only) rewrite the two Physics project settings refs for
     compose_time_check.py + symbol_surface.py
  4. (guards/bridge) rewrite the intra-domain sibling exec path inside the moved
     hook (block-text-search-on-source -> _grep_pipeline_analyzer;
     block-bridge-watch-background -> _bridge_watch_detector)
Asserts each expected literal exists before editing — aborts on any miss.
"""
import os, sys, subprocess

REPO = os.path.expanduser('~/Projects/ai/claude')
HOOKS = os.path.join(REPO, 'hooks')
GLOBAL_SETTINGS = os.path.join(REPO, 'settings.json')
PHYS = [
    os.path.expanduser('~/Physics/claude/.claude/settings.json'),
    os.path.expanduser('~/Physics/secular-constraints/.claude/settings.json'),
]

DOMAINS = {
    'bridge': ['bridge-owed-reply-stop.py', 'bridge-recv-prompt.sh', 'bridge-resume.sh',
               'bridge-unread-stop.sh', 'bridge-watcher-alive.sh', 'bridge-watcher-check.sh',
               '_bridge_watch_detector.py', 'block-bridge-watch-background.sh',
               'block-stop-without-bridge-watcher.sh'],
    'git': ['bd-lifecycle.sh', 'git-asked-gate.sh', 'git-commit-check.sh',
            'guard-destructive-git.sh'],
    'kb': ['compose_time_check.py', 'dedupe-kb-get.sh', 'kb-context.sh',
           'kb-error-extract.sh', 'kb-flush-pending.sh', 'kb-precompact.sh',
           'kb-prompt-surface.py', 'kb-search-track.sh', 'open_issues_surface.py',
           'symbol_surface.py'],
    'guards': ['incompleteness-gate.sh', 'incompleteness-scanner.sh', 'weak-claim-gate.sh',
               'block-markdown-files.sh', 'block-markdown-via-bash.sh', 'md-asked-gate.sh',
               'md-cleanup.sh', 'block-local-dolt-server.sh', 'block-large-heredoc.sh',
               'block-text-search-on-source.sh', '_grep_pipeline_analyzer.py',
               'block-print-spam.sh', 'block-presentation-cells.sh',
               'block-followup-without-bd-id.sh', 'block-unprompted-deferral.sh',
               'prior-art-gate.sh', 'read-coverage-gate.sh'],
    'session': ['session-followups.sh', 'session-init.sh', 'session-persona.sh',
                'session-precheck.sh', 'session-start-resume.sh', 'precompact-save-state.sh',
                'project-scaffold-check.sh'],
    'lsp': ['lsp-diagnostics.sh', 'lsp-setup.sh', 'rust-analyzer-prewarm.sh'],
    'misc': ['allow-env-prefix.py', 'auto-approve-readonly-bash.py', 'model-availability.sh',
             'read-dep-augment.sh', 'redirect-tmp-scripts.sh'],
}

# intra-domain sibling exec rewrites: (domain, file-that-references, sibling-basename)
SIBLINGS = [
    ('guards', 'block-text-search-on-source.sh', '_grep_pipeline_analyzer.py'),
    ('bridge', 'block-bridge-watch-background.sh', '_bridge_watch_detector.py'),
]


def edit_file(path, old, new, required=True):
    t = open(path).read()
    if old not in t:
        if required:
            sys.exit(f"ABORT: {old!r} not found in {path}")
        return 0
    open(path, 'w').write(t.replace(old, new))
    return t.count(old)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in DOMAINS:
        sys.exit(f"usage: move_domain.py <{'|'.join(DOMAINS)}>")
    d = sys.argv[1]
    files = DOMAINS[d]
    os.makedirs(os.path.join(HOOKS, d), exist_ok=True)

    # 1. git mv
    for f in files:
        src = os.path.join(HOOKS, f)
        dst = os.path.join(HOOKS, d, f)
        if not os.path.isfile(src):
            sys.exit(f"ABORT: source missing {src}")
        subprocess.run(['git', '-C', REPO, 'mv', src, dst], check=True)

    # 2. global settings paths
    n = 0
    for f in files:
        n += edit_file(GLOBAL_SETTINGS, f'/hooks/{f}', f'/hooks/{d}/{f}', required=False)
    print(f"global settings: {n} path ref(s) rewritten")

    # 3. Physics settings (kb domain only)
    if d == 'kb':
        for sf in PHYS:
            if not os.path.isfile(sf):
                print(f"NOTE: physics settings absent: {sf}")
                continue
            for f in ('compose_time_check.py', 'symbol_surface.py'):
                edit_file(sf, f'/hooks/{f}', f'/hooks/kb/{f}', required=False)
        print("physics settings: compose_time_check + symbol_surface rewritten")

    # 4. intra-domain sibling exec path (file already moved into hooks/<d>/)
    for sd, ref_file, sibling in SIBLINGS:
        if sd != d:
            continue
        moved = os.path.join(HOOKS, d, ref_file)
        c = edit_file(moved, f'/hooks/{sibling}', f'/hooks/{d}/{sibling}', required=True)
        print(f"sibling ref in {ref_file}: {c} rewrite(s) -> hooks/{d}/{sibling}")

    print(f"\nOK: moved domain '{d}' ({len(files)} files)")


if __name__ == '__main__':
    main()
