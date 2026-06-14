"""
Scan kb/ package files (excluding kb/server/) for annotation-shadowing bugs:
Any function/method where a parameter, local variable, or attribute is named
after a Python builtin AND that same name appears as a type annotation.
"""
import ast
import builtins
from pathlib import Path

BUILTIN_NAMES = set(vars(builtins).keys())

def collect_annotation_names(annotation):
    """Get all Name nodes used in an annotation."""
    if annotation is None:
        return set()
    names = set()
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names

def check_file(path):
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"  SYNTAX ERROR: {e}")
        return []

    has_future_annotations = any(
        isinstance(node, ast.ImportFrom) and node.module == '__future__' and
        any(a.name == 'annotations' for a in node.names)
        for node in ast.walk(tree)
    )

    if has_future_annotations:
        return []  # Already fixed

    issues = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Collect all parameter names
        args = node.args
        all_params = args.posonlyargs + args.args + args.kwonlyargs
        if args.vararg:
            all_params = all_params + [args.vararg]
        if args.kwarg:
            all_params = all_params + [args.kwarg]
        param_names = {arg.arg for arg in all_params}

        # Collect all local variable names assigned in this function
        local_names = set()
        for stmt in ast.walk(node):
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt is not node:
                continue  # don't descend into nested functions
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        local_names.add(target.id)
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name):
                    local_names.add(stmt.target.id)
            elif isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name):
                    local_names.add(stmt.target.id)

        shadowed = (param_names | local_names) & BUILTIN_NAMES

        # Now check every annotation in the function for use of these shadowed names
        for arg in all_params:
            if arg.annotation:
                ann_names = collect_annotation_names(arg.annotation)
                hits = ann_names & BUILTIN_NAMES  # annotation uses a builtin name
                # The real bug: does the function body also assign a local with that name?
                # OR: does a param shadow a builtin used in its own annotation?
                # Check if any param name == an annotation name (e.g. param named 'list', annotation 'list')
                for hit in hits:
                    if hit in shadowed:
                        issues.append((arg.col_offset, node.lineno, arg.arg,
                            f"param '{arg.arg}': annotation uses '{hit}' which is shadowed by local/param '{hit}' in {node.name}()"))

        # Check return annotation
        if node.returns:
            ann_names = collect_annotation_names(node.returns)
            for hit in ann_names & BUILTIN_NAMES:
                if hit in shadowed:
                    issues.append((node.lineno, node.lineno, 'return',
                        f"return annotation uses '{hit}' shadowed by local '{hit}' in {node.name}()"))

    return issues


base = Path("/home/mcelrath/Projects/ai/kb/kb")
for py_file in sorted(base.rglob("*.py")):
    if "server" in py_file.parts:
        continue
    src = py_file.read_text()
    has_future = "from __future__ import annotations" in src
    issues = check_file(py_file)
    print(f"{'[HAS_FUTURE]' if has_future else '[NO_FUTURE ]'} {py_file}")
    for issue in issues:
        print(f"  line {issue[1]}: {issue[3]}")
