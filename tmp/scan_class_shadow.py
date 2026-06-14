"""
Find classes where a method/attribute name shadows a builtin,
and that builtin name also appears in an annotation within the same class body.
This causes TypeError on 3.11-3.13 under eager annotation evaluation.
"""
import ast
import builtins
from pathlib import Path

BUILTIN_NAMES = set(vars(builtins).keys())

def get_annotation_names(node):
    """All Name ids referenced in an annotation node."""
    names = set()
    if node is None:
        return names
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)
    return names

def check_file(path):
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [], False

    has_future = any(
        isinstance(n, ast.ImportFrom) and n.module == '__future__'
        and any(a.name == 'annotations' for a in n.names)
        for n in ast.walk(tree)
    )

    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Collect all names defined at class body level (methods + class vars)
        class_level_names = set()
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                class_level_names.add(item.name)
            elif isinstance(item, ast.Assign):
                for t in item.targets:
                    if isinstance(t, ast.Name):
                        class_level_names.add(t.id)
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    class_level_names.add(item.target.id)

        shadowed_builtins = class_level_names & BUILTIN_NAMES

        if not shadowed_builtins:
            continue

        # Now find all annotations in the class body that reference shadowed names
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check parameter annotations and return annotation
                all_args = (item.args.posonlyargs + item.args.args +
                           item.args.kwonlyargs)
                if item.args.vararg:
                    all_args = all_args + [item.args.vararg]
                if item.args.kwarg:
                    all_args = all_args + [item.args.kwarg]
                for arg in all_args:
                    if arg.annotation:
                        ann_names = get_annotation_names(arg.annotation)
                        hits = ann_names & shadowed_builtins
                        for h in hits:
                            issues.append((item.lineno, node.name, item.name,
                                f"param annotation uses '{h}' which is shadowed by class-level def '{h}'"))
                if item.returns:
                    ann_names = get_annotation_names(item.returns)
                    hits = ann_names & shadowed_builtins
                    for h in hits:
                        issues.append((item.lineno, node.name, item.name,
                            f"return annotation uses '{h}' which is shadowed by class-level def '{h}'"))
            elif isinstance(item, ast.AnnAssign):
                if item.annotation:
                    ann_names = get_annotation_names(item.annotation)
                    hits = ann_names & shadowed_builtins
                    for h in hits:
                        tname = item.target.id if isinstance(item.target, ast.Name) else '?'
                        issues.append((item.lineno, node.name, tname,
                            f"class attribute annotation uses '{h}' shadowed by class-level def '{h}'"))

    return issues, has_future


base = Path("/home/mcelrath/Projects/ai/kb/kb")
for py_file in sorted(base.rglob("*.py")):
    if "server" in py_file.parts:
        continue
    issues, has_future = check_file(py_file)
    rel = py_file.relative_to("/home/mcelrath/Projects/ai/kb")
    if issues:
        status = "HAS_FUTURE" if has_future else "NEEDS_FIX"
        print(f"\n[{status}] {rel}")
        for lineno, cls, method, msg in issues:
            print(f"  line {lineno} in {cls}.{method}: {msg}")
    elif not has_future:
        print(f"[clean    ] {rel}")
    else:
        print(f"[has_future] {rel}")
