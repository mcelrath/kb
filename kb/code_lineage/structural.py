"""structural.py — normalized-AST structural hash for a single Python function.

Design goals (from PLAN-semantic-prior-art-surfacing.md, Tier A):
- Invariant to: function rename, local-var/param rename, file path, position-in-file,
  whitespace, comments, docstring.
- Sensitive to: called-function names, attribute names, literals, decorators,
  full signature INCLUDING return annotation, and all logic-bearing structure.

Normalization:
1. Parse the source fragment with `ast.parse`.
2. Strip the leading docstring node (if present) from the function body.
3. Alpha-rename the function's OWN name to the fixed placeholder `__fn__`.
4. Alpha-rename all local variables and parameters to positional placeholders
   `v0, v1, …` in first-occurrence order (DFS over the body).
   - "local" = names that appear in a Store context within this function scope,
     plus the parameter names.
   - Names that appear ONLY in Load context and were never stored/paramed are
     kept verbatim (they are references to outer/global/builtins — caller names,
     attribute access targets, etc.).
4a. Decorators and the return annotation are NOT subject to alpha-rename (they
    reference external names) and are included in the hash input verbatim.
5. Unparse the normalized AST to a canonical string and SHA-256-hash it.

Why keep decorators + return annotation:
  @lru_cache def f(x): return x   differs behaviourally from   def f(x): return x
  def f(x) -> int: return x       differs from                 def f(x) -> str: return x
  Omitting them collapses genuinely-different functions.
"""

import ast
import copy
import hashlib
import textwrap
from typing import Optional


# ---------------------------------------------------------------------------
# Internal AST transformers
# ---------------------------------------------------------------------------


class _LocalCollector(ast.NodeVisitor):
    """Collect names that are assigned/defined inside the function scope
    (parameters + Store-context names + comprehension targets etc.).
    Excludes names that are only ever loaded (external references).
    """

    def __init__(self) -> None:
        self.locals: list[str] = []  # ordered by first encounter
        self._seen: set[str] = set()

    def _add(self, name: str) -> None:
        if name not in self._seen:
            self._seen.add(name)
            self.locals.append(name)

    # Function arguments
    def visit_arguments(self, node: ast.arguments) -> None:
        for arg in (
            node.posonlyargs
            + node.args
            + ([node.vararg] if node.vararg else [])
            + node.kwonlyargs
            + ([node.kwarg] if node.kwarg else [])
        ):
            self._add(arg.arg)
        self.generic_visit(node)

    # Assignment targets (Store context)
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._add(node.id)
        self.generic_visit(node)

    # For-loop targets
    def visit_For(self, node: ast.For) -> None:
        self._collect_target(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._collect_target(node.target)
        self.generic_visit(node)

    # With-statement targets
    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars:
                self._collect_target(item.optional_vars)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars:
                self._collect_target(item.optional_vars)
        self.generic_visit(node)

    # Exception handler targets
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._add(node.name)
        self.generic_visit(node)

    # Comprehension targets (list/set/dict/gen)
    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._collect_target(node.target)
        self.generic_visit(node)

    # walrus operator
    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._add(node.target.id)
        self.generic_visit(node)

    def _collect_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self._add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_target(elt)
        elif isinstance(target, ast.Starred):
            self._collect_target(target.value)


class _Normalizer(ast.NodeTransformer):
    """Replace the function's own name with __fn__ and local names with
    positional placeholders v0, v1, … in first-occurrence order.
    """

    def __init__(self, fn_name: str, local_names: list[str]) -> None:
        self._fn_name = fn_name
        # map: original name → placeholder
        self._remap: dict[str, str] = {
            name: f"v{i}" for i, name in enumerate(local_names)
        }

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Rename the function itself, but do NOT descend into nested functions
        # for their own names — only their bodies are visited.
        node.name = "__fn__"
        self.generic_visit(node)
        return node

    # AsyncFunctionDef has the same structure
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = "__fn__"
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self._remap:
            node.id = self._remap[node.id]
        elif node.id == self._fn_name:
            # Recursive self-call
            node.id = "__fn__"
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self._remap:
            node.arg = self._remap[node.arg]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        if node.name and node.name in self._remap:
            node.name = self._remap[node.name]
        self.generic_visit(node)
        return node


def _strip_docstring(func_node: ast.FunctionDef) -> None:
    """Remove the leading docstring expression from the function body in-place."""
    if (
        func_node.body
        and isinstance(func_node.body[0], ast.Expr)
        and isinstance(func_node.body[0].value, ast.Constant)
    ):
        func_node.body = func_node.body[1:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def structural_hash(source: str) -> str:
    """Return a hex SHA-256 hash of the normalized AST of *source*.

    *source* must be valid Python containing exactly one top-level function
    definition (with any leading indentation dedented automatically).

    Raises ValueError if parsing fails or no function definition is found.
    """
    source = textwrap.dedent(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Failed to parse source: {exc}") from exc

    # Find the top-level (Module-level) function definition.
    # ast.walk gives no ordering guarantees; scan the Module.body directly.
    func_node = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_node = node
            break
    if func_node is None:
        raise ValueError("No function definition found in source")

    # 1. Strip docstring
    _strip_docstring(func_node)

    # 2. Collect locals (params + assigned names) from the body only,
    #    NOT from decorators or the return annotation.
    collector = _LocalCollector()
    # Visit arguments first (params)
    collector.visit(func_node.args)
    # Then the body
    for stmt in func_node.body:
        collector.visit(stmt)

    # 3. Normalize
    normalizer = _Normalizer(func_node.name, collector.locals)
    normalized_tree = normalizer.visit(copy.deepcopy(tree))
    ast.fix_missing_locations(normalized_tree)

    # 4. Unparse to canonical text
    canonical = ast.unparse(normalized_tree)

    # 5. SHA-256
    return hashlib.sha256(canonical.encode()).hexdigest()


def parse_function_info(source: str) -> dict:
    """Extract name, qualname-stub, and line number from a function source fragment.

    Returns a dict with keys: name, line (1-based in the fragment).
    """
    source = textwrap.dedent(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Failed to parse source: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return {"name": node.name, "line": node.lineno}

    raise ValueError("No function definition found")
