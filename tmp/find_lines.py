import ast
src = open('/home/mcelrath/Projects/ai/kb/kb.py').read()
tree = ast.parse(src)
out = []
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        out.append(f"Class {node.lineno} {node.name}")
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ('__init__', 'embed_status'):
        out.append(f"Func {node.lineno} {node.name}")
for r in sorted(out):
    print(r)
