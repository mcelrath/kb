import ast
src = open('/home/mcelrath/Projects/ai/kb/kb.py').read()
print(f"File length: {len(src)}")
tree = ast.parse(src)
count = 0
for node in ast.walk(tree):
    count += 1
    if isinstance(node, ast.ClassDef):
        print(f"Class {node.lineno} {node.name}")
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ('__init__', 'embed_status'):
        print(f"Func {node.lineno} {node.name}")
print(f"Total nodes: {count}")
