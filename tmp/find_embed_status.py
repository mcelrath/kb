import ast
src = open('/home/mcelrath/Projects/ai/kb/kb.py').read()
lines = src.split('\n')
for i, line in enumerate(lines, 1):
    if 'embed' in line.lower() and ('status' in line.lower() or 'meta' in line.lower()):
        print(f"{i}: {line}")
