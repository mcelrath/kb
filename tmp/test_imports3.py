"""Show full tracebacks for annotation failures."""
import sys, types, traceback

sqlite_vec_stub = types.ModuleType('sqlite_vec')
sqlite_vec_stub.load = lambda: None
sys.modules['sqlite_vec'] = sqlite_vec_stub

try:
    import anthropic
except ImportError:
    sys.modules['anthropic'] = types.ModuleType('anthropic')

sys.path.insert(0, '/home/mcelrath/Projects/ai/kb')
import importlib

SHOW_FULL = {'kb.constants', 'kb.entities.base', 'kb.entities.scripts',
             'kb.code_lineage.structural', 'kb.hooks.check_symbols', 'kb.bd_import'}

for mod_name in SHOW_FULL:
    to_del = [k for k in sys.modules if k == mod_name or k.startswith(mod_name + '.')]
    for k in to_del:
        del sys.modules[k]
    try:
        importlib.import_module(mod_name)
        print(f"OK  {mod_name}")
    except Exception as e:
        print(f"\n=== {mod_name} ===")
        traceback.print_exc()
