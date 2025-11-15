import importlib
import sys
import os

# Ensure senga-sde root is on sys.path so imports like 'backend...' work
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

modules = [
    "backend.core.powell.engine",
    "backend.core.powell.dla",
    "backend.core.powell.cfa",
    "backend.core.powell.vfa",
    "backend.services.state_manager",
    "backend.services.event_orchestrator",
    "backend.services.learning_coordinator",
]

ok = True
for m in modules:
    try:
        importlib.import_module(m)
        print(f"IMPORT_OK: {m}")
    except Exception as e:
        print(f"IMPORT_FAIL: {m} -> {e}")
        ok = False

if not ok:
    sys.exit(2)

print("ALL_IMPORTS_OK")
