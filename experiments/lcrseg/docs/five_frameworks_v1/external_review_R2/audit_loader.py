"""Run narrow helper tests on hash-verified source snapshots, NOT full integration.

Only module-level relative imports are omitted. Function/class bodies are unchanged.
Native runners, DAG execution, source checkpoints, and real data are never used.
For the revised repository regression, leave SSLCL_R2_SNAPSHOT unset; tests then
import its actual production modules without this loader.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import sys
import types


def load_snapshot(root: Path) -> dict[str, types.ModuleType]:
    expected=json.loads((root/'expected_hashes.json').read_text())
    for name,sha in expected.items():
        got=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if got!=sha:raise ValueError(f'snapshot mismatch: {name}')
    modules={}
    for name in ('gate','integration','evaluate','numerics'):
        path=root/f'{name}.py'
        tree=ast.parse(path.read_text())
        # Dependencies for functions exercised below are real gate definitions.
        # Unused entrypoints needing parent/planner modules are never invoked.
        tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.level)]
        mod=types.ModuleType(f'external_review_r2_snapshot_{name}')
        sys.modules[mod.__name__]=mod
        if name=='integration':
            gate=modules['gate']
            mod.__dict__.update({k:getattr(gate,k) for k in
                ('ReviewRequired','require_approval','code_manifest','digest','CAPS')})
        exec(compile(tree,str(path),'exec'),mod.__dict__)
        modules[name]=mod
    return modules
