"""Scientific admission receipts; a successful process exit is not qualification."""
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_fingerprint():
    root = Path(__file__).parent
    return fingerprint({p.name: file_fingerprint(p) for p in sorted(root.glob('*.py')) if not p.name.startswith('.')})


QUALIFIED_STATES = {
    'UNIT': 'UNIT_PASS',
    'REVISION': 'REVISION_PASS',
    'SYNTHETIC_LEARNING': 'SYNTHETIC_LEARNING_QUALIFIED',
    'MODEL_LEARNING': 'MODEL_LEARNING_QUALIFIED',
}


def write_receipt(path, kind, config, report, evidence_fingerprint, checkpoint=None, code_version=None):
    """Reports are relative immutable artifacts; oracle diagnostics are never learning receipts."""
    path, report = Path(path), Path(report)
    if kind not in QUALIFIED_STATES or not report.resolve().is_relative_to(path.parent.resolve()):
        raise ValueError('invalid receipt kind or evidence location')
    result = json.loads(report.read_text())
    qualified = result.get('scientific_state') == QUALIFIED_STATES[kind]
    if result.get('oracle', False) and kind not in ('UNIT', 'REVISION'):
        qualified = False
    receipt = dict(schema=1, kind=kind, status='PASS' if qualified else 'FAIL',
                   scientific_state=result.get('scientific_state'), code_version=code_version,
                   code_fingerprint=code_fingerprint(), config_fingerprint=fingerprint(config.to_dict()),
                   evidence_fingerprint=evidence_fingerprint,
                   report=report.resolve().relative_to(path.parent.resolve()).as_posix(),
                   report_fingerprint=file_fingerprint(report),
                   checkpoint_fingerprint=None if checkpoint is None else file_fingerprint(checkpoint))
    with path.open('x') as f:
        json.dump(receipt, f, indent=2)
    return receipt


def require_receipt(path, kind, config, evidence_fingerprint, checkpoint=None):
    if not path or not Path(path).is_file():
        raise PermissionError('scientific qualification receipt missing')
    path = Path(path)
    receipt = json.loads(path.read_text())
    expected = dict(schema=1, kind=kind, status='PASS', scientific_state=QUALIFIED_STATES[kind],
                    code_fingerprint=code_fingerprint(), config_fingerprint=fingerprint(config.to_dict()),
                    evidence_fingerprint=evidence_fingerprint)
    if not evidence_fingerprint or any(receipt.get(k) != v for k, v in expected.items()):
        raise PermissionError('failed, stale or mismatched scientific qualification')
    relative = Path(receipt['report'])
    report = (path.parent / relative).resolve()
    if relative.is_absolute() or not report.is_relative_to(path.parent.resolve()):
        raise PermissionError('qualification evidence escapes receipt directory')
    if file_fingerprint(report) != receipt['report_fingerprint']:
        raise PermissionError('qualification evidence changed')
    result = json.loads(report.read_text())
    if result.get('scientific_state') != QUALIFIED_STATES[kind] or (kind not in ('UNIT', 'REVISION') and result.get('oracle', False)):
        raise PermissionError('report is not the required scientific qualification')
    if kind == 'UNIT' and ({r['id'] for r in result.get('results', [])} != {f'T{i:02d}' for i in range(1, 19)} or
                           any(r['status'] != 'PASS' for r in result['results']) or
                           fingerprint(result['results']) != evidence_fingerprint):
        raise PermissionError('full unit evidence missing or inconsistent')
    if kind == 'REVISION' and ({r['id'] for r in result.get('results', [])} != {f'R{i:02d}' for i in range(1, 10)} or
                               any(r['status'] != 'PASS' for r in result['results']) or
                               fingerprint(result['results']) != evidence_fingerprint):
        raise PermissionError('full revision evidence missing or inconsistent')
    actual = None if checkpoint is None else file_fingerprint(checkpoint)
    if receipt.get('checkpoint_fingerprint') != actual or (kind == 'MODEL_LEARNING' and actual is None):
        raise PermissionError('qualification checkpoint mismatch')
    return fingerprint(receipt)
