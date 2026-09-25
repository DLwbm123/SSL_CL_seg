#!/usr/bin/env python3
"""Build exact rsync lists, verify bytes, and promote files without overwriting.

No SSH, model import, HDF5 payload parsing, network request or training occurs here.
Private JSON outputs contain source paths and must NOT be published.
Python 3.10+; standard library only. Designed for a trusted, single-owner directory.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

DOMAINS = ('REFUGE', 'RIM_ONE_r3', 'Drishti_GS')
COUNTS = {'REFUGE': (40, 160), 'RIM_ONE_r3': (16, 63), 'Drishti_GS': (10, 41)}
MANIFEST = 'manifests/training/lcrseg_v1_seed0.csv'
SPLIT = 'splits/fundus_seed0.json'
ROOTS = ('code', 'data', 'assets')
SOURCE_ROOTS = ROOTS + ('metadata',)
DENIED = {'.git', '.ssh', '.aws', '.azure', '.venv', 'venv', 'node_modules', '__pycache__', 'wandb', '.cache'}
CODE_SUFFIXES = {'.py', '.json', '.yaml', '.yml', '.md', '.txt', '.toml', '.cfg', '.ini', '.sh', '.lock', '.csv'}

class Refusal(ValueError):
    pass

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def relpath(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or '\x00' in value or '\\' in value:
        raise Refusal(f'invalid relative path: {value!r}')
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ('', '.', '..') for x in value.split('/')):
        raise Refusal(f'absolute/traversal/directory path forbidden: {value!r}')
    if any(x in DENIED or x.startswith('.env') or x.startswith('id_rsa') or x.startswith('id_ed25519') for x in p.parts):
        raise Refusal(f'sensitive/cache path forbidden: {value!r}')
    if any(x in value for x in '*?[]\n\r'):
        raise Refusal('wildcards/newlines forbidden')
    return p

def canonical_root(value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        raise Refusal('roots must be confirmed absolute paths')
    p = p.resolve(strict=True)
    if not p.is_dir():
        raise Refusal('root is not a directory')
    return p

def safe_file(root: Path, relative: str) -> Path:
    p = root
    for part in relpath(relative).parts:
        p = p / part
        if p.is_symlink():
            raise Refusal(f'symlink is not a permitted transfer file: {relative}')
    if not p.is_file() or not stat.S_ISREG(p.stat().st_mode):
        raise Refusal(f'missing or non-regular file: {relative}')
    if not p.resolve().is_relative_to(root.resolve()):
        raise Refusal('file escaped root')
    return p

def check_digest(text: str) -> None:
    if not isinstance(text, str) or not re.fullmatch(r'[0-9a-f]{64}', text):
        raise Refusal('expected a full lowercase SHA256')

def write_json_new(path: Path, payload: dict) -> None:
    # Create-only; no previous receipt or manifest is replaced.
    with path.open('x', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    path.chmod(0o600)

def build(config_path: Path, out: Path) -> dict:
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    if cfg.get('study_id') != 'SSLCL_QPROMPT_RL_V1' or cfg.get('phase') not in ('M1', 'M2'):
        raise Refusal('wrong study or unrecognized migration phase')
    roots = {k: canonical_root(cfg['roots'][k]) for k in ROOTS}
    metadata_id = 'data'
    if cfg.get('metadata_root'):
        roots['metadata'] = canonical_root(cfg['metadata_root'])
        metadata_id = 'metadata'
    commit = cfg['code_commit']
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise Refusal('bind the actual 40-character code commit first')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=roots['code'], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=roots['code'], text=True)
    if head != commit or dirty:
        raise Refusal('code commit mismatch or dirty working tree; freeze code before migration')
    for relative, key in ((MANIFEST, 'manifest_sha256'), (SPLIT, 'split_sha256')):
        check_digest(cfg[key])
        if sha256(safe_file(roots[metadata_id], relative)) != cfg[key]:
            raise Refusal(f'frozen metadata hash mismatch: {relative}')
    with safe_file(roots[metadata_id], MANIFEST).open(newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r.get('dataset') == 'fundus']
    split = json.loads(safe_file(roots[metadata_id], SPLIT).read_text())
    if split.get('seed') != 0:
        raise Refusal('only frozen seed0 split permitted')
    sr = {r['case_id']: r for r in split['records']}
    if len(sr) != len(split['records']):
        raise Refusal('duplicate case in split')
    seen, patients, counts = set(), {}, Counter()
    entries = {}

    def add(root_id: str, relative: str, reason: str, expected: str | None = None,
            role: str = '', domain: str = '') -> None:
        if not reason or not reason.strip():
            raise Refusal('every transferred file requires a reason')
        file = safe_file(roots[root_id], relative)
        before = file.stat()
        actual = sha256(file)
        after = file.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise Refusal('source changed while hashing')
        if expected is not None:
            check_digest(expected)
            if actual != expected:
                raise Refusal(f'payload hash mismatch: {root_id}/{relative}')
        key = ('data' if root_id == 'metadata' else root_id) + '/' + relative
        item = dict(root_id=root_id, source_relative=relative, target_relative=key,
                    bytes=after.st_size, sha256=actual, reason=reason, role=role, domain=domain)
        if key in entries and entries[key]['sha256'] != actual:
            raise Refusal('conflicting duplicate destination')
        if key not in entries:
            entries[key] = item

    files = cfg.get('code_files', [])
    if not files:
        raise Refusal('empty code allowlist')
    for item in files:
        p = relpath(item['path'])
        if p.suffix.lower() not in CODE_SUFFIXES and not p.name.startswith(('LICENSE', 'NOTICE', 'COPYING', 'README')):
            raise Refusal(f'not an allowed code/text artifact: {p}')
        add('code', item['path'], item['reason'], item.get('sha256'))
    n_pretrained = 0
    for item in cfg.get('asset_files', []):
        if item['kind'] not in ('paper', 'pretrained', 'third_party_source', 'license', 'dependency_lock'):
            raise Refusal('old checkpoints/caches are not default migration assets')
        if item['kind'] == 'pretrained':
            n_pretrained += 1
        if 'sha256' not in item:
            raise Refusal('pin every external asset SHA256 before transfer')
        add('assets', item['path'], item['reason'], item['sha256'], role=item['kind'])
    if n_pretrained > 1:
        raise Refusal('initial scope permits only one selected pretrained weight file')
    add(metadata_id, MANIFEST, 'Frozen canonical manifest; metadata only', cfg['manifest_sha256'], 'metadata')
    add(metadata_id, SPLIT, 'Frozen split; metadata only', cfg['split_sha256'], 'metadata')
    selected_records = Counter()
    for r in rows:
        cid, dom, role = r['case_id'], r['site_or_vendor'], r['primary_20pct_split']
        if cid in seen or dom not in DOMAINS or r['split_seed'] != '0':
            raise Refusal('duplicate case, unknown domain or wrong split seed')
        if role not in ('train_labeled', 'train_unlabeled', 'val', 'test'):
            raise Refusal('unrecognized role')
        seen.add(cid)
        counts[(dom, role)] += 1
        for field in ('patient_id', 'site_or_vendor', 'primary_20pct_split'):
            if cid not in sr or r[field] != sr[cid][field]:
                raise Refusal('manifest/split identity mismatch')
        pid = r['patient_id']
        if not pid or patients.setdefault(pid, (dom, role)) != (dom, role):
            raise Refusal('empty patient id or patient crosses domain/role')
        if role == 'train_unlabeled' and (r.get('label_h5_relpath') or r.get('label_sha256')):
            raise Refusal('U ground-truth metadata must be absent')
        selected = (dom != 'REFUGE' and role in ('train_labeled', 'val'))
        if cfg['phase'] == 'M2':
            selected = role in ('train_labeled', 'val') or (dom != 'REFUGE' and role == 'train_unlabeled')
        if not selected:
            continue
        selected_records[(dom, role)] += 1
        keys = ('image',) if role == 'train_unlabeled' else ('image', 'label')
        for key in keys:
            add('data', r[key + '_h5_relpath'], f'{cfg["phase"]}: {dom}/{role}/{key}',
                r[key + '_sha256'], role, dom)
    for dom, (nl, nu) in COUNTS.items():
        if counts[(dom, 'train_labeled')] != nl or counts[(dom, 'train_unlabeled')] != nu:
            raise Refusal('canonical L/U record counts do not match frozen protocol')
    for dom in (DOMAINS[1:] if cfg['phase'] == 'M1' else DOMAINS):
        if selected_records[(dom, 'val')] == 0:
            raise Refusal('selected domain has no validation records')
    ordered = [entries[x] for x in sorted(entries)]
    payload = dict(schema=1, study_id=cfg['study_id'], phase=cfg['phase'], status='PLANNED_BYTES_ONLY',
                   code_commit=commit, roots={k: str(v) for k, v in roots.items()},
                   manifest_sha256=cfg['manifest_sha256'], split_sha256=cfg['split_sha256'],
                   records={f'{a}/{b}': n for (a, b), n in sorted(selected_records.items())},
                   files=ordered, file_count=len(ordered), total_bytes=sum(x['bytes'] for x in ordered),
                   excluded=['test payload', 'all U labels', 'REFUGE U', 'old checkpoints', 'old environments', 'secrets'])
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    for root_id in roots:
        with (out / (root_id + '.files0')).open('xb') as f:
            for item in ordered:
                if item['root_id'] == root_id:
                    f.write(item['source_relative'].encode('utf-8') + b'\0')
    manifest_path = out / 'MIGRATION_MANIFEST.private.json'
    write_json_new(manifest_path, payload)
    (out / 'MANIFEST.sha256').write_text(sha256(manifest_path) + '\n', encoding='ascii')
    return payload

def load_manifest(path: Path, expected: str | None = None) -> dict:
    if expected:
        check_digest(expected)
        if sha256(path) != expected:
            raise Refusal('migration manifest digest mismatch')
    m = json.loads(path.read_text())
    if m.get('schema') != 1 or m.get('study_id') != 'SSLCL_QPROMPT_RL_V1':
        raise Refusal('unknown manifest schema/study')
    destinations = set()
    for x in m['files']:
        if x['root_id'] not in SOURCE_ROOTS or x['target_relative'] != ('data' if x['root_id'] == 'metadata' else x['root_id']) + '/' + x['source_relative']:
            raise Refusal('destination must remain under its allowlisted root')
        relpath(x['target_relative']); check_digest(x['sha256'])
        if x['target_relative'] in destinations:
            raise Refusal('duplicate target')
        destinations.add(x['target_relative'])
        if not isinstance(x['bytes'], int) or x['bytes'] < 0:
            raise Refusal('invalid byte count')
    if len(destinations) != m['file_count'] or sum(x['bytes'] for x in m['files']) != m['total_bytes']:
        raise Refusal('manifest aggregate mismatch')
    return m

def verify(m: dict, destination: Path, strict: bool = True) -> dict:
    destination = canonical_root(str(destination))
    expected = {x['target_relative'] for x in m['files']}
    for item in m['files']:
        p = safe_file(destination, item['target_relative'])
        if p.stat().st_size != item['bytes'] or sha256(p) != item['sha256']:
            raise Refusal(f'target bytes differ: {item["target_relative"]}')
    if strict:
        for root_id in ROOTS:
            root = destination / root_id
            if root.is_symlink():
                raise Refusal('symlink destination root')
            if root.exists():
                for p in root.rglob('*'):
                    if p.is_symlink():
                        raise Refusal('symlink in destination')
                    if p.is_file() and p.relative_to(destination).as_posix() not in expected:
                        raise Refusal('extra file outside allowlist: ' + p.relative_to(destination).as_posix())
    return dict(status='VERIFIED_BYTES_ONLY', file_count=m['file_count'], total_bytes=m['total_bytes'],
                strict=strict, hdf5_semantics='NOT_CHECKED', model_execution='NOT_RUN')

def promote(m: dict, staging: Path, destination: Path) -> dict:
    verify(m, staging, strict=True)
    destination = canonical_root(str(destination))
    # Preflight the whole destination before committing any file. No deletions.
    for item in m['files']:
        p = destination
        for part in relpath(item['target_relative']).parts:
            p = p / part
            if p.is_symlink():
                raise Refusal('symlink in final destination')
        if p.exists() and (not p.is_file() or p.stat().st_size != item['bytes'] or sha256(p) != item['sha256']):
            raise Refusal('existing different destination: ' + item['target_relative'])
    copied, reused = 0, 0
    for item in m['files']:
        src = safe_file(staging, item['target_relative'])
        dst = destination / item['target_relative']
        if dst.exists():
            reused += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(prefix='.incoming-file-', dir=dst.parent)
        try:
            with os.fdopen(fd, 'wb') as out, src.open('rb') as inp:
                shutil.copyfileobj(inp, out, 1024 * 1024)
                out.flush(); os.fsync(out.fileno())
            if sha256(Path(tmp)) != item['sha256']:
                raise Refusal('local copy changed')
            # Hard-link commits a newly created temp file; fails rather than overwrites.
            os.link(tmp, dst)
            copied += 1
        finally:
            Path(tmp).unlink(missing_ok=True)
    verify(m, destination, strict=False)  # M2 is a cumulative extension; unrelated run outputs untouched.
    return dict(status='PROMOTED_VERIFIED_BYTES', copied=copied, reused=reused, source_deleted=False)

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    b = subs.add_parser('build'); b.add_argument('--config', type=Path, required=True); b.add_argument('--out', type=Path, required=True)
    for name in ('verify', 'promote'):
        q = subs.add_parser(name); q.add_argument('--manifest', type=Path, required=True)
        q.add_argument('--expected-digest', required=True); q.add_argument('--destination', type=Path, required=True)
        if name == 'promote':
            q.add_argument('--staging', type=Path, required=True)
        else:
            q.add_argument('--allow-extra', action='store_true', help='only for a previously validated cumulative target')
    args = p.parse_args()
    try:
        if args.command == 'build':
            result = build(args.config, args.out)
            result = {k: result[k] for k in ('status', 'phase', 'file_count', 'total_bytes', 'records')}
        else:
            manifest = load_manifest(args.manifest, args.expected_digest)
            result = verify(manifest, args.destination, not args.allow_extra) if args.command == 'verify' else promote(manifest, args.staging, args.destination)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (Refusal, OSError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
