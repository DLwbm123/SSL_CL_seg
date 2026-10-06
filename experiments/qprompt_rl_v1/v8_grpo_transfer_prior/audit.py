"""Metadata-only V8 admission audit. Never treats case aliases as patient evidence.

Call audit(manifest, split, builder_source); no image/label/checkpoint payload reads.
The caller supplies the already designated paths. Returned fields are anonymous.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path

BASELINE = '09d33e8d031ce7f4e50ea4d7a384360528b1127d'
EXPECTED = ('0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3',
            'f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88')
DOMAINS = ('REFUGE', 'RIM_ONE_r3', 'Drishti_GS')


def patient_alias_source(source):
    tree = ast.parse(source)
    function = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == '_fundus_pairs')
    return any(isinstance(n, ast.Dict) and any(
        isinstance(k, ast.Constant) and k.value == 'patient_id'
        and isinstance(v, ast.Name) and v.id == 'case_id'
        for k, v in zip(n.keys, n.values)) for n in ast.walk(function))


def summarize(rows, records, aliases):
    rows = [r for r in rows if r['dataset'] == 'fundus']
    index = {r['case_id']: r for r in records}
    if len(index) != len(records) or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate case metadata')
    if set(index) != {r['case_id'] for r in rows}:
        raise ValueError('manifest/split coverage mismatch')
    for r in rows:
        if r['site_or_vendor'] not in DOMAINS:
            raise ValueError('unknown domain')
        for k in ('patient_id', 'site_or_vendor', 'primary_20pct_split'):
            if r[k] != index[r['case_id']][k]:
                raise ValueError('manifest/split role mismatch')
        if r['primary_20pct_split'] == 'train_unlabeled' and (
                r.get('label_h5_relpath') or r.get('label_sha256')):
            raise ValueError('U exposes label location')
    counts = {}
    for d in DOMAINS:
        rr = [r for r in rows if r['site_or_vendor'] == d]
        counts[d] = {role: dict(images=len(v), unique_manifest_ids=len({r['patient_id'] for r in v}),
                               ids_equal_case=sum(r['patient_id'] == r['case_id'] for r in v))
                     for role in ('train_labeled', 'train_unlabeled')
                     for v in [[r for r in rr if r['primary_20pct_split'] == role]]}
    return dict(status='BLOCKED_PATIENT_IDENTITY_UNVERIFIED', metadata_consistency='PASS',
                counts=counts, distinct_clinical_patients=None,
                manifest_patient_id_origin='case_id alias' if aliases else 'UNVERIFIED',
                patient_disjointness='UNVERIFIED', case_uniqueness='PASS',
                actual_leakage_demonstrated=False, roles_assigned=False,
                proposed_counts_if_one_case_per_patient=dict(M_fit=16, A_fit=8,
                    Q_train_old=4, Q_train_new=4, Q_dev_old=4, Q_dev_new=4),
                first_domain_label_budget_status='UNRESOLVED_PATIENT_COUNT',
                note='40 labeled images meet the arithmetic threshold only if 40 distinct patients are verified. '
                     'No clinical patient map or binding one-image-per-patient provenance was supplied. '
                     'A case-level disjointness test cannot establish patient disjointness.',
                reads=dict(metadata_only=True, image_arrays=0, label_arrays=0,
                           checkpoint_payloads=0, optimizer_calls=0))


def audit(manifest, split, builder_source):
    # Frozen native metadata digests are an existing admission requirement.
    for path, expected in zip((manifest, split), EXPECTED):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('designated frozen metadata differs')
    with Path(manifest).open() as f:
        rows = list(csv.DictReader(f))
    payload = json.loads(Path(split).read_text())
    if payload['seed'] != 0:
        raise ValueError('only seed0 authorized')
    result = summarize(rows, payload['records'], patient_alias_source(builder_source))
    result.update(baseline_commit=BASELINE, frozen_manifest_digest=EXPECTED[0],
                  frozen_split_digest=EXPECTED[1],
                  historical_overlap_flag=payload['patient_overlap_check'],
                  historical_overlap_flag_scope='uniqueness of generated identifiers')
    return result


def selfcheck():
    source = "def _fundus_pairs(dataset):\n return [{'patient_id': case_id}]\n"
    assert patient_alias_source(source)
    row = dict(dataset='fundus', site_or_vendor='REFUGE', case_id='a', patient_id='a',
               primary_20pct_split='train_labeled')
    out = summarize([row], [row], True)
    assert out['status'] == 'BLOCKED_PATIENT_IDENTITY_UNVERIFIED'
    assert out['distinct_clinical_patients'] is None and not out['roles_assigned']
    bad = dict(row, patient_id='different')
    try:
        summarize([row], [bad], True)
    except ValueError:
        pass
    else:
        raise AssertionError('identity mismatch was accepted')
    print('metadata admission selfcheck PASS; zero optimizer calls')


if __name__ == '__main__':
    selfcheck()
