import json
from collections import defaultdict
from pathlib import Path
from ..di_dmpa_jascl.data import stable_seed
from ..main_style_ssl_v1.data import Accessor, CurrentDomain
from ..main_style_ssl_v1.runtime import permit

NODE = dict(dataset='mnms', domain='Philips', seed=163, order=1, stage=1, steps=10000)


def subset(access, patients):
    keep = [i for i, r in enumerate(access.rows) if r['patient_id'] in patients]
    access.rows = [access.rows[i] for i in keep]
    access.depths = [access.depths[i] for i in keep]
    access.checked.clear(); access.by_patient = defaultdict(list)
    for i, row in enumerate(access.rows):
        access.by_patient[row['patient_id']].append(i)
    access.patients = sorted(access.by_patient)
    if not access.patients:
        raise ValueError('empty patient subset')
    return access


def split(data):
    access = Accessor(data, 'mnms', 'Philips', 'train_labeled')
    patients = sorted(access.patients, key=lambda p: (stable_seed(163, 'relation_holdout_v1', p), p))
    n = max(2, round(len(patients)*.2))
    if len(patients) != 13 or len(patients)-n < 2:
        raise ValueError('unexpected Philips patient population')
    return dict(holdout=sorted(patients[:n]), train=sorted(patients[n:]),
                rule='stable_seed(163, relation_holdout_v1, patient); first round(0.2*N), minimum 2')


def provider(cfg):
    frozen = json.loads(Path(cfg['split']).read_text())
    if frozen != split(cfg['data']):
        raise ValueError('registered patient split changed')
    result = CurrentDomain(cfg['data'], 'mnms', 'Philips', 163, 1, 1,
                           {'kind': 'source_FixMatch_EMA'}, 'cuda', permit(NODE, 'formal'))
    subset(result._l, set(frozen['train']))
    subset(result._u, set(result._u.patients)-set(frozen['holdout']))
    if set(result._l.patients) & set(frozen['holdout']) or set(result._u.patients) & set(frozen['holdout']):
        raise PermissionError('held-out patient leaked into training')
    return result
