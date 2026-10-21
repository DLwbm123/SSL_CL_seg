"""Publish already audited phase metrics and risk decompositions without new scoring."""
import json
import os
from pathlib import Path

import torch


def export(settings):
    root = Path(settings['output'])
    read = lambda p: json.loads(Path(p).read_text())
    public = read(root / 'PUBLIC_RESULTS.json')
    assert public['status'] == 'COMPLETE_AUDIT_PASS'
    assert read(root / 'AUDIT.json')['status'] == 'PASS'
    load = lambda name: torch.load(root / name, map_location='cpu', weights_only=False)
    maps = {
        'train': read(Path(settings['train_readout_source']) / 'PRIVATE_EVALUATION_MAP.json'),
        'val': read(Path(settings['patient_control']) / 'PRIVATE_EVALUATION_MAP.json'),
    }
    patients, cases = {}, {}
    for cohort, rows in maps.items():
        patients.update({p: f'{cohort.upper()}{i:03}' for i, p in
                         enumerate(sorted({r['patient_id'] for r in rows}))})
        cases.update({r['case_id']: f'{cohort.upper()}_PHASE{i:03}'
                      for i, r in enumerate(rows)})

    def anonymous(value):
        if isinstance(value, dict):
            return {k: patients[v] if k == 'patient' else cases[v] if k == 'case'
                    else anonymous(v) for k, v in value.items()}
        if isinstance(value, list):
            return [anonymous(v) for v in value]
        return value

    public['phase_metrics'] = anonymous(load('PRIVATE_PHASE_RESULTS.pt'))
    public['phase_risks'] = {
        arm: [dict(phase=f'VAL_PHASE{i:03}', **anonymous(row))
              for i, row in enumerate(rows)]
        for arm, rows in load('PRIVATE_RISKS.pt').items()
    }
    assert sum(len(rows) for co in public['phase_metrics'].values()
               for rows in co.values()) == 264
    assert sum(map(len, public['phase_risks'].values())) == 184
    public['detail_export'] = dict(audited_records_only=True, phase_metrics=264,
                                  phase_risks=184, new_network=0, new_GT=0,
                                  new_optimizer=0, new_fits=0)
    (root / 'PUBLIC_RESULTS.json').write_text(json.dumps(public, indent=2) + '\n')


if __name__ == '__main__':
    export(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
