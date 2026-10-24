"""Read-only 3D connected-component audit of sealed R2 probabilities."""
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage

from ..data_adapter import confined, restore_probability
from ..fullfield_bayeseg_a1.run import records
from ..real_diagnostic import write
from ...sslcl_v1.prostate_metrics import binary_phase_metrics


def main():
    root = Path(os.environ['RUN_ROOT'])
    p = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose'] == 'R2_COMPONENT_AUDIT_T0'
    r2 = Path(p['r2_root'])
    assert json.loads((r2 / 'EXIT.json').read_text())['exit_code'] == 0
    assert json.loads((r2 / 'STATUS.json').read_text())['stage'] == 'COMPLETE'
    started = time.monotonic()
    rows = []
    structure = ndimage.generate_binary_structure(3, 1)
    for split, name in (('train_labeled', 'BMC_train'), ('val', 'BMC')):
        cohort = records(p, 'BMC', split)
        for i, record in enumerate(cohort):
            image_path = confined(Path(p['data_root']) / 'h5/v1', record['image_h5_relpath'])
            label_path = confined(Path(p['data_root']) / 'h5/v1', record['label_h5_relpath'])
            with h5py.File(image_path, 'r') as f:
                native_shape = tuple(f['image'].shape[-2:])
                spacing = json.loads(f.attrs['processed_spacing'])
            with h5py.File(label_path, 'r') as f:
                truth = f['label'][...]
            gt_components = ndimage.label(truth == 1, structure=structure)[1]
            for arm in ('ReplayCut', 'ReplayJoint'):
                folder = r2 / f'results/{arm}_{name}/validation'
                sealed = json.loads((folder / f'PRIVATE_METRICS_{i}.json').read_text())
                assert sealed['patient'] == record['patient_id']
                probabilities = torch.load(folder / f'PRIVATE_PREDICTION_{i}.pt',
                                           map_location='cpu', weights_only=True)
                restored = restore_probability(probabilities, native_shape)
                original = binary_phase_metrics(restored, truth, spacing)['classes'][0]
                old = sealed['metrics']['classes'][0]
                assert abs(original['Dice'] - old['Dice']) <= 1e-9
                assert abs(original['HD95_penalized_mm'] - old['HD95_penalized_mm']) <= 1e-9
                prediction = restored.argmax(1).numpy() == 1
                labels, count = ndimage.label(prediction, structure=structure)
                if count:
                    sizes = np.bincount(labels.ravel())[1:]
                    keep = labels == (int(sizes.argmax()) + 1)
                else:
                    keep = prediction
                removed = prediction & ~keep
                hard = torch.from_numpy(np.stack((~keep, keep), axis=1).copy()).double()
                changed = binary_phase_metrics(hard, truth, spacing)['classes'][0]
                rows.append(dict(group=name, arm=arm, index=i + 1,
                    PRIVATE_patient=record['patient_id'],
                    gt_components=int(gt_components),
                    predicted_components=int(count),
                    largest_fraction=float(keep.sum() / max(1, prediction.sum())),
                    removed_voxels=int(removed.sum()),
                    removed_FP=int((removed & (truth != 1)).sum()),
                    removed_TP=int((removed & (truth == 1)).sum()),
                    baseline_Dice=float(original['Dice']),
                    baseline_HD95_mm=float(original['HD95_penalized_mm']),
                    largest_Dice=float(changed['Dice']),
                    largest_HD95_mm=float(changed['HD95_penalized_mm'])))
            assert time.monotonic() - started < 600
    assert len(rows) == 16
    write(root / 'PRIVATE_RESULTS.json', dict(status='COMPLETE', rows=rows,
        elapsed_compute_seconds=time.monotonic() - started,
        baseline_exact=True, formal_updates=0,
        foreground_rule='3D_6_connected_largest_first_tie'))
    write(root / 'STATUS.json', dict(stage='COMPLETE', time=time.time()))


if __name__ == '__main__':
    main()
