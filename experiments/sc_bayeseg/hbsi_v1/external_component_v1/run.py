"""Sealed R2 models on all four previously unused prostate validation sites."""
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from ..data_adapter import confined, restore_probability
from ..feedback_c1.model import FeedbackModel
from ..real_diagnostic import write
from ...sslcl_v1.prostate_metrics import binary_phase_metrics


SITES = ('BIDMC', 'HK', 'I2CVB', 'UCL')
COUNTS = {'BIDMC': 2, 'HK': 2, 'I2CVB': 3, 'UCL': 2}
ARMS = (('ReplayCut', True), ('ReplayJoint', False))


def main():
    root = Path(os.environ['RUN_ROOT'])
    p = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose'] == 'EXTERNAL_COMPONENT_V1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    r2 = Path(p['r2_root'])
    started = time.monotonic()

    def status(stage, **extra):
        tmp = root / 'STATUS.tmp'
        tmp.write_text(json.dumps(dict(stage=stage, time=time.time(), **extra), indent=2))
        tmp.replace(root / 'STATUS.json')

    def checkpoint(arm):
        return r2 / f'results/{arm}/PRIVATE_CHECKPOINT_800.pt'

    try:
        status('PREFLIGHT')
        assert json.loads((r2 / 'STATUS.json').read_text())['stage'] == 'COMPLETE'
        assert json.loads((r2 / 'EXIT.json').read_text())['exit_code'] == 0
        previous = json.loads((r2 / 'RESULTS.json').read_text())['cells']
        assert previous['ReplayJoint_BMC']['HD95_penalized_mm'] > (
            1.1 * previous['ReplayCut_BMC']['HD95_penalized_mm'])
        manifest = Path(p['data_root']) / 'manifests/training/lcrseg_v1_seed0.csv'
        all_rows = list(csv.DictReader(manifest.open()))
        external = [row for site in SITES for row in all_rows
                    if row['dataset'] == 'prostate'
                    and row['site_or_vendor'] == site
                    and row['primary_20pct_split'] == 'val']
        counts = Counter(row['site_or_vendor'] for row in external)
        assert dict(counts) == COUNTS and len(external) == 9
        old_ids = {row['patient_id'] for row in all_rows
                   if row['dataset'] == 'prostate'
                   and row['site_or_vendor'] in ('RUNMC', 'BMC')}
        assert not old_ids.intersection(row['patient_id'] for row in external)
        assert len({row['patient_id'] for row in external}) == 9
        for row in external:
            for key in ('image_h5_relpath', 'label_h5_relpath'):
                assert confined(Path(p['data_root']) / 'h5/v1', row[key]).is_file()
        for arm, cut in ARMS:
            sealed = torch.load(checkpoint(arm), map_location='cpu', weights_only=True)
            assert sealed['step'] == 800
            model = FeedbackModel(cut=cut).double().cuda().eval()
            model.load_state_dict(sealed['model'])
            with torch.no_grad():
                synthetic = torch.zeros((1, 1, 128, 128), dtype=torch.float64,
                                        device='cuda')
                output = model.predict(synthetic, seed=1101)
            assert output.shape == (1, 2, 128, 128)
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output.sum(1), torch.ones_like(output[:, 0]),
                                       rtol=1e-12, atol=1e-12)
            del model, sealed, synthetic, output
            torch.cuda.empty_cache()
        write(root / 'PREFLIGHT.json', dict(status='PASS', formal_updates=0,
            real_external_forward=0, synthetic_forward=2, sites=COUNTS,
            patient_identity_overlap=0, r2_failed_verdict_preserved=True))

        folder = root / 'private_predictions'
        folder.mkdir(exist_ok=False)
        slice_calls = 0
        for arm, cut in ARMS:
            model = FeedbackModel(cut=cut).double().cuda().eval()
            sealed = torch.load(checkpoint(arm), map_location='cpu', weights_only=True)
            model.load_state_dict(sealed['model'])
            status('PREDICT', arm=arm)
            for global_i, row in enumerate(external):
                assert time.monotonic() - started < 6600
                image_path = confined(Path(p['data_root']) / 'h5/v1',
                                      row['image_h5_relpath'])
                with h5py.File(image_path, 'r') as f:
                    image = torch.from_numpy(f['image'][...].copy())[:, None].double()
                probabilities = []
                with torch.no_grad():
                    for j, item in enumerate(image):
                        resized = F.interpolate(item[None], (128, 128),
                                                mode='bilinear', align_corners=False).cuda()
                        probabilities.append(model.predict(resized,
                            seed=1101 + 10000 * global_i + 100 * j).cpu())
                        slice_calls += 1
                torch.save(torch.cat(probabilities),
                           folder / f'PRIVATE_{arm}_{global_i}.pt')
            del model, sealed
            torch.cuda.empty_cache()
        write(root / 'PREDICTIONS_SEALED.json', dict(status='PASS',
            site_counts=COUNTS, models=2, patients=9,
            source='frozen_R2_step800', labels_read=False))

        status('SCORE')
        audit = []
        structure = ndimage.generate_binary_structure(3, 1)
        for global_i, row in enumerate(external):
            assert time.monotonic() - started < 6600
            image_path = confined(Path(p['data_root']) / 'h5/v1', row['image_h5_relpath'])
            label_path = confined(Path(p['data_root']) / 'h5/v1', row['label_h5_relpath'])
            with h5py.File(image_path, 'r') as f:
                native_shape = tuple(f['image'].shape[-2:])
                spacing = json.loads(f.attrs['processed_spacing'])
            with h5py.File(label_path, 'r') as f:
                truth = f['label'][...]
            assert truth.ndim == 3 and np.isin(truth, [0, 1]).all()
            site = row['site_or_vendor']
            index = sum(v['site_or_vendor'] == site for v in external[:global_i]) + 1
            for arm, _ in ARMS:
                probability = torch.load(folder / f'PRIVATE_{arm}_{global_i}.pt',
                                         map_location='cpu', weights_only=True)
                restored = restore_probability(probability, native_shape)
                raw = binary_phase_metrics(restored, truth, spacing)
                pred = restored.argmax(1).numpy() == 1
                labels, count = ndimage.label(pred, structure=structure)
                if count:
                    sizes = np.bincount(labels.ravel())[1:]
                    keep = labels == (int(sizes.argmax()) + 1)
                else:
                    keep = pred
                removed = pred & ~keep
                hard = torch.from_numpy(np.stack((~keep, keep), axis=1).copy()).double()
                changed = binary_phase_metrics(hard, truth, spacing)['classes'][0]
                audit.append(dict(site=site, index=index, arm=arm,
                    PRIVATE_patient=row['patient_id'],
                    original=dict(raw['classes'][0], NLL=raw['NLL'], Brier=raw['Brier']),
                    component=changed, components=int(count),
                    largest_fraction=float(keep.sum() / max(1, pred.sum())),
                    removed_FP=int((removed & (truth != 1)).sum()),
                    removed_TP=int((removed & (truth == 1)).sum())))
        assert len(audit) == 18
        write(root / 'PRIVATE_RESULTS.json', dict(status='COMPLETE', rows=audit,
            formal_updates=0, inference_patient_volumes=18,
            synthetic_forward=2, slice_forward_calls=slice_calls, metric_calls=36,
            elapsed_compute_seconds=time.monotonic() - started))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
