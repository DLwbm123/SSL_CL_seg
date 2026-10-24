"""Zero-update, source-only oracle shape and class-mean appearance audit."""
import json
import os
import time
from pathlib import Path

import h5py
import torch
from torch.nn import functional as F

from ..budgets import ResourceLedger
from ..data_adapter import confined, restore_probability
from ..fullfield_bayeseg_a1.run import records
from ..real_diagnostic import write
from ..semantic_shape_l1.model import SemanticShape
from ...sslcl_v1.prostate_metrics import binary_phase_metrics


def main():
    root = Path(os.environ['RUN_ROOT'])
    p = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose'] == 'ORACLE_APPEARANCE_Q0'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ledger = ResourceLedger(root / 'results', max_seconds=1800, max_updates=0,
                            max_bytes=1_000_000_000)

    def status(stage, **extra):
        tmp = root / 'STATUS.tmp'
        tmp.write_text(json.dumps(dict(stage=stage, time=time.time(), **extra), indent=2))
        tmp.replace(root / 'STATUS.json')

    try:
        l1 = Path(p['l1_root'])
        assert json.loads((l1 / 'EXIT.json').read_text())['exit_code'] == 0
        rows = records(p, 'RUNMC', 'train_labeled')
        assert len(rows) == 4
        volumes = []
        labels = []
        spacings = []
        for row in rows:
            with h5py.File(confined(Path(p['data_root']) / 'h5/v1',
                                    row['image_h5_relpath']), 'r') as f:
                volumes.append(torch.from_numpy(f['image'][...].copy())[:, None].double())
                spacings.append(json.loads(f.attrs['processed_spacing']))
            with h5py.File(confined(Path(p['data_root']) / 'h5/v1',
                                    row['label_h5_relpath']), 'r') as f:
                labels.append(f['label'][...].copy())
        assert sum(len(v) for v in volumes) == 80

        model = SemanticShape(True).double().cuda()
        saved = torch.load(l1 / 'results/Cut/PRIVATE_CHECKPOINT_1600.pt',
                           map_location='cpu', weights_only=True)
        assert saved['step'] == 1600
        model.load_state_dict(saved['model'])
        model.eval()
        reference = torch.load(Path(p['unet_root']) /
                               'results/U1600/PRIVATE_CHECKPOINT.pt',
                               map_location='cpu', weights_only=True)['model']
        status('PREFLIGHT')
        with ledger.attempt('preflight', seconds=180, updates=0):
            state = model.base.unet.state_dict()
            assert set(state) == set(reference)
            assert max(float((state[k].cpu() - reference[k]).abs().max())
                       for k in state) == 0
            assert all(rows[i]['patient_id'] != rows[(i+1) % 4]['patient_id']
                       for i in range(4))
            means = []
            with torch.no_grad():
                for volume, truth in zip(volumes, labels):
                    assert len(volume) == len(truth)
                    sums = [0., 0.]
                    counts = [0, 0]
                    for image, mask in zip(volume, truth):
                        x = F.interpolate(image[None], (128, 128), mode='bilinear',
                                          align_corners=False)
                        y = F.interpolate(torch.from_numpy(mask.copy())[None, None].double(),
                                          (128, 128), mode='nearest')
                        assert torch.isfinite(x).all()
                        assert bool(((y == 0) | (y == 1)).all())
                        for cls in (0, 1):
                            which = y == cls
                            sums[cls] += float(x[which].sum())
                            counts[cls] += int(which.sum())
                    assert min(counts) > 0
                    mu = [sums[c] / counts[c] for c in (0, 1)]
                    assert all(-1 <= value <= 1 for value in mu)
                    means.append(mu)
                x = F.interpolate(volumes[0][:1], (128, 128), mode='bilinear',
                                  align_corners=False).cuda()
                assert torch.isfinite(model.base.unet(x).logits).all()
            write(root / 'PREFLIGHT.json', dict(status='PASS', formal_updates=0,
                l1_complete=True, checkpoint_step1600=True, U_exact=True,
                source_patients=4, source_slices=80, partner_different=True,
                each_class_nonempty=True, means_finite=True,
                deterministic_inference_finite=True))

        status('AUDIT')
        patient_rows = []
        with ledger.attempt('audit', seconds=1500, updates=0) as out:
            with torch.no_grad():
                for i, (row, volume, truth, spacing) in enumerate(
                        zip(rows, volumes, labels, spacings)):
                    own_bg, own_fg = means[i]
                    other_bg, other_fg = means[(i + 1) % 4]
                    predictions = {name: [] for name in ('original', 'class_only', 'delta')}
                    reconstruction_sse = 0.
                    reconstruction_pixels = 0
                    outside_pixels = 0
                    delta_pixels = 0
                    for image, mask in zip(volume, truth):
                        ledger.check()
                        x = F.interpolate(image[None], (128, 128), mode='bilinear',
                                          align_corners=False).cuda()
                        y = F.interpolate(torch.from_numpy(mask.copy())[None, None].double(),
                                          (128, 128), mode='nearest').cuda()
                        class_only = own_bg * (1 - y) + own_fg * y
                        delta = x + (other_bg - own_bg) * (1 - y) + (other_fg - own_fg) * y
                        assert torch.isfinite(class_only).all() and torch.isfinite(delta).all()
                        reconstruction_sse += float((class_only - x).square().sum())
                        reconstruction_pixels += x.numel()
                        outside_pixels += int(((delta < -1) | (delta > 1)).sum())
                        delta_pixels += delta.numel()
                        for name, value in (('original', x), ('class_only', class_only),
                                            ('delta', delta)):
                            prob = model.base.unet(value).logits.softmax(1).cpu()
                            predictions[name].append(prob)
                    metrics = {}
                    for name, parts in predictions.items():
                        probability = restore_probability(torch.cat(parts), volume.shape[-2:])
                        metrics[name] = binary_phase_metrics(probability, truth, spacing)
                        torch.save(probability, out / f'PRIVATE_{i}_{name}.pt')
                    patient_rows.append(dict(patient=row['patient_id'], metrics=metrics,
                        class_only_mse=reconstruction_sse/reconstruction_pixels,
                        delta_outside_fraction=outside_pixels/delta_pixels,
                        slices=len(volume)))
            write(out / 'PRIVATE_PATIENTS.json', patient_rows)
        total_slices = sum(v['slices'] for v in patient_rows)
        result = dict(patients=4, slices=total_slices, formal_updates=0,
            target_data_read=False, oracle_source_labels_used=True,
            class_only_mse=sum(v['class_only_mse']*v['slices'] for v in patient_rows)/total_slices,
            delta_outside_fraction=sum(v['delta_outside_fraction']*v['slices']
                                       for v in patient_rows)/total_slices,
            **{name+'_Dice':sum(v['metrics'][name]['classes'][0]['Dice']
                                for v in patient_rows)/4
               for name in ('original', 'class_only', 'delta')})
        write(root / 'RESULTS.json', result)
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
