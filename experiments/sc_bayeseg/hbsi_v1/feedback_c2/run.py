"""BMC source-only paired replication of C1 feedback; private artifacts stay on NAS."""
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..budgets import ResourceLedger
from ..config import Config
from ..feedback_c1.model import FeedbackModel
from ..fullfield_bayeseg_a1.run import batch, evaluate as evaluate_base, train
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write
from ..unet_transfer_v45.run import risk as standard_risk, unet


class MCView:
    def __init__(self, model):
        self.model = model

    def eval(self):
        self.model.eval()

    def predict(self, image, seed, draws=4):
        return self.model.predict_mc(image, seed, draws)


def main():
    root = Path(os.environ['RUN_ROOT'])
    protocol = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert protocol['purpose'] == 'BMC_FEEDBACK_C2'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = replace(Config(**protocol['config']), seed=20261023)
    ledger = ResourceLedger(root / 'results', max_seconds=12600, max_updates=3200,
                            max_bytes=3_000_000_000)

    def status(stage, **extra):
        p = root / 'STATUS.tmp'
        p.write_text(json.dumps(dict(stage=stage, time=time.time(), **extra), indent=2))
        p.replace(root / 'STATUS.json')

    def fresh(cut):
        model = FeedbackModel(cut=cut).double().cuda()
        model.load_state_dict(torch.load(root / 'results/preflight/PRIVATE_INITIAL.pt',
                                         map_location='cpu', weights_only=True))
        return model

    def score(arm, model, domain, split='val', mode='mean'):
        name = f'{arm}_{domain}' + ('' if split == 'val' else '_train') + ('_MC4' if mode == 'mc' else '')
        status('EVALUATE', arm=arm, domain=domain, split=split, mode=mode)
        with ledger.attempt(name, seconds=800) as out:
            obj = model if mode == 'mean' else MCView(model)
            summary = evaluate_base(protocol, obj, domain, split, out / 'validation')
            summary['predictor'] = 'standard_UNet_two_logit_softmax' if mode == 'mean' else 'local_qZ_MC4_secondary'
            summary['draws'] = 1 if mode == 'mean' else 4
            (out / 'validation/SUMMARY.json').write_text(json.dumps(summary, indent=2))
        return summary

    try:
        assert json.loads((Path(protocol['c1_root']) / 'EXIT.json').read_text())['exit_code'] == 0
        source = data(protocol, cfg, 'BMC', 'L')
        assert source.counts == {'L': 134, 'U': 0}
        assert cfg.height == cfg.width == 128 and cfg.dtype == 'float64' and cfg.outer_lr == 3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight', seconds=300, updates=0) as out:
            reference = unet()
            canonical = FeedbackModel(cut=False).double().cuda()
            assert all(torch.equal(canonical.unet.state_dict()[k], reference.state_dict()[k])
                       for k in reference.state_dict())
            torch.save(canonical.state_dict(), out / 'PRIVATE_INITIAL.pt')
            cut, joint = fresh(True), fresh(False)
            assert all(torch.equal(cut.state_dict()[k], joint.state_dict()[k]) for k in cut.state_dict())
            image, label = batch(source, 0, 263000)
            noise = 54000000 + (263000 - 163000) * 10000
            ca, cv, clogits, _ = cut.objectives(image, label,
                                                 torch.Generator(device='cuda').manual_seed(noise))
            ja, jv, jlogits, _ = joint.objectives(image, label,
                                                  torch.Generator(device='cuda').manual_seed(noise))
            for a, b in ((ca, ja), (cv, jv), (clogits, jlogits)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            baseline = standard_risk(reference, image, label)
            torch.testing.assert_close(ca, baseline, rtol=0, atol=0)
            cp, jp, rp = list(cut.unet.parameters()), list(joint.unet.parameters()), list(reference.parameters())
            vg_cut = torch.autograd.grad(cv, cp, allow_unused=True, retain_graph=True)
            assert all(g is None for g in vg_cut)
            vg_joint = torch.autograd.grad(jv, jp, allow_unused=True, retain_graph=True)
            assert any(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in vg_joint)
            task_cut = torch.autograd.grad(ca + .01 * cv, cp, retain_graph=True)
            task_ref = torch.autograd.grad(baseline, rp)
            for a, b in zip(task_cut, task_ref):
                torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
            write(root / 'PREFLIGHT.json', dict(status='PASS', production_updates=0,
                                                paired_initial_exact=True, cut_standard_gradient_exact=True,
                                                joint_variational_gradient_nonzero=True,
                                                BMC_labeled_slices=source.counts['L']))
            del cut, joint, canonical, reference
        cells = {}
        for arm, detached in (('Cut', True), ('Joint', False)):
            model = fresh(detached)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)
            status('TRAIN', arm=arm)
            train(model, optimizer, source, 263000, 1600, arm, ledger, 3600)
            for domain in ('BMC', 'RUNMC'):
                cells[f'{arm}_{domain}'] = score(arm, model, domain)
                cells[f'{arm}_{domain}_MC4'] = score(arm, model, domain, mode='mc')
            cells[f'{arm}_BMC_train'] = score(arm, model, 'BMC', split='train_labeled')
            del model, optimizer
        write(root / 'RESULTS.json', dict(cells=cells, formal_updates=3200,
                                          training_seed=20261023, target_not_run=True))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
