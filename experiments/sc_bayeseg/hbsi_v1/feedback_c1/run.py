"""Source-gated Cut/Joint experiment; private artifacts remain on NAS."""
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..budgets import ResourceLedger
from ..config import Config
from ..fullfield_bayeseg_a1.run import batch, evaluate as evaluate_base, train
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write
from ..unet_transfer_v45.run import risk as historical_risk, unet
from .model import FeedbackModel


class MCView:
    def __init__(self, model):
        self.model = model

    def eval(self):
        self.model.eval()

    def predict(self, image, seed, draws=4):
        return self.model.predict_mc(image, seed, draws)


def evaluate(protocol, model, domain, split, out, mode):
    obj = model if mode == 'mean' else MCView(model)
    summary = evaluate_base(protocol, obj, domain, split, out)
    summary['predictor'] = 'standard_UNet_two_logit_softmax' if mode == 'mean' else 'local_qZ_MC4_secondary'
    summary['draws'] = 1 if mode == 'mean' else 4
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
    return summary


def patients(folder):
    rows = [json.loads((folder / f'PRIVATE_METRICS_{i}.json').read_text()) for i in range(4)]
    return rows, [row['metrics']['classes'][0] for row in rows]


def historical_patients(folder):
    rows = [json.loads((folder / f'PRIVATE_METRICS_{i}.json').read_text()) for i in range(4)]
    return rows, [row['metrics'][0]['classes'][0] for row in rows]


def exact_unet(model, path):
    old = torch.load(path, map_location='cpu', weights_only=True)['model']
    new = model.unet.state_dict()
    assert set(new) == set(old)
    max_error = max(float((new[k].detach().cpu() - old[k]).abs().max()) for k in new)
    assert max_error <= 1e-10, max_error
    return max_error


def main():
    root = Path(os.environ['RUN_ROOT'])
    protocol = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert protocol['purpose'] == 'HIERARCHICAL_FEEDBACK_C1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = replace(Config(**protocol['config']), seed=20261023)
    ledger = ResourceLedger(root / 'results', max_seconds=12600, max_updates=4800,
                            max_bytes=3_000_000_000)

    def status(stage, **extra):
        p = root / 'STATUS.tmp'
        p.write_text(json.dumps(dict(stage=stage, time=time.time(), **extra), indent=2))
        p.replace(root / 'STATUS.json')

    def initial(cut):
        model = FeedbackModel(cut=cut).double().cuda()
        model.load_state_dict(torch.load(root / 'results/preflight/PRIVATE_INITIAL.pt',
                                         map_location='cpu', weights_only=True))
        return model

    def score(arm, model, stage, domain, split='val', mode='mean'):
        key = f'{arm}_{stage}_{domain}' + ('' if split == 'val' else '_train') + ('_MC4' if mode == 'mc' else '')
        status('EVALUATE', arm=arm, stage_name=stage, domain=domain, split=split, mode=mode)
        with ledger.attempt(key, seconds=800) as out:
            cells[key] = evaluate(protocol, model, domain, split, out / 'validation', mode)
        return key

    try:
        base = Path(protocol['unet_root'])
        assert json.loads((base / 'EXIT.json').read_text())['exit_code'] == 0
        assert json.loads((Path(protocol['b1_root']) / 'EXIT.json').read_text())['exit_code'] == 0
        source = data(protocol, cfg, 'RUNMC', 'L')
        target = data(protocol, cfg, 'BMC', 'L')
        assert source.counts == {'L': 80, 'U': 0} and target.counts == {'L': 134, 'U': 0}
        assert cfg.height == cfg.width == 128 and cfg.dtype == 'float64' and cfg.outer_lr == 3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight', seconds=300, updates=0) as out:
            canonical = FeedbackModel(cut=False).double().cuda()
            reference = unet()
            assert all(torch.equal(canonical.unet.state_dict()[k], reference.state_dict()[k])
                       for k in reference.state_dict())
            torch.save(canonical.state_dict(), out / 'PRIVATE_INITIAL.pt')
            cut, joint = initial(True), initial(False)
            assert all(torch.equal(cut.state_dict()[k], joint.state_dict()[k])
                       for k in cut.state_dict())
            probes = {}
            for domain, dataset, seed in (('RUNMC', source, 163000), ('BMC', target, 263000)):
                image, label = batch(dataset, 0, seed)
                noise = 54000000
                ca, cv, clogits, cd = cut.objectives(image, label,
                                                      torch.Generator(device='cuda').manual_seed(noise))
                ja, jv, jlogits, jd = joint.objectives(image, label,
                                                       torch.Generator(device='cuda').manual_seed(noise))
                torch.testing.assert_close(ca, ja, rtol=0, atol=0)
                torch.testing.assert_close(cv, jv, rtol=0, atol=0)
                torch.testing.assert_close(clogits, jlogits, rtol=0, atol=0)
                for name in ('shape','appearance','z','probability','logq'):
                    torch.testing.assert_close(cd[name], jd[name], rtol=0, atol=0)
                baseline = historical_risk(reference, image, label)
                torch.testing.assert_close(ca, baseline, rtol=0, atol=0)
                cp = list(cut.unet.parameters());jp = list(joint.unet.parameters());rp = list(reference.parameters())
                v_cut = torch.autograd.grad(cv, cp, allow_unused=True, retain_graph=True)
                assert all(v is None for v in v_cut)
                v_joint = torch.autograd.grad(jv, jp, allow_unused=True, retain_graph=True)
                assert any(v is not None and torch.isfinite(v).all() and v.norm()>0 for v in v_joint)
                cut_grad = torch.autograd.grad(ca + .01*cv, cp, retain_graph=True)
                ref_grad = torch.autograd.grad(baseline, rp)
                for a,b in zip(cut_grad,ref_grad):
                    torch.testing.assert_close(a,b,rtol=0,atol=1e-12)
                torch.testing.assert_close(cut.predict(image, seed=1101).sum(1),
                                           torch.ones_like(image[:,0]),rtol=0,atol=1e-12)
                probes[domain] = dict(cut_task=float(ca.detach()), bound_per_pixel=float((-cv).detach()),
                                      joint_variational_gradient_parameters=sum(v is not None and bool(v.norm()>0)
                                                                         for v in v_joint),
                                      cut_variational_gradient_parameters=sum(v is not None for v in v_cut),
                                      max_supervised_gradient_error=max(float((a-b).abs().max())
                                                                    for a,b in zip(cut_grad,ref_grad)))
            restored = initial(True)
            assert all(torch.equal(cut.state_dict()[k], restored.state_dict()[k]) for k in cut.state_dict())
            write(root / 'PREFLIGHT.json', dict(status='PASS', true_backward_probes=4,
                                              production_optimizer_updates=0,
                                              initial_forward_identical=True, cut_std_gradient_equivalent=True,
                                              cut_variational_gradient_zero=True,
                                              joint_variational_gradient_nonzero=True,
                                              probes=probes))
            del cut, joint, canonical, reference, restored
        cells = {}
        cut = initial(True)
        optimizer = torch.optim.Adam(cut.parameters(), lr=cfg.outer_lr)
        status('TRAIN_SOURCE', arm='Cut')
        cut_source = train(cut, optimizer, source, 163000, 1600, 'Cut_source', ledger, 3000)
        cut_source_error = exact_unet(cut, base / 'results/U1600/PRIVATE_CHECKPOINT.pt')
        for domain in ('RUNMC','BMC'):
            score('Cut', cut, 'source', domain)
            score('Cut', cut, 'source', domain, mode='mc')
        score('Cut', cut, 'source', 'RUNMC', split='train_labeled')
        old_source = json.loads((base/'RESULTS.json').read_text())['cells']
        assert abs(cells['Cut_source_RUNMC']['Dice']-old_source['U1600_RUNMC']['repeats'][0]['Dice']) < 1e-10
        joint = initial(False)
        optimizer = torch.optim.Adam(joint.parameters(), lr=cfg.outer_lr)
        status('TRAIN_SOURCE', arm='Joint')
        joint_source = train(joint, optimizer, source, 163000, 1600, 'Joint_source', ledger, 3000)
        for domain in ('RUNMC','BMC'):
            score('Joint', joint, 'source', domain)
            score('Joint', joint, 'source', domain, mode='mc')
        score('Joint', joint, 'source', 'RUNMC', split='train_labeled')
        cut_rows, cut_pat = patients(root/'results/Cut_source_RUNMC/validation')
        joint_rows, joint_pat = patients(root/'results/Joint_source_RUNMC/validation')
        old_rows, old_pat = historical_patients(base/'results/eval_U1600_RUNMC/validation')
        assert [r['patient'] for r in cut_rows] == [r['patient'] for r in joint_rows] == [r['patient'] for r in old_rows]
        source_gates = dict(
            Cut_exact_v45=(cut_source_error<=1e-10),
            Joint_minus_Cut_mean_ge_minus_1pp=cells['Joint_source_RUNMC']['Dice']>=cells['Cut_source_RUNMC']['Dice']-.01,
            Joint_each_minus_Cut_ge_minus_5pp=all(j['Dice']>=c['Dice']-.05 for j,c in zip(joint_pat,cut_pat)),
            Joint_absolute_vs_U_mean_ge_minus_1pp=cells['Joint_source_RUNMC']['Dice']>=old_source['U1600_RUNMC']['repeats'][0]['Dice']-.01,
            Joint_each_absolute_vs_U_ge_minus_5pp=all(j['Dice']>=u['Dice']-.05 for j,u in zip(joint_pat,old_pat)),
            Joint_HD95_within_10pct_of_Cut=cells['Joint_source_RUNMC']['HD95_penalized_mm']<=1.1*cells['Cut_source_RUNMC']['HD95_penalized_mm'],
            no_empty_collapse=(all(cells[a]['empty_predictions']==0 for a in ('Cut_source_RUNMC','Joint_source_RUNMC','Cut_source_RUNMC_train','Joint_source_RUNMC_train'))
                               and all(p['TP']>0 for p in cut_pat+joint_pat)))
        write(root/'SOURCE_GATE.json', dict(status='PASS' if all(source_gates.values()) else 'FAILED',
                                            gates=source_gates, cut_v45_max_tensor_error=cut_source_error,
                                            formal_updates=3200,
                                            paired_dice_pp=[100*(j['Dice']-c['Dice']) for j,c in zip(joint_pat,cut_pat)]))
        if not all(source_gates.values()):
            write(root/'RESULTS.json', dict(cells=cells, formal_updates=3200, target_not_run=True,
                                            source_gate='FAILED', training_seed=20261023))
            status('SOURCE_GATE_FAILED')
            return
        for arm, model, checkpoint in (('Cut',cut,cut_source),('Joint',joint,joint_source)):
            sealed = torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert sealed['step']==1600
            model.load_state_dict(sealed['model'])
            optimizer = torch.optim.Adam(model.parameters(),lr=cfg.outer_lr)
            assert not optimizer.state
            status('TRAIN_TARGET',arm=arm)
            train(model,optimizer,target,263000,800,arm+'_target',ledger,1800)
            if arm=='Cut':
                target_error=exact_unet(model,base/'results/U_B800/PRIVATE_CHECKPOINT.pt')
                write(root/'CUT_TARGET_MATCH.json',dict(status='PASS',max_tensor_error=target_error))
            score(arm,model,'target','BMC',split='train_labeled')
            for domain in ('RUNMC','BMC'):
                score(arm,model,'target',domain)
                score(arm,model,'target',domain,mode='mc')
        write(root/'RESULTS.json',dict(cells=cells,formal_updates=4800,source_gate='PASS',
                                       target_complete=True,training_seed=20261023))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':
    main()
