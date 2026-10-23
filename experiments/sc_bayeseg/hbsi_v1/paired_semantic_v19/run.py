"""Bounded sequential L/U and two-domain factorial experiment."""
import csv
import json
import os
from pathlib import Path
import time
from dataclasses import replace
import numpy as np
import torch
from torch.nn import functional as F
from ..config import Config
from ..stream_state import StreamState, digest
from ..data_adapter import CurrentTrainingData, confined, restore_probability
from ..budgets import ResourceLedger, Counters
from ..local_inference import Refiner
from ..training import run_stage
from ..qualification import fingerprint
from ..real_diagnostic import write
from .method import UnifiedObjective, predict, balanced_risk
from ..unified_sslcl_v18.method import UnifiedObjective as OriginalObjective


def data(p, cfg, domain, arm):
    prepared = CurrentTrainingData.inspect_manifest(p['data_root'], 'prostate', domain, p['data_version'], roles=('L',) if arm == 'L' else ('L','U'))
    return CurrentTrainingData(p['data_root'], 'prostate', domain, domain, cfg, data_version=p['data_version'], prepared=prepared)


def evaluate(p, state, domain, memory, out):
    import h5py
    from sc_bayeseg.sslcl_v1.prostate_metrics import binary_phase_metrics
    out.mkdir(exist_ok=False)
    rows = [r for r in csv.DictReader((Path(p['data_root'])/'manifests/training/lcrseg_v1_seed0.csv').open()) if r['dataset']=='prostate' and r['site_or_vendor']==domain and r['primary_20pct_split']=='val']
    summaries = []
    for i, row in enumerate(rows):
        with h5py.File(confined(Path(p['data_root'])/'h5/v1',row['image_h5_relpath']),'r') as f:
            x = torch.from_numpy(f['image'][...].copy())[:,None].double()
            spacing = json.loads(f.attrs['processed_spacing'])
        small = F.interpolate(x,(128,128),mode='bilinear',align_corners=False).cuda()
        probabilities = torch.stack([predict(small,state,memory,seed).cpu() for seed in (1101,1102,1103)])
        torch.save(probabilities, out/f'PRIVATE_PREDICTION_{i}.pt')
        with h5py.File(confined(Path(p['data_root'])/'h5/v1',row['label_h5_relpath']),'r') as f:
            truth = f['label'][...]
        metrics = [binary_phase_metrics(restore_probability(q,x.shape[-2:]),truth,spacing) for q in probabilities]
        write(out/f'PRIVATE_METRICS_{i}.json', dict(patient=row['patient_id'],metrics=metrics))
        summaries.append(metrics)
    if not summaries:
        raise ValueError('empty validation domain')
    repeats = []
    for j in range(3):
        mm = [m[j] for m in summaries]
        cells = [m['classes'][0] for m in mm]
        repeats.append(dict(Dice=float(np.mean([v['Dice'] for v in cells])), NLL=float(np.mean([v['NLL'] for v in mm])), Brier=float(np.mean([v['Brier'] for v in mm])), HD95_penalized_mm=float(np.mean([v['HD95_penalized_mm'] for v in cells])),empty_predictions=sum(v['empty_prediction_failure'] for v in cells)))
    result = dict(domain=domain,memory=memory,patients=len(rows),repeats=repeats,predictor='variational_approximation',complete=True)
    write(out/'SUMMARY.json',result)
    return result


def main():
    root = Path(os.environ['RUN_ROOT'])
    p = json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    if p['purpose'] != 'PAIRED_SEMANTIC_V19' or p['seeds'] != [20261023,20261024,20261025] or p['source_steps']!=1600 or p['target_steps']!=800:
        raise ValueError('protocol mismatch')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ledger = ResourceLedger(root/'results', max_seconds=42000,max_updates=19200,max_bytes=9_000_000_000)
    counter = Counters()
    cfg = Config(**p['config'])
    def status(stage, **extra):
        tmp=root/'STATUS.tmp';tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**extra),indent=2));tmp.replace(root/'STATUS.json')
    try:
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=300,updates=0) as out:
            # A class-imbalanced fixture distinguishes balanced from pixel CE.
            q=torch.tensor([[[[.8,.8,.8,.8]],[[.2,.2,.2,.2]]]],dtype=torch.float64)
            y=torch.tensor([[[[1.,1.,1.,0.]],[[0.,0.,0.,1.]]]],dtype=torch.float64)
            reference=(-torch.log(q.new_tensor(.8))-torch.log(q.new_tensor(.2)))/2 + 1-(.4+1e-6)/(1.8+1e-6)
            torch.testing.assert_close(balanced_risk(q,y)[0],reference)
            empty=y.clone();empty[:,0]=1;empty[:,1]=0
            assert torch.isfinite(balanced_risk(q,empty)).all()
            times={}
            records=list(csv.DictReader((Path(p['data_root'])/'manifests/training/lcrseg_v1_seed0.csv').open()))
            for domain in ('RUNMC','BMC'):
                groups={}
                for row in records:
                    if row['dataset']=='prostate' and row['site_or_vendor']==domain:
                        groups.setdefault(row['primary_20pct_split'],set()).add(row['patient_id'])
                assert all(groups.get(role) for role in ('train_labeled','train_unlabeled','val','test'))
                keys=list(groups)
                assert all(not (groups[a]&groups[b]) for i,a in enumerate(keys) for b in keys[i+1:])
            for domain in ('RUNMC','BMC'):
                dl,du=data(p,cfg,domain,'L'),data(p,cfg,domain,'LU')
                assert dl.counts['U']==0 and du.counts['U']>0
                for arm,d in [('L',dl),('LU',du)]:
                    s=StreamState(cfg);s.begin_task(domain,d.manifest_hash,d.evidence);s.to_device('cuda')
                    L=d.batch('L',1,torch.Generator().manual_seed(163000))
                    reference=dl.batch('L',1,torch.Generator().manual_seed(163000))
                    assert torch.equal(L['x'],reference['x']) and torch.equal(L['observations']['y_obs'],reference['observations']['y_obs'])
                    U=d.batch('U',1,torch.Generator().manual_seed(163001))
                    for b in (L,U):
                        if b is not None:
                            b['x']=b['x'].cuda()
                            if 'observations' in b:b['observations']={k:v.cuda() for k,v in b['observations'].items()}
                    before=s.generator.get_state()
                    old_loss,_=OriginalObjective(s,Refiner(s.model,s.phi,counter)).estimate(L,U,d.counts,s.generator)
                    old_end=s.generator.get_state();del old_loss
                    s.generator.set_state(before)
                    torch.cuda.synchronize();start=time.monotonic()
                    loss,parts=UnifiedObjective(s,Refiner(s.model,s.phi,counter)).estimate(L,U,d.counts,s.generator)
                    assert torch.equal(s.generator.get_state(),old_end), 'changed MC random stream'
                    loss.backward();torch.cuda.synchronize()
                    times[domain+'_'+arm]=time.monotonic()-start
                    for group in s.optimizer.param_groups:
                        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in group['params'])
                    assert s.q.raw_scale.grad.abs().sum()>0
                    s.finalize_task()
                    prob=predict(L['x'],s,domain,1101,outer=2,local=2)
                    assert torch.isfinite(prob).all()
                    torch.testing.assert_close(prob.sum(1),torch.ones_like(prob[:,0]))
            estimate=4800*(times['RUNMC_L']+times['RUNMC_LU'])+4800*(times['BMC_L']+times['BMC_LU'])
            write(out/'CHECKS.json',dict(step_seconds=times,estimated_training_seconds=estimate,real_optimizer_updates=0,distribution_gradients='PASS',paired_L='PASS',prediction='PASS',historical_risk_formula='PASS',paired_model_random_stream='PASS'))
            if estimate*2+7200>42000:raise RuntimeError('cost admission failed')
        all_results={}
        for seed in p['seeds']:
            c=replace(cfg,seed=seed)
            sources={}
            # Both single-domain arms finish before entering the two-domain stage.
            for arm in ('L','LU'):
                name=f's{seed}_{arm}_A';status('TRAIN_SOURCE',job=name)
                d=data(p,c,'RUNMC',arm);s=StreamState(c);s.begin_task('RUNMC',d.manifest_hash,d.evidence);s.to_device('cuda')
                run_stage(s,d,ledger,counter,name,1600,3600,fingerprint(p),data_seed=163000,checkpoint_stride=400,objective_factory=UnifiedObjective)
                sources[arm]=root/'results'/f'{name}_attempt001/PRIVATE_SEALED_STATE.pt'
                for domain in ('RUNMC','BMC'):
                    status('EVALUATE_SOURCE',job=name,domain=domain)
                    with ledger.attempt(f'eval_{name}_{domain}',seconds=900) as out:
                        all_results[f'{name}_{domain}']=evaluate(p,s,domain,'RUNMC',out/'validation')
                write(root/f'SOURCE_{seed}_{arm}.json',{k:v for k,v in all_results.items() if k.startswith(name)})
            status('ENTER_TWO_DOMAIN',seed=seed,performance_gate=False)
            for arm in ('L','LU'):
                for memory in (0,1):
                    name=f's{seed}_{arm}_M{memory}_B';status('TRAIN_TARGET',job=name)
                    s=StreamState.load(sources[arm]);d=data(p,c,'BMC',arm)
                    s.begin_task('BMC',d.manifest_hash,d.evidence)
                    if memory==0:
                        fresh=StreamState(c);fresh.begin_task('BMC',d.manifest_hash,d.evidence)
                        s.prior.load_state_dict(fresh.prior.state_dict());s.prior_hash=digest(s.prior.state_dict())
                    s.to_device('cuda')
                    run_stage(s,d,ledger,counter,name,800,3600,fingerprint(p),data_seed=263000,checkpoint_stride=400,objective_factory=UnifiedObjective)
                    for domain in ('RUNMC','BMC'):
                        status('EVALUATE_TARGET',job=name,domain=domain)
                        with ledger.attempt(f'eval_{name}_{domain}',seconds=900) as out:
                            all_results[f'{name}_{domain}']=evaluate(p,s,domain,domain,out/'validation')
                    write(root/f'TARGET_{seed}_{arm}_M{memory}.json',{k:v for k,v in all_results.items() if k.startswith(name)})
        write(root/'RESULTS.json',all_results)
        status('COMPLETE',scientific_claim='requires_analysis')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
