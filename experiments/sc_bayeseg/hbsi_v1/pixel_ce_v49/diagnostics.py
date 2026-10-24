"""Current-train-only volume diagnostics without changing access roles or scoring."""
import json
from pathlib import Path
from statistics import mean
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from ..data_adapter import restore_probability
from ..paired_semantic_v19.method import predict
from ..real_diagnostic import write
from sc_bayeseg.sslcl_v1.prostate_metrics import binary_phase_metrics


def prepare(base,root):
    assert base.counts==dict(L=134,U=0)
    groups={}
    for r in base.records['L']:groups.setdefault(str(r['image']),[]).append(r)
    assert len(groups)==4
    definitions=[];positive=[];lookup={}
    for case,(key,records) in enumerate(sorted(groups.items()),1):
        records=sorted(records,key=lambda r:r['index']);r=records[0]
        base.gate.check(r,'label')
        with h5py.File(r['label'],'r') as f:y=f['label'][...]
        assert len(y)==len(records) and [v['index'] for v in records]==list(range(len(y)))
        area=(y==1).sum((1,2)).tolist();positive.extend(v for v in area if v>0)
        definitions.append(dict(case=case,record=r,areas=area,exposures=[0]*len(records)))
        for i in range(len(records)):lookup[(key,i)]=(case-1,i)
    assert positive
    q1,q2=np.quantile(positive,[1/3,2/3]).tolist()
    for d in definitions:d['strata']=['empty' if n==0 else 'small' if n<=q1 else 'medium' if n<=q2 else 'large' for n in d['areas']]
    for step in range(800):
        i=int(torch.randint(len(base.records['L']),(1,),generator=torch.Generator().manual_seed(263000+2*step)))
        r=base.records['L'][i];case,index=lookup[(str(r['image']),r['index'])];definitions[case]['exposures'][index]+=1
    summary=dict(patients=4,slices=134,updates=800,positive_area_quantiles=[q1,q2],strata_counts={g:sum(d['strata'].count(g) for d in definitions) for g in ('empty','small','medium','large')},cases=[{k:d[k] for k in ('case','areas','strata','exposures')} for d in definitions],based_on_training_labels_only=True)
    write(root/'TRAIN_DEFINITION.json',summary)
    write(root/'PRIVATE_TRAIN_MANIFEST.json',[dict(case=d['case'],image=str(d['record']['image']),label=str(d['record']['label']),exposures=d['exposures']) for d in definitions])
    assert sum(sum(d['exposures']) for d in definitions)==800
    return definitions


def extras(prob,y,spacing,strata):
    pred=prob.argmax(1);target=torch.from_numpy(y.astype(np.int64));fg=target==1
    nll=-prob.double().clamp_min(1e-30).log().gather(1,target[:,None]).squeeze(1)
    brier=(prob.double()-F.one_hot(target,2).permute(0,3,1,2)).square().sum(1)
    conditional={name:dict(pixels=int(mask.sum()),NLL=float(nll[mask].mean()) if mask.any() else None,Brier=float(brier[mask].mean()) if mask.any() else None) for name,mask in [('foreground',fg),('background',~fg)]}
    slices=[]
    for i in range(len(y)):
        a=pred[i]==1;b=fg[i];tp=int((a&b).sum());pc=int(a.sum());gt=int(b.sum());fp=pc-tp
        slices.append(dict(index=i,stratum=strata[i],TP=tp,FP=fp,FN=gt-tp,predicted=pc,GT=gt,pixels=b.numel(),FP_volume_mm3=fp*float(np.prod(spacing))))
    empty=[v for v in slices if v['GT']==0]
    return dict(conditional_probability=conditional,slices=slices,empty_FP_volume_mm3=sum(v['FP_volume_mm3'] for v in empty),empty_FP_fraction=sum(v['FP'] for v in empty)/sum(v['pixels'] for v in empty) if empty else None)


def train_evaluate(state,base,definitions,out):
    out=Path(out);out.mkdir(exist_ok=False);patients=[];before=state.generator.get_state()
    for d in definitions:
        r=d['record'];base.gate.check(r,'image')
        with h5py.File(r['image'],'r') as f:
            x=torch.from_numpy(f['image'][...].copy())[:,None].double();spacing=json.loads(f.attrs['processed_spacing'])
        small=F.interpolate(x,(128,128),mode='bilinear',align_corners=False).cuda()
        probs=torch.stack([predict(small,state,'BMC',seed).cpu() for seed in (1101,1102,1103)])
        torch.save(probs,out/f"PRIVATE_PREDICTION_{d['case']}.pt")
        base.gate.check(r,'label')
        with h5py.File(r['label'],'r') as f:truth=f['label'][...]
        rows=[]
        for p in probs:
            p=restore_probability(p,x.shape[-2:]);rows.append(dict(**binary_phase_metrics(p,truth,spacing),**extras(p,truth,spacing,d['strata'])))
        patients.append(dict(case=d['case'],metrics=rows));write(out/f"CASE_{d['case']}.json",patients[-1])
    assert torch.equal(before,state.generator.get_state())
    result=dict(role='train_diagnostic',patients=patients,complete=True,semantic_sample_forwards=134*8*4*3,training_rng_unchanged=True)
    write(out/'SUMMARY.json',result);return result


def group_snapshot(state):
    return [[p.detach().clone() for p in g['params']] for g in state.optimizer.param_groups]


def step_delta(state,before,step):
    return dict(step=step,groups={name:dict(delta_norm=sum(float((p.detach()-b).square().sum()) for p,b in zip(g['params'],old))**.5,parameter_norm=sum(float(p.detach().square().sum()) for p in g['params'])**.5) for name,g,old in zip(('q','eta','phi'),state.optimizer.param_groups,before)})
