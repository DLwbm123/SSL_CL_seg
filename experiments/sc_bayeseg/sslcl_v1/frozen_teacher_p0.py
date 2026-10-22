"""Pre-registered zero-update teacher correctness audit on current training L."""
import json, os, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import Schedule
from .model import initialize
from .train import setup
from sc_bayeseg.window_v1.r_start import write


def weak(x, style):
    return (x-x.mean((2,3),keepdim=True))/x.std((2,3),unbiased=False,keepdim=True).clamp_min(1e-4)*style[1]+style[0]


def score(q, y, mask):
    conf, pred=q.max(1); wrong=pred!=y
    return dict(nll=float(-q.gather(1,y[:,None]).squeeze(1).clamp_min(1e-8).log()[mask].sum()),
                brier=float((q[:,1]-y.float()).square()[mask].sum()), error=float(wrong[mask].sum()),
                confident_wrong=float((wrong & (conf>=.9))[mask].sum()))


@torch.no_grad()
def run(s):
    setup(); out=Path(s['output']); out.mkdir(exist_ok=False)
    L=Schedule(s['data'],'BMC',seed=s['schedule_seed'],dataset='prostate')
    old=initialize(s,'U'); ck=torch.load(s['old_checkpoint'],map_location='cpu',weights_only=False)
    assert ck['step']==5000 and ck['family']=='U'; old.load_state_dict(ck['model']); old.eval().requires_grad_(False)
    teachers={}
    for step in (1000,3000,5000):
        ck=torch.load(str(Path(s['ema_dir'])/f'S{step}.pt'),map_location='cpu',weights_only=False)
        assert ck['step']==step and ck['family']=='U'
        t=initialize(s,'U');t.load_state_dict(ck['teacher']);t.eval().requires_grad_(False);teachers[step]=t
    patients=[]; forwards=0; slices=0
    for patient in L.patients:
        sums=defaultdict(lambda:defaultdict(float)); counts=defaultdict(int)
        ids=[i for phase in L.groups[patient].values() for i in phase]
        for i in ids:
            assert time.time()-s['original_started_unix']<85200
            sample=L.ds.samples[i]
            x=F.interpolate(L.ds._read_image(sample)[None],(128,128),mode='bilinear',align_corners=False).cuda()
            predictions={}
            for name,z in [('original',x),('current',weak(x,s['styles']['BMC'])),('history',weak(x,s['styles']['RUNMC']))]:
                o=old.probability(z);forwards+=1
                for step,t in teachers.items():
                    e=t.probability(z);forwards+=1
                    for q in (o,e):
                        assert torch.isfinite(q).all() and torch.allclose(q.sum(1),torch.ones_like(q[:,0]),atol=1e-5)
                    predictions[name,step]=(o,e,(o+e)/2)
            # No label is read until every teacher prediction for this image exists.
            y=F.interpolate(L.ds._read_label(sample)[None,None].float(),(128,128),mode='nearest')[:,0].long().cuda()
            for layer,mask in [('fg',y==1),('bg',y==0)]:
                counts[layer]+=int(mask.sum())
                if not mask.any():continue
                for (name,step),(o,e,m) in predictions.items():
                    key=f'{name}/{step}/{layer}'
                    for model,q in [('old',o),('ema',e),('mix',m)]:
                        for metric,v in score(q,y,mask).items():sums[key][model+'_'+metric]+=v
                    sums[key]['old_correct_ema_wrong']+=float(((o.argmax(1)==y)&(e.argmax(1)!=y))[mask].sum())
                    sums[key]['ema_correct_old_wrong']+=float(((e.argmax(1)==y)&(o.argmax(1)!=y))[mask].sum())
            for (name,step),qs in predictions.items():
                for model,q in zip(('old','ema','mix'),qs):sums[f'{name}/{step}/coverage'][model]+=float((q.argmax(1)==1).sum())
            slices+=1
        assert counts['fg']>0 and counts['bg']>0
        row={}
        for key,v in sums.items():
            layer=key.split('/')[-1];den=counts['fg'] if layer=='coverage' else counts[layer]
            row[key]={metric:val/den for metric,val in v.items()}
        patients.append(row)
    means={k:{m:float(np.mean([p[k][m] for p in patients])) for m in patients[0][k]} for k in patients[0]}
    failures=[]
    for name in ('current','history'):
        fg=means[f'{name}/1000/fg'];bg=means[f'{name}/1000/bg'];coverage=means[f'{name}/1000/coverage']['old']
        if 1-fg['old_error']<.25:failures.append(name+':old_recall_below_.25')
        if not .1<=coverage<=4:failures.append(name+':old_coverage_outside_.1_4')
        for layer,v in [('fg',fg),('bg',bg)]:
            if v['mix_nll']-v['ema_nll']>.05:failures.append(name+':'+layer+':mix_nll_increment_above_.05')
            if v['mix_confident_wrong']-v['ema_confident_wrong']>.02:failures.append(name+':'+layer+':mix_confident_wrong_increment_above_.02')
    if means['history/1000/fg']['old_correct_ema_wrong']<.01:failures.append('history:fg_complement_below_.01')
    assert all(not t.training and all(not p.requires_grad and p.grad is None for p in t.parameters()) for t in [old,*teachers.values()])
    write(out/'METRICS.json',dict(status='COMPLETE',gate='FAIL' if failures else 'PASS',failures=failures,seed=s['schedule_seed'],patients=len(patients),slices=slices,forward_calls=forwards,optimizer_updates=0,means=means,patient_metrics=patients,gt_read_after_predictions=True))
    write(out/'EXIT.json',dict(exit_code=0))

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
