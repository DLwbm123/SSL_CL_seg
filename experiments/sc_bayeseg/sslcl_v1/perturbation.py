"""Image-only, zero-update audit of paired style perturbations on target U."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import Schedule,view
from .model import initialize
from .train import setup
from sc_bayeseg.window_v1.r_start import write

@torch.no_grad()
def run(s):
    setup();out=Path(s['output']);out.mkdir(exist_ok=False)
    model=initialize(s,s['family']);ck=torch.load(s['checkpoint'],map_location='cpu',weights_only=False)
    assert ck['step']==5000 and ck['family']==s['family'];model.load_state_dict(ck['model']);model.eval()
    schedule=Schedule(s['data'],'BMC',True,dataset='prostate');assert len(schedule.patients)==14
    current=s['styles']['BMC'];history=s['styles']['RUNMC'];rows=[];selected=0
    for pi,patient in enumerate(schedule.patients):
        assert time.time()-s['original_started_unix']<85200
        phases=schedule.groups[patient];assert len(phases)==1
        ids=next(iter(phases.values()));chosen=[ids[int(q*(len(ids)-1))] for q in (.25,.5,.75)]
        assert len(set(chosen))==3
        samples=[schedule.ds.samples[i] for i in chosen]
        assert all(not sample.row['label_h5_relpath'] for sample in samples)
        x=torch.cat([F.interpolate(schedule.ds._read_image(sample)[None],(128,128),mode='bilinear',align_corners=False) for sample in samples]).cuda()
        repeats=[]
        for repeat in range(4):
            step=1+pi*4+repeat;draw_seed=910163+pi*4+repeat
            torch.manual_seed(draw_seed);p=model.probability(x).clamp_min(1e-8)
            modes={}
            g=torch.Generator(device=x.device).manual_seed(163000+step)
            selected+=int((torch.rand((len(x),5,1,1),device=x.device,generator=g)[:,0]<.5).sum())
            for name,strength in [('NONE',0.),('HALF',.5),('FULL',1.)]:
                z=view(x,current,history,step,history_strength=strength)
                torch.manual_seed(draw_seed);q=model.probability(z).clamp_min(1e-8)
                assert torch.isfinite(q).all()
                modes[name]=dict(KL=float((p*(p.log()-q.log())).sum(1).mean()),flip_fraction=float((p.argmax(1)!=q.argmax(1)).float().mean()),foreground_probability_shift=float((q[:,1]-p[:,1]).mean()),foreground_probability_abs_change=float((q[:,1]-p[:,1]).abs().mean()),entropy=float(-(q*q.log()).sum(1).mean()))
            repeats.append(modes)
        rows.append({mode:{key:float(np.mean([r[mode][key] for r in repeats])) for key in repeats[0][mode]} for mode in repeats[0]})
    aggregate={mode:{key:dict(mean=float(np.mean([r[mode][key] for r in rows])),patient_sd=float(np.std([r[mode][key] for r in rows],ddof=1))) for key in rows[0][mode]} for mode in rows[0]}
    contrasts={mode+'-NONE':{key:dict(mean=float(np.mean(v:=[r[mode][key]-r['NONE'][key] for r in rows])),positive_patients=sum(x>0 for x in v),patient_sd=float(np.std(v,ddof=1))) for key in rows[0][mode]} for mode in ('HALF','FULL')}
    write(out/'METRICS.json',dict(status='COMPLETE',optimizer_updates=0,patients=14,slices_per_patient=3,repeats=4,history_selected_images=selected,total_image_views=168,metrics=aggregate,contrasts=contrasts,caveat='Prediction sensitivity, not accuracy. Same latent random draws paired across original/NONE/HALF/FULL. Target train U images only; no GT opened.'))
    write(out/'EXIT.json',dict(exit_code=0))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
