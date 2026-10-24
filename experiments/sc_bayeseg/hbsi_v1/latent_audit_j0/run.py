"""Zero-update patient-aggregated audit of shape and appearance semantics."""
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sc_bayeseg.model import sar, student_log_prob
from ..budgets import ResourceLedger
from ..config import Config
from ..feedback_c1.model import FeedbackModel
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write


def paircorr(a,b):
    if a.std()==0 or b.std()==0:
        return None
    return float(np.corrcoef(a,b)[0,1])


def main():
    root=Path(os.environ['RUN_ROOT'])
    p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='LATENT_AUDIT_J0'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=replace(Config(**p['config']),seed=20261023)
    ledger=ResourceLedger(root/'results',max_seconds=3600,max_updates=0,
                          max_bytes=1_000_000_000)

    def status(stage,**extra):
        tmp=root/'STATUS.tmp'
        tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**extra),indent=2))
        tmp.replace(root/'STATUS.json')

    try:
        c1=Path(p['c1_root']);h1=Path(p['h1_root'])
        assert json.loads((c1/'EXIT.json').read_text())['exit_code']==0
        assert json.loads((h1/'EXIT.json').read_text())['exit_code']==0
        source=data(p,cfg,'RUNMC','L')
        assert source.counts=={'L':80,'U':0}
        records=source.records['L']
        anonymous={name:i+1 for i,name in enumerate(sorted({str(v['image']) for v in records}))}
        assert len(anonymous)==4
        assert cfg.height==cfg.width==128 and cfg.dtype=='float64'
        status('PREFLIGHT')
        write(root/'PREFLIGHT.json',dict(status='PASS',production_updates=0,
             source_L_slices=len(records),source_L_patients=4,
             BMC_and_U_labels_not_accessed=True,read_only_states=True))
        results={}
        for arm in ('C1','H1_Shape'):
            status('AUDIT',arm=arm)
            with ledger.attempt(arm,seconds=1200,updates=0) as out:
                model=FeedbackModel(cut=True).double().cuda()
                if arm=='C1':
                    saved=torch.load(c1/'results/Cut_source/PRIVATE_CHECKPOINT.pt',
                                     map_location='cpu',weights_only=True)
                else:
                    saved=torch.load(h1/'results/Shape/PRIVATE_CHECKPOINT_1600.pt',
                                     map_location='cpu',weights_only=True)
                assert saved['step']==1600
                state=saved['model']
                if arm=='H1_Shape':
                    state={k.removeprefix('base.'):v for k,v in state.items()
                           if k.startswith('base.')}
                model.load_state_dict(state)
                model.eval()
                acc={i:dict(shape=[],appearance=[],label=[],reconstruction_mse=[],
                            conditional_logdensity_delta_per_pixel=[],slices=0)
                     for i in range(1,5)}
                with torch.no_grad():
                    for ordinal,record in enumerate(records):
                        ledger.check()
                        idx=anonymous[str(record['image'])]
                        x=source.read(record,'image').double()[None,None]
                        y=source.read(record,'label').double()[None,None]
                        x=F.interpolate(x,(128,128),mode='bilinear',align_corners=False).cuda()
                        y=F.interpolate(y,(128,128),mode='nearest').cuda()
                        assert set(y.unique().tolist())<=set((0.,1.))
                        shape=model.shape_posterior(x)[:,:1]
                        appearance=model.appearance_posterior(x)[:,:1]
                        output=model.unet(x)
                        generator=torch.Generator(device='cuda').manual_seed(84000000+ordinal)
                        d=model.draw(x,generator,output)
                        truth=torch.cat((1-y,y),1)
                        z=student_log_prob(sar(d['shape']-model.shape_mean(d['probability']),.9),.2)
                        label=student_log_prob(sar(d['shape']-model.shape_mean(truth),.9),.2)
                        group=acc[idx]
                        group['shape'].append(shape.cpu().numpy().ravel())
                        group['appearance'].append(appearance.cpu().numpy().ravel())
                        group['label'].append(y.cpu().numpy().ravel())
                        group['reconstruction_mse'].append(float((x-shape-appearance).square().mean()))
                        group['conditional_logdensity_delta_per_pixel'].append(float((label-z)/(128*128)))
                        group['slices']+=1
                patients=[]
                for idx in range(1,5):
                    group=acc[idx]
                    s=np.concatenate(group['shape']);a=np.concatenate(group['appearance'])
                    y=np.concatenate(group['label'])
                    cs=paircorr(s,y);ca=paircorr(a,y);csa=paircorr(s,a)
                    patients.append(dict(index=idx,slices=group['slices'],pixels=len(y),
                        foreground_fraction=float(y.mean()),
                        shape_label_pearson=cs,appearance_label_pearson=ca,
                        abs_shape_label_pearson=abs(cs) if cs is not None else None,
                        abs_appearance_label_pearson=abs(ca) if ca is not None else None,
                        shape_appearance_pearson=csa,
                        shape_mean_std=float(s.std()),appearance_mean_std=float(a.std()),
                        mean_reconstruction_mse=float(np.mean(group['reconstruction_mse'])),
                        mean_log_p_S_given_Y_minus_Z_per_pixel=float(np.mean(
                            group['conditional_logdensity_delta_per_pixel']))))
                assert sum(v['slices'] for v in patients)==80
                assert all(v['slices']>0 and v['foreground_fraction']>0 for v in patients)
                write(out/'ANONYMOUS_PATIENTS.json',patients)
                results[arm]=patients
                del model
        write(root/'RESULTS.json',dict(status='COMPLETE',patients=results,
             production_updates=0,source_L_slices=80,source_L_patients=4,
             BMC_and_U_labels_not_accessed=True))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':
    main()
