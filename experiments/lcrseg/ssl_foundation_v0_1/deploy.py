"""Fresh process loads only the final student and reproduces evaluator predictions."""
import argparse, time
from pathlib import Path
import numpy as np
import torch
from .core import *
from .freeze import verify
from experiments.lcrseg.di_dmpa_jascl.modeling import _official_probabilistic_classifier

def deploy(root,data,reference,device,*,qualification=False,expected=None,shape=(384,384)):
    root=Path(root);source='qualification' if qualification else verify();p=torch.load(root/'deploy_student.pt',map_location=device,weights_only=False)
    assert p['source']==source and p['complete']
    # Meta construction avoids allocating a second initialized full state.
    with rng(device,'deploy_import'):_official_probabilistic_classifier(reference,upstream_path=UPSTREAM_PATH)
    with torch.device('meta'):
        m=build_lcrseg_unet_jascl_model(reference,upstream_path=UPSTREAM_PATH,input_channels=3,num_classes=3)
    m.load_state_dict(p.pop('student'),assign=True);m.eval();m.decoder.conv_logit.grad_update.requires_grad_(False);precision()
    from experiments.lcrseg.single_teacher_scd_v0_1.engine import audit_live_models
    assert audit_live_models(1)==1 and state_hash(m)==p['student_hash']
    # No additional GT reads: use stripped val image-only records for deployment.
    ds=DomainData(data,p['domain'],'val',evaluator=True,shape=shape,**({} if expected is None else dict(expected=expected)))
    ds._dataset.role='train_unlabeled'
    ds._dataset.rows=[{k:v for k,v in r.items() if k in ('case_id','image_h5_relpath','image_sha256')} for r in ds._dataset.rows]
    predictions=np.load(root/f"eval{p['epoch']}"/'evaluator_only_deployment_predictions.npz')['predictions'];count=0
    with torch.no_grad():
        for i in range(len(ds)):
            pred=m(ds[i]['image'][None].to(device),stochastic_classifier=False)[0].argmax(1)[0].cpu().numpy()
            if not np.array_equal(pred,predictions[i]):raise RuntimeError('student deployment mismatch')
            count+=1
    result=dict(status='PASS',source=source,models=1,student_hash=p['student_hash'],cases=count,additional_gt_reads=0,deterministic_head=True)
    write_json(root/'deployment.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','data','reference'):p.add_argument('--'+k,required=True)
    deploy(**vars(p.parse_args()),device=torch.device('cuda:0'))
