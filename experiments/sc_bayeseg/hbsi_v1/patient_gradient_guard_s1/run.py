"""Target-stage BayeSeg feedback constrained by four source-patient risk gradients."""
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import h5py
import torch
from torch.nn import functional as F

from ..budgets import ResourceLedger
from ..config import Config
from ..data_adapter import confined
from ..feedback_c1.model import FeedbackModel, standard_risk_from_logits
from ..feedback_c1.run import evaluate, exact_unet
from ..fullfield_bayeseg_a1.run import batch, records, train
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write
from .projection import PatientCone


def flatten(grads, parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                      for p,g in zip(parameters,grads)])


def source_anchors(p, model, out):
    model.eval()
    parameters=list(model.unet.parameters())
    rows=records(p,'RUNMC','train_labeled')
    assert len(rows)==4
    anchors=[];audit=[]
    for row in rows:
        with h5py.File(confined(Path(p['data_root'])/'h5/v1',row['image_h5_relpath']),'r') as f:
            volume=torch.from_numpy(f['image'][...].copy())[:,None].double()
        with h5py.File(confined(Path(p['data_root'])/'h5/v1',row['label_h5_relpath']),'r') as f:
            labels=torch.from_numpy(f['label'][...].copy())
        assert len(volume)==len(labels)
        total=torch.zeros(sum(v.numel() for v in parameters),device='cuda',dtype=torch.float64)
        for image,label in zip(volume,labels):
            x=F.interpolate(image[None],(128,128),mode='bilinear',align_corners=False).cuda()
            y=F.interpolate(label[None,None].double(),(128,128),mode='nearest')[:,0].long().cuda()
            assert bool(((y==0)|(y==1)).all())
            task=standard_risk_from_logits(model.unet(x).logits,y)
            grads=torch.autograd.grad(task,parameters,allow_unused=True)
            total+=flatten(grads,parameters)/len(volume)
        norm=total.norm()
        assert torch.isfinite(norm) and norm>0
        anchors.append(total/norm)
        audit.append(dict(patient=row['patient_id'],slices=len(volume),
                          unnormalized_gradient_norm=float(norm)))
    assert sum(v['slices'] for v in audit)==80
    stacked=torch.stack(anchors)
    cone=PatientCone(stacked)
    torch.save(stacked.detach().cpu(),out/'PRIVATE_SOURCE_ANCHORS.pt')
    write(out/'PRIVATE_SOURCE_PATIENT_GRADIENTS.json',audit)
    return cone,dict(patients=4,slices=80,
                     unnormalized_norms=[v['unnormalized_gradient_norm'] for v in audit],
                     gram=cone.gram.tolist())


def main():
    root=Path(os.environ['RUN_ROOT'])
    p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='PATIENT_GRADIENT_GUARD_S1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=replace(Config(**p['config']),seed=20261023)
    ledger=ResourceLedger(root/'results',max_seconds=12600,max_updates=1600,
                          max_bytes=3_000_000_000)

    def status(stage,**extra):
        tmp=root/'STATUS.tmp'
        tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**extra),indent=2))
        tmp.replace(root/'STATUS.json')

    c1=Path(p['c1_root']);r1=Path(p['r1_root']);u=Path(p['unet_root'])
    source_checkpoint=c1/'results/Cut_source/PRIVATE_CHECKPOINT.pt'

    def fresh():
        model=FeedbackModel(cut=False).double().cuda()
        saved=torch.load(source_checkpoint,map_location='cpu',weights_only=True)
        assert saved['step']==1600
        model.load_state_dict(saved['model'])
        return model

    def score(arm,model,domain,split='val'):
        name=f'{arm}_{domain}'+('' if split=='val' else '_train')
        status('EVALUATE',arm=arm,domain=domain,split=split)
        with ledger.attempt(name,seconds=900,updates=0) as out:
            return evaluate(p,model,domain,split,out/'validation','mean')

    try:
        assert json.loads((c1/'EXIT.json').read_text())['exit_code']==0
        assert json.loads((c1/'SOURCE_GATE.json').read_text())['status']=='FAILED'
        assert json.loads((r1/'EXIT.json').read_text())['exit_code']==0
        assert json.loads((u/'EXIT.json').read_text())['exit_code']==0
        source=data(p,cfg,'RUNMC','L');target=data(p,cfg,'BMC','L')
        assert source.counts=={'L':80,'U':0} and target.counts=={'L':134,'U':0}
        assert cfg.height==cfg.width==128 and cfg.dtype=='float64' and cfg.outer_lr==3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=1200,updates=0) as out:
            full=fresh();guard=fresh()
            assert exact_unet(full,u/'results/U1600/PRIVATE_CHECKPOINT.pt')==0
            assert all(torch.equal(full.state_dict()[k],guard.state_dict()[k])
                       for k in full.state_dict())
            cone,anchor_audit=source_anchors(p,full,out)
            # A deliberately conflicting vector must be corrected by the same solver.
            synthetic,synthetic_audit=cone.project(-cone.anchors[0])
            assert synthetic_audit['changed'] and min(synthetic_audit['after'])>=-2e-10
            image,label=batch(target,0,263000)
            full.train();guard.train()
            seed=54000000+(263000-163000)*10000
            ftask,fvar,flogits,fd=full.objectives(image,label,
                torch.Generator(device='cuda').manual_seed(seed))
            gtask,gvar,glogits,gd=guard.objectives(image,label,
                torch.Generator(device='cuda').manual_seed(seed))
            for a,b in ((ftask,gtask),(fvar,gvar),(flogits,glogits)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            for key in ('shape','appearance','z','probability','logq'):
                torch.testing.assert_close(fd[key],gd[key],rtol=0,atol=0)
            up=list(full.unet.parameters())
            tgrad=flatten(torch.autograd.grad(ftask,up,retain_graph=True,allow_unused=True),up)
            ggrad=flatten(torch.autograd.grad(.01*fvar,up,retain_graph=True,allow_unused=True),up)
            projected,projection_audit=cone.project(ggrad)
            (ftask+.01*fvar).backward()
            actual=flatten([v.grad for v in up],up)
            torch.testing.assert_close(actual,tgrad+ggrad,rtol=0,atol=1e-9)
            (gtask+.01*gvar).backward()
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in guard.parameters())
            assert all(torch.equal(full.state_dict()[k],guard.state_dict()[k])
                       for k in full.state_dict())
            write(root/'PREFLIGHT.json',dict(status='PASS',formal_updates=0,
                source_C1_Cut_step=1600,U_exact=True,shared_state_loss_draws=True,
                source_anchors=anchor_audit,synthetic_projection_changed=True,
                first_target_projection=projection_audit,
                full_gradient_decomposition_exact=True,
                full_backward_finite=True,state_restored=True))
            del full,guard,source,synthetic,projected

        cells={}
        full=fresh();optimizer=torch.optim.Adam(full.parameters(),lr=cfg.outer_lr)
        status('TRAIN_TARGET',arm='Full')
        train(full,optimizer,target,263000,800,'Full',ledger,3500)
        sealed=torch.load(r1/'results/SwitchJoint/PRIVATE_CHECKPOINT.pt',
                          map_location='cpu',weights_only=True)
        assert sealed['step']==800
        state=full.state_dict()
        assert set(state)==set(sealed['model'])
        full_error=max(float((state[k].detach().cpu()-sealed['model'][k]).abs().max())
                       for k in state)
        assert full_error==0,full_error
        write(root/'FULL_MATCH.json',dict(status='PASS',max_tensor_error=full_error,
                                          reference='R1_SwitchJoint_target800'))
        cells['Full_BMC_train']=score('Full',full,'BMC','train_labeled')
        for domain in ('RUNMC','BMC'):
            cells[f'Full_{domain}']=score('Full',full,domain)
        del full,optimizer

        guard=fresh();optimizer=torch.optim.Adam(guard.parameters(),lr=cfg.outer_lr)
        parameters=list(guard.unet.parameters())
        anchors=torch.load(root/'results/preflight/PRIVATE_SOURCE_ANCHORS.pt',
                           map_location='cpu',weights_only=True).cuda()
        cone=PatientCone(anchors)
        changed=0;violations=[0]*4;sum_correction=0.;max_violation=0.
        status('TRAIN_TARGET',arm='PatientGuard')
        with ledger.attempt('PatientGuard',seconds=5000,updates=800) as out:
            def save(step):
                tmp=out/'checkpoint.tmp'
                torch.save(dict(model=guard.state_dict(),optimizer=optimizer.state_dict(),
                                step=step),tmp)
                tmp.replace(out/f'PRIVATE_CHECKPOINT_{step}.pt')
            save(0)
            guard.train()
            with (out/'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
                for step in range(800):
                    ledger.check()
                    image,label=batch(target,step,263000)
                    generator=torch.Generator(device='cuda').manual_seed(
                        54000000+(263000-163000)*10000+step)
                    optimizer.zero_grad(set_to_none=True)
                    task,var,_,d=guard.objectives(image,label,generator)
                    assert torch.isfinite(task+.01*var)
                    task.backward(retain_graph=True)
                    task_grad=flatten([v.grad for v in parameters],parameters)
                    optimizer.zero_grad(set_to_none=True)
                    (.01*var).backward()
                    gen_grad=flatten([v.grad for v in parameters],parameters)
                    projected,audit=cone.project(gen_grad)
                    new_grad=task_grad+projected
                    offset=0
                    for v in parameters:
                        size=v.numel()
                        v.grad=new_grad[offset:offset+size].reshape_as(v).clone()
                        offset+=size
                    assert offset==new_grad.numel()
                    assert all(v.grad is None or torch.isfinite(v.grad).all()
                               for v in guard.parameters())
                    for i,dot in enumerate(audit['before']):
                        violations[i]+=int(dot < -1e-10)
                    max_violation=max(max_violation,max(0.,-min(audit['after'])))
                    changed+=int(audit['changed'])
                    sum_correction+=audit['correction_norm']
                    optimizer.step()
                    log.write(json.dumps(dict(step=step+1,task=float(task.detach()),
                        bound_per_pixel=float((-var).detach()),
                        shape_logsd=float(d['shape_logsd'].detach().mean()),
                        z_logsd=float(d['z_logsd'].detach().mean()),
                        projection_changed=audit['changed'],
                        correction_norm=audit['correction_norm'],
                        generator_norm=audit['generator_norm'],
                        min_after_dot=min(audit['after'])))+'\n')
                    log.flush()
                    if step+1 in (400,800):save(step+1)
            write(out/'RECEIPT.json',dict(status='COMPLETE',steps=800,
                source_patient_anchor_count=4,projection_changed_steps=changed,
                source_anchor_negative_dot_steps=violations,
                mean_correction_norm=sum_correction/800,
                max_post_projection_violation=max_violation,
                source_images_read_during_target_updates=False))
        cells['PatientGuard_BMC_train']=score('PatientGuard',guard,'BMC','train_labeled')
        for domain in ('RUNMC','BMC'):
            cells[f'PatientGuard_{domain}']=score('PatientGuard',guard,domain)
        write(root/'RESULTS.json',dict(cells=cells,formal_updates=1600,
            source_training_reused='C1_Cut_source1600',target_labeled_slices=134,
            target_unlabeled_used=False,source_images_read_during_target_updates=False,
            projection_changed_steps=changed,max_post_projection_violation=max_violation))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':
    main()
