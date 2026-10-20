"""Matched H training with one predeclared frozen G3 bias source."""
import json
import os
import time
import traceback
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from lcrseg.data import H5LabeledDataset
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.patient_image_interface import PatientImageInterface
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.r_start import cpu,write,append,save,semantic_loss
from sc_bayeseg.window_v1.patient_validation import Schedule,phase_metrics,aggregate
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();active={}
    costs=dict(optimizer=0,reverse=0,training_encoder_images=0,train_theta=0,train_g_forward=0,train_g_image_forward=0,
      validation_encoder_images=0,validation_g_forward=0,validation_g_image_forward=0,validation_theta=0,validation_label_phase_reads=0)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(root/'CONTEXT.json',s);storage(s,reserve=3_000_000_000)
        load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
        parent=Path(s['patient_control']);closure=Path(s['g_bias_source'])
        assert json.loads((parent/'AUDIT.json').read_text())['status']=='PASS'
        assert json.loads((closure/'AUDIT.json').read_text())['status']=='PASS'
        records=json.loads((parent/'PRIVATE_EVALUATION_MAP.json').read_text())
        assert len(records)==46 and len({r['patient_id'] for r in records})==23
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None);schedule=Schedule(ds)
        assert not set(schedule.patients)&{r['patient_id'] for r in records}
        sourcefiles=[Path(s['checkpoint']),Path(s['H_initial']),parent/'PRETRAIN/S5000.pt',closure/'RESULTS.json']
        meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in sourcefiles}
        shared=load(sourcefiles[2]);g=load(sourcefiles[0]);assert g['cursor']==8675
        delta=json.loads(sourcefiles[3].read_text())['arms']['ZERO']['readouts']['G3']['delta']
        assert len(delta)==3
        encoder=build_model().cuda();encoder.load_state_dict(shared['model'])
        generator=SpatialResidualModel().cuda();generator.load_state_dict(g['model'])
        with torch.no_grad():generator.anatomy[2].mean_b.add_(torch.tensor(delta,device='cuda',dtype=generator.anatomy[2].mean_b.dtype))
        fixed=cpu(generator.state_dict())
        assert all(same(v,g['model'][k]) for k,v in fixed.items() if k!='anatomy.2.mean_b')
        model=PatientImageInterface(encoder,generator).cuda();hinit=load(sourcefiles[1])['model']
        for prefix,module in (('hc.',model.hc),('hr.',model.hr)):
            module.load_state_dict({k[len(prefix):]:v for k,v in hinit.items() if k.startswith(prefix)})
        restore_rng_state(shared['rng']);params=model.head_parameters()
        opt=torch.optim.Adam(params,lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        active=dict(model=model,opt=opt,step=0)
        def post(optimizer,args,kwargs):
            costs['optimizer']+=1;active['step']=costs['optimizer'];assert costs['optimizer']<=5000
            append(root/'physical_updates.jsonl',dict(index=costs['optimizer'],stage='CONDITIONAL_G3',step=costs['optimizer']))
        handle=opt.register_step_post_hook(post)
        def snapshot(step):return dict(stage='CONDITIONAL_G3',step=step,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng_state(),costs=costs)
        save(root/'S0000.pt',snapshot(0))
        for step in range(1,5001):
            assert time.time()-s['original_started_unix']<85500
            x,y,ids,flips=schedule.batch(5000+step);x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
            loss,_=semantic_loss(generator,model.posterior(x),y,step,costs);costs['training_encoder_images']+=2
            assert torch.isfinite(loss);loss.backward();costs['reverse']+=1
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
            norm=float(nn.utils.clip_grad_norm_(params,1.));opt.step();generator.clear_draw()
            append(root/'steps.jsonl',dict(stage='CONDITIONAL_G3',step=step,absolute_schedule_step=5000+step,indices=ids,flips=flips,loss=float(loss.detach()),gradient_norm_before_clip=norm))
            if step%100==0:write(root/'STATUS.json',dict(status='TRAINING',step=step,costs=costs))
            if step%1000==0:
                assert same(shared['model'],model.backbone.state_dict()) and same(fixed,generator.state_dict())
                storage(s,reserve=2_000_000_000);save(root/f'S{step:04}.pt',snapshot(step))
        handle.remove();assert costs['optimizer']==costs['reverse']==5000
        assert all(int(v['step'])==5000 for v in opt.state_dict()['state'].values());active={};model.eval()
        folder=root/'predictions';folder.mkdir(exist_ok=False)
        for index,row in enumerate(records):
            assert time.time()-s['original_started_unix']<85500
            with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:
                volume=f['image'][...];spacing=json.loads(f.attrs['processed_spacing'])
            output=[]
            for offset in range(0,len(volume),2):
                image=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda()
                p=model.predict_probabilities(image,seed=620163+1000*index+offset//2,theta_draws=4,local_draws=4)
                assert torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6
                output.append(p.cpu());costs['validation_encoder_images']+=len(image)
                costs['validation_g_forward']+=16;costs['validation_g_image_forward']+=16*len(image);costs['validation_theta']+=4
            save(folder/f'CASE{index:03}.pt',dict(probability=torch.cat(output),spacing=spacing))
            storage(s,reserve=500_000_000);write(root/'STATUS.json',dict(status='PREDICTING',phase_index=index,costs=costs))
        write(root/'PREDICTIONS_SEALED.json',dict(phases=46,costs=costs,time_unix=time.time()))
        rows=[]
        for index,row in enumerate(records):
            p=load(folder/f'CASE{index:03}.pt')
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            costs['validation_label_phase_reads']+=1
            rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p['probability'],y,p['spacing'])))
            write(root/'STATUS.json',dict(status='SCORING',phase_index=index,costs=costs))
        agg=aggregate(rows);controls=json.loads((parent/'RESULTS.json').read_text())['aggregates'];paired={}
        b=np.array([v['macro_Dice'] for v in agg['patients']])
        for name,control in controls.items():
            assert [v['patient'] for v in control['patients']]==[v['patient'] for v in agg['patients']]
            a=np.array([v['macro_Dice'] for v in control['patients']]);rng=np.random.default_rng(163)
            boot=np.mean((b-a)[rng.integers(0,23,size=(10000,23))],axis=1)
            paired[name]=dict(mean_Dice_difference=float((b-a).mean()),patient_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in meta.items())
        assert same(shared['model'],model.backbone.state_dict()) and same(fixed,generator.state_dict())
        save(root/'PRIVATE_PHASE_RESULTS.pt',rows)
        write(root/'RESULTS.json',dict(status='COMPLETE',aggregate=agg,paired=paired,costs=costs,source_frozen=True,
          validation_predictions_before_labels=True,test_access=False,G_source='ZERO/G3; fixed source ordering, no patient validation selection',
          phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:save(root/'failure_partial.pt',dict(step=active['step'],model=active['model'].state_dict(),optimizer=active['opt'].state_dict(),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
