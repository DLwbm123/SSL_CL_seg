"""Matched end-to-end U-Net and conditional G training from shared pretraining."""
import copy
import json
import math
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.data.h5_dataset import load_training_records
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.patient_image_interface import PatientImageInterface
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.r_start import cpu,write,append,save,semantic_loss
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage


from sc_bayeseg.window_v1.patient_validation import DirectHead,Schedule,phase_metrics,aggregate

class EndToEndDirect(DirectHead):
    def logits(self,image):
        image=F.interpolate(image,(128,128),mode='bilinear',align_corners=False)
        return self.head(self.backbone(image).decoder_features['dec1'])

class EndToEndG(PatientImageInterface):
    def posterior(self,image):
        if image.ndim!=4 or image.shape[1]!=1:raise ValueError('image must be [B,1,H,W]')
        resized=F.interpolate(image,(128,128),mode='bilinear',align_corners=False)
        feature=self.backbone(resized).decoder_features['dec1']
        b,c,h,w=feature.shape;assert (c,h,w)==(16,128,128)
        cm,cs=self.hc(feature.mean((2,3))).chunk(2,1)
        rm,rs=self.hr(feature.reshape(b,16,16,8,16,8).mean((3,5))).chunk(2,1)
        return cm,-4+3*torch.tanh(cs/3),rm,-4+3*torch.tanh(rs/3)


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();active={}
    costs=dict(optimizer=0,reverse=0,training_encoder_images=0,train_theta=0,train_g_forward=0,train_g_image_forward=0,
      validation_encoder_images=0,validation_g_forward=0,validation_g_image_forward=0,validation_theta=0,validation_label_phase_reads=0)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(root/'CONTEXT.json',s);storage(s,reserve=4_000_000_000)
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        assert all(r['cohort']=='canonical320' and r['primary_20pct_split']=='train_labeled' for r in ds.records)
        schedule=Schedule(ds)
        records=load_training_records(Path(s['data']),seed=0,dataset='mnms',sites=['Siemens'],roles=['val'])
        assert len(records)==46 and len({r['patient_id'] for r in records})==23
        assert not set(schedule.patients)&{r['patient_id'] for r in records}
        records=sorted(records,key=lambda r:(r['patient_id'],r['case_id']))
        write(root/'PRIVATE_EVALUATION_MAP.json',records)
        sourcefiles=[Path(s['checkpoint']),Path(s['unet_initial']),Path(s['H_initial']),Path(s['patient_control'])/'PRETRAIN/S5000.pt',Path(s['patient_control'])/'RESULTS.json']
        meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in sourcefiles}
        sourceG=torch.load(sourcefiles[0],map_location='cpu',weights_only=False);assert sourceG['cursor']==8675
        hinit=torch.load(sourcefiles[2],map_location='cpu',weights_only=False)['model']
        shared=torch.load(Path(s['patient_control'])/'PRETRAIN/S5000.pt',map_location='cpu',weights_only=False)
        assert json.loads((Path(s['patient_control'])/'AUDIT.json').read_text())['status']=='PASS'
        models={};stage_results={}
        for stage in ('DIRECT_E2E','CONDITIONAL_G_E2E'):
            out=root/stage;out.mkdir(exist_ok=False)
            encoder=build_model().cuda();encoder.load_state_dict(shared['model'])
            if stage=='DIRECT_E2E':model=EndToEndDirect(encoder).cuda()
            else:
                generator=SpatialResidualModel().cuda();generator.load_state_dict(sourceG['model'])
                model=EndToEndG(encoder,generator).cuda()
                model.hc.load_state_dict({k[3:]:v for k,v in hinit.items() if k.startswith('hc.')})
                model.hr.load_state_dict({k[3:]:v for k,v in hinit.items() if k.startswith('hr.')})
            model.backbone.requires_grad_(True);model.backbone.segmentation_head.requires_grad_(False)
            params=tuple(p for p in model.parameters() if p.requires_grad)
            assert sum(p.numel() for p in params)==(482500 if stage=='DIRECT_E2E' else 484576)
            if shared is not None:restore_rng_state(shared['rng'])
            opt=torch.optim.Adam(params,lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
            count=[0];active=dict(stage=stage,model=model,opt=opt,step=0)
            def post(optimizer,args,kwargs):
                count[0]+=1;costs['optimizer']+=1;active['step']=count[0]
                assert count[0]<=5000 and costs['optimizer']<=10000
                append(root/'physical_updates.jsonl',dict(index=costs['optimizer'],stage=stage,step=count[0]))
            handle=opt.register_step_post_hook(post)
            def snapshot(step):return dict(stage=stage,step=step,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng_state(),costs=costs)
            save(out/'S0000.pt',snapshot(0))
            for step in range(1,5001):
                assert time.time()-s['original_started_unix']<85500
                absolute=5000+step
                x,y,ids,flips=schedule.batch(absolute);x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
                if stage=='DIRECT_E2E':loss=F.cross_entropy(model.logits(x),y,reduction='none').mean()
                else:loss,_=semantic_loss(model.generator,model.posterior(x),y,step,costs)
                costs['training_encoder_images']+=2
                assert torch.isfinite(loss);loss.backward();costs['reverse']+=1
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
                norm=float(nn.utils.clip_grad_norm_(params,1.));opt.step()
                if stage=='CONDITIONAL_G_E2E':model.generator.clear_draw()
                append(root/'steps.jsonl',dict(stage=stage,step=step,absolute_schedule_step=absolute,indices=ids,flips=flips,loss=float(loss.detach()),gradient_norm_before_clip=norm))
                if step%100==0:write(root/'STATUS.json',dict(status='TRAINING',stage=stage,step=step,costs=costs))
                if step%1000==0:
                    storage(s,reserve=4_000_000_000);save(out/f'S{step:04}.pt',snapshot(step))
            handle.remove();assert count[0]==5000 and all(int(v['step'])==5000 for v in opt.state_dict()['state'].values())
            stage_results[stage]=dict(optimizer=5000,trainable_parameters=sum(v.numel() for v in params))
            assert not same(shared['model'],model.backbone.state_dict())
            if stage=='CONDITIONAL_G_E2E':assert same(sourceG['model'],model.generator.state_dict())
            models[stage]=model.eval()
            active={};del opt;torch.cuda.empty_cache()
        assert costs['optimizer']==costs['reverse']==10000
        assert costs['training_encoder_images']==20000 and costs['train_theta']==10000 and costs['train_g_forward']==20000
        for stage,model in models.items():
            folder=root/stage/'predictions';folder.mkdir(exist_ok=False)
            for case_index,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500;file=Path(s['data'])/'h5/v1'/row['image_h5_relpath']
                with h5py.File(file,'r') as f:
                    volume=f['image'][...];spacing=json.loads(f.attrs['processed_spacing'])
                assert volume.ndim==3 and volume.shape[1:]==(384,384)
                result=[]
                for offset in range(0,len(volume),2):
                    image=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda()
                    p=model.predict_probabilities(image,seed=620163+1000*case_index+offset//2,theta_draws=4,local_draws=4)
                    assert torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6
                    result.append(p.cpu());costs['validation_encoder_images']+=len(image)
                    if stage=='CONDITIONAL_G_E2E':
                        costs['validation_g_forward']+=16;costs['validation_g_image_forward']+=16*len(image);costs['validation_theta']+=4
                save(folder/f'CASE{case_index:03}.pt',dict(probability=torch.cat(result),spacing=spacing))
                storage(s,reserve=1_000_000_000)
                write(root/'STATUS.json',dict(status='PREDICTING',stage=stage,phase_index=case_index,costs=costs))
        assert costs['validation_label_phase_reads']==0
        write(root/'PREDICTIONS_SEALED.json',dict(stages=list(models),phases_each=46,costs=costs,time_unix=time.time()))
        all_records={};aggregates={}
        for stage in models:
            phase_rows=[]
            for index,row in enumerate(records):
                assert (root/'PREDICTIONS_SEALED.json').exists()
                p=torch.load(root/stage/'predictions'/f'CASE{index:03}.pt',map_location='cpu',weights_only=False)
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                costs['validation_label_phase_reads']+=1
                phase_rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p['probability'],y,p['spacing'])))
                write(root/'STATUS.json',dict(status='SCORING',stage=stage,phase_index=index,costs=costs))
            all_records[stage]=phase_rows;aggregates[stage]=aggregate(phase_rows)
        assert costs['validation_label_phase_reads']==92
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==st for p,st in meta.items())
        frozen=json.loads((Path(s['patient_control'])/'RESULTS.json').read_text())['aggregates']
        arrays={name:np.array([v['macro_Dice'] for v in agg['patients']]) for name,agg in aggregates.items()}
        controls={name:np.array([v['macro_Dice'] for v in agg['patients']]) for name,agg in frozen.items()}
        for name in ('DIRECT','CONDITIONAL_G'):
            assert [v['patient'] for v in frozen[name]['patients']]==[v['patient'] for v in aggregates['DIRECT_E2E']['patients']]
        contrasts=dict(G_minus_DIRECT=arrays['CONDITIONAL_G_E2E']-arrays['DIRECT_E2E'],
          DIRECT_unfreeze=arrays['DIRECT_E2E']-controls['DIRECT'],G_unfreeze=arrays['CONDITIONAL_G_E2E']-controls['CONDITIONAL_G'])
        contrasts['unfreeze_interaction']=contrasts['G_unfreeze']-contrasts['DIRECT_unfreeze'];paired={}
        for name,difference in contrasts.items():
            rng=np.random.default_rng(163);boot=np.mean(difference[rng.integers(0,23,size=(10000,23))],axis=1)
            paired[name]=dict(mean_Dice_difference=float(difference.mean()),patient_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
        save(root/'PRIVATE_PHASE_RESULTS.pt',all_records)
        write(root/'RESULTS.json',dict(status='COMPLETE',stages=stage_results,aggregates=aggregates,paired=paired,
          costs=costs,source_frozen=True,validation_predictions_before_labels=True,test_access=False,
          phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:save(root/'failure_partial.pt',dict(stage=active['stage'],step=active['step'],model=active['model'].state_dict(),optimizer=active['opt'].state_dict(),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
