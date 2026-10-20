"""Patient-disjoint development validation with sealed image-only predictions."""
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


class DirectHead(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone.requires_grad_(False)
        self.head=copy.deepcopy(backbone.segmentation_head).requires_grad_(True)
    def head_parameters(self):return tuple(self.head.parameters())
    def logits(self,image):
        image=F.interpolate(image,(128,128),mode='bilinear',align_corners=False)
        with torch.no_grad():feature=self.backbone(image).decoder_features['dec1']
        return self.head(feature)
    @torch.no_grad()
    def predict_probabilities(self,image,**kwargs):
        p=self.logits(image).softmax(1);p=F.interpolate(p,image.shape[-2:],mode='bilinear',align_corners=False)
        return p/p.sum(1,keepdim=True)


class Schedule:
    def __init__(self,dataset):
        self.ds=dataset;self.groups=defaultdict(lambda:defaultdict(list))
        for i,s in enumerate(dataset.samples):self.groups[s.row['patient_id']][s.row['case_id']].append(i)
        self.patients=sorted(self.groups);assert len(self.patients)==10 and len(dataset)==230
    def batch(self,step):
        g=torch.Generator().manual_seed(163+100003*step);ids=[];flips=[]
        for j in torch.randperm(10,generator=g)[:2].tolist():
            cases=self.groups[self.patients[j]];keys=sorted(cases);assert len(keys)==2
            key=keys[int(torch.randint(2,(),generator=g))];indices=cases[key]
            ids.append(indices[int(torch.randint(len(indices),(),generator=g))]);flips.append(int(torch.randint(4,(),generator=g)))
        b=collate_labeled([self.ds[i] for i in ids])
        x=F.interpolate(b.image,(128,128),mode='bilinear',align_corners=False)
        y=F.interpolate(b.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        return torch.stack([flip(v,k) for v,k in zip(x,flips)]),torch.stack([flip(v,k) for v,k in zip(y,flips)]),ids,flips


def phase_metrics(p,y,spacing):
    pred=p.argmax(1).numpy();y=np.asarray(y);rows=[]
    assert pred.shape==y.shape and len(spacing)==3 and min(spacing)>0
    penalty=float(np.linalg.norm((np.array(y.shape)-1)*np.array(spacing)))
    for k in (1,2,3):
        a=pred==k;b=y==k;tp=int((a&b).sum());pn=int(a.sum());gn=int(b.sum())
        dice=2*tp/(pn+gn) if pn+gn else None;hd=None
        if pn and gn:
            structure=ndimage.generate_binary_structure(3,1)
            sa=a&~ndimage.binary_erosion(a,structure=structure,border_value=0)
            sb=b&~ndimage.binary_erosion(b,structure=structure,border_value=0)
            da=ndimage.distance_transform_edt(~sb,sampling=spacing)[sa]
            db=ndimage.distance_transform_edt(~sa,sampling=spacing)[sb]
            hd=float(np.percentile(np.r_[da,db],95))
        rows.append(dict(class_index=k,GT=gn,predicted=pn,TP=tp,FP=pn-tp,FN=gn-tp,Dice=dice,
          HD95_mm=hd,HD95_penalized_mm=hd if hd is not None else penalty if gn or pn else None,
          empty_prediction_failure=gn>0 and pn==0,GT_absent=gn==0))
    yt=torch.from_numpy(y.astype(np.int64));nll=0.;brier=0.
    for i in range(len(y)):
        q=p[i].double();target=yt[i]
        nll+=float(-q.clamp_min(1e-30).log().gather(0,target[None]).mean())
        one=F.one_hot(target,4).permute(2,0,1);brier+=float((q-one).square().sum(0).mean())
    return dict(classes=rows,NLL=nll/len(y),Brier=brier/len(y))


def aggregate(records):
    grouped=defaultdict(list)
    for v in records:grouped[v['patient']].append(v)
    assert len(grouped)==23 and all(len(v)==2 for v in grouped.values())
    patients=[]
    for patient,phases in sorted(grouped.items()):
        classes=[]
        for k in (1,2,3):
            cells=[p['metrics']['classes'][k-1] for p in phases];valid=[v for v in cells if not v['GT_absent']]
            classes.append(dict(class_index=k,Dice=float(np.mean([v['Dice'] for v in valid])) if valid else None,
              HD95_penalized_mm=float(np.mean([v['HD95_penalized_mm'] for v in valid])) if valid else None,
              HD95_success_only_mm=float(np.mean([v['HD95_mm'] for v in valid if v['HD95_mm'] is not None])) if any(v['HD95_mm'] is not None for v in valid) else None,
              empty_prediction_failures=sum(v['empty_prediction_failure'] for v in cells),GT_absent_phases=sum(v['GT_absent'] for v in cells)))
        dice=[v['Dice'] for v in classes if v['Dice'] is not None]
        patients.append(dict(patient=patient,classes=classes,macro_Dice=float(np.mean(dice)) if dice else None,
          NLL=float(np.mean([v['metrics']['NLL'] for v in phases])),Brier=float(np.mean([v['metrics']['Brier'] for v in phases]))))
    values=np.array([v['macro_Dice'] for v in patients]);assert np.isfinite(values).all()
    return dict(patients=patients,mean_Dice=float(values.mean()),median_Dice=float(np.median(values)),std_Dice=float(values.std(ddof=1)),
      class_Dice=[float(np.mean([p['classes'][k]['Dice'] for p in patients if p['classes'][k]['Dice'] is not None])) for k in range(3)],
      empty_prediction_failures=sum(c['empty_prediction_failures'] for p in patients for c in p['classes']))


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();active={}
    costs=dict(optimizer=0,reverse=0,training_encoder_images=0,train_theta=0,train_g_forward=0,train_g_image_forward=0,
      validation_encoder_images=0,validation_g_forward=0,validation_g_image_forward=0,validation_theta=0,validation_label_phase_reads=0)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(root/'CONTEXT.json',s);storage(s,reserve=6_000_000_000)
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        assert all(r['cohort']=='canonical320' and r['primary_20pct_split']=='train_labeled' for r in ds.records)
        schedule=Schedule(ds)
        records=load_training_records(Path(s['data']),seed=0,dataset='mnms',sites=['Siemens'],roles=['val'])
        assert len(records)==46 and len({r['patient_id'] for r in records})==23
        assert not set(schedule.patients)&{r['patient_id'] for r in records}
        records=sorted(records,key=lambda r:(r['patient_id'],r['case_id']))
        write(root/'PRIVATE_EVALUATION_MAP.json',records)
        sourcefiles=[Path(s['checkpoint']),Path(s['unet_initial']),Path(s['H_initial'])]
        meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in sourcefiles}
        initial=torch.load(sourcefiles[1],map_location='cpu',weights_only=False);assert initial['step']==0
        torch.manual_seed(20261020);encoder=build_model().cuda();encoder.load_state_dict(initial['model'])
        sourceG=torch.load(sourcefiles[0],map_location='cpu',weights_only=False);assert sourceG['cursor']==8675
        hinit=torch.load(sourcefiles[2],map_location='cpu',weights_only=False)['model']
        shared=None;models={};stage_results={}
        for stage in ('PRETRAIN','DIRECT','CONDITIONAL_G'):
            out=root/stage;out.mkdir(exist_ok=False)
            if stage=='PRETRAIN':model=encoder;params=tuple(model.parameters())
            elif stage=='DIRECT':
                encoder=build_model().cuda();encoder.load_state_dict(shared['model']);model=DirectHead(encoder).cuda();params=model.head_parameters()
            else:
                encoder=build_model().cuda();encoder.load_state_dict(shared['model'])
                generator=SpatialResidualModel().cuda();generator.load_state_dict(sourceG['model'])
                model=PatientImageInterface(encoder,generator).cuda()
                model.hc.load_state_dict({k[3:]:v for k,v in hinit.items() if k.startswith('hc.')})
                model.hr.load_state_dict({k[3:]:v for k,v in hinit.items() if k.startswith('hr.')});params=model.head_parameters()
            if shared is not None:restore_rng_state(shared['rng'])
            opt=torch.optim.Adam(params,lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
            count=[0];active=dict(stage=stage,model=model,opt=opt,step=0)
            def post(optimizer,args,kwargs):
                count[0]+=1;costs['optimizer']+=1;active['step']=count[0]
                assert count[0]<=5000 and costs['optimizer']<=15000
                append(root/'physical_updates.jsonl',dict(index=costs['optimizer'],stage=stage,step=count[0]))
            handle=opt.register_step_post_hook(post)
            def snapshot(step):return dict(stage=stage,step=step,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng_state(),costs=costs)
            save(out/'S0000.pt',snapshot(0))
            for step in range(1,5001):
                assert time.time()-s['original_started_unix']<85500
                absolute=step if stage=='PRETRAIN' else 5000+step
                x,y,ids,flips=schedule.batch(absolute);x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
                if stage=='PRETRAIN':loss=F.cross_entropy(model(x).logits,y,reduction='none').mean()
                elif stage=='DIRECT':loss=F.cross_entropy(model.logits(x),y,reduction='none').mean()
                else:loss,_=semantic_loss(model.generator,model.posterior(x),y,step,costs)
                costs['training_encoder_images']+=2
                assert torch.isfinite(loss);loss.backward();costs['reverse']+=1
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
                norm=float(nn.utils.clip_grad_norm_(params,1.));opt.step()
                if stage=='CONDITIONAL_G':model.generator.clear_draw()
                append(root/'steps.jsonl',dict(stage=stage,step=step,absolute_schedule_step=absolute,indices=ids,flips=flips,loss=float(loss.detach()),gradient_norm_before_clip=norm))
                if step%100==0:write(root/'STATUS.json',dict(status='TRAINING',stage=stage,step=step,costs=costs))
                if step%1000==0:
                    storage(s,reserve=6_000_000_000);save(out/f'S{step:04}.pt',snapshot(step))
            handle.remove();assert count[0]==5000 and all(int(v['step'])==5000 for v in opt.state_dict()['state'].values())
            stage_results[stage]=dict(optimizer=5000,trainable_parameters=sum(v.numel() for v in params))
            if stage=='PRETRAIN':shared=cpu(snapshot(5000))
            else:
                assert same(shared['model'],model.backbone.state_dict())
                if stage=='CONDITIONAL_G':assert same(sourceG['model'],model.generator.state_dict())
                models[stage]=model.eval()
            active={};del opt;torch.cuda.empty_cache()
        assert costs['optimizer']==costs['reverse']==15000
        assert costs['training_encoder_images']==30000 and costs['train_theta']==10000 and costs['train_g_forward']==20000
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
                    if stage=='CONDITIONAL_G':
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
        a=np.array([v['macro_Dice'] for v in aggregates['DIRECT']['patients']]);b=np.array([v['macro_Dice'] for v in aggregates['CONDITIONAL_G']['patients']])
        rng=np.random.default_rng(163);boot=np.mean((b-a)[rng.integers(0,23,size=(10000,23))],axis=1)
        save(root/'PRIVATE_PHASE_RESULTS.pt',all_records)
        write(root/'RESULTS.json',dict(status='COMPLETE',stages=stage_results,aggregates=aggregates,paired=dict(mean_Dice_difference=float((b-a).mean()),patient_bootstrap95=np.quantile(boot,[.025,.975]).tolist()),
          costs=costs,source_frozen=True,validation_predictions_before_labels=True,test_access=False,
          phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:save(root/'failure_partial.pt',dict(stage=active['stage'],step=active['step'],model=active['model'].state_dict(),optimizer=active['opt'].state_dict(),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
