"""Bounded R32 interface diagnostic under CE+Dice and beta=0."""
import json,os,time,traceback
from pathlib import Path
import torch,numpy as np,h5py
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.patient_e2e_validation import EndToEndG
from sc_bayeseg.window_v1.patient_validation import Schedule,phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training
from sc_bayeseg.window_v1.patient_loss_factor import g_loss,predict_native,restore_probability,paired_summary
from sc_bayeseg.window_v1.r_start import write,append,save
from sc_bayeseg.window_v1.common import storage


class EndToEndG32(EndToEndG):
    def posterior(self,image):
        if image.ndim!=4 or image.shape[1]!=1:raise ValueError('image must be [B,1,H,W]')
        f=self.backbone(F.interpolate(image,(128,128),mode='bilinear',align_corners=False)).decoder_features['dec1']
        b,c,h,w=f.shape;assert (c,h,w)==(16,128,128)
        cm,cs=self.hc(f.mean((2,3))).chunk(2,1)
        rm,rs=self.hr(f.reshape(b,16,32,4,32,4).mean((3,5))).chunk(2,1)
        assert rm.shape==(b,16,32,32)
        return cm,-4+3*torch.tanh(cs/3),rm,-4+3*torch.tanh(rs/3)


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();active={}
    costs=dict(optimizer=0,reverse=0,training_encoder_images=0,train_theta=0,train_g_forward=0,train_g_image_forward=0,evaluation_encoder_images=0,evaluation_g_forward=0,evaluation_g_image_forward=0,evaluation_theta=0,label_phase_reads=0,reconstruction_phases=0)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(root/'CONTEXT.json',s);storage(s,reserve=600_000_000)
        factor=Path(s['factor_source']);assert json.loads((factor/'AUDIT.json').read_text())['status']=='PASS'
        load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
        initial=load(factor/'G_DK0/S0000.pt');assert initial['step']==0 and not initial['optimizer']['state'] and initial['lambda_Dice']==1 and initial['beta_KL']==0
        sourcefiles=[factor/'G_DK0/S0000.pt',factor/'RESULTS.json'];meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in sourcefiles}
        model=EndToEndG32(build_model().cuda(),SpatialResidualModel().cuda()).cuda();model.load_state_dict(initial['model'])
        model.backbone.requires_grad_(True);model.backbone.segmentation_head.requires_grad_(False)
        params=tuple(p for p in model.parameters() if p.requires_grad);assert sum(p.numel() for p in params)==484576
        gsource={k[10:]:v for k,v in initial['model'].items() if k.startswith('generator.')}
        restore_rng_state(initial['rng']);opt=torch.optim.Adam(params,lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        active=dict(model=model,opt=opt,step=0)
        def post(optimizer,args,kwargs):
            costs['optimizer']+=1;active['step']=costs['optimizer'];assert costs['optimizer']<=5000
            append(root/'physical_updates.jsonl',dict(index=costs['optimizer'],step=costs['optimizer']))
        handle=opt.register_step_post_hook(post)
        def snapshot(step):return dict(step=step,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng_state(),costs=costs,residual_grid=32,lambda_Dice=1,beta_KL=0)
        save(root/'S0000.pt',snapshot(0));schedule=Schedule(H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None))
        for step in range(1,5001):
            assert time.time()-s['original_started_unix']<85500
            x,y,ids,flips=schedule.batch(5000+step);x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
            loss,parts=g_loss(model.generator,model.posterior(x),y,step,costs,1,0);assert torch.isfinite(loss)
            costs['training_encoder_images']+=len(x);loss.backward();costs['reverse']+=1
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();model.generator.clear_draw()
            append(root/'steps.jsonl',dict(step=step,absolute_schedule_step=5000+step,indices=ids,flips=flips,loss=float(loss.detach()),gradient_norm_before_clip=norm,**parts))
            if step%100==0:write(root/'STATUS.json',dict(status='TRAINING',step=step,costs=costs))
            if step%1000==0:
                assert same(model.generator.state_dict(),gsource);storage(s,reserve=400_000_000);save(root/f'S{step:04}.pt',snapshot(step))
        handle.remove();assert costs['optimizer']==costs['reverse']==5000 and all(int(v['step'])==5000 for v in opt.state_dict()['state'].values());active={};model.eval()
        maps=dict(train=json.loads((Path(s['train_readout_source'])/'PRIVATE_EVALUATION_MAP.json').read_text()),val=json.loads((Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json').read_text()))
        assert len(maps['train'])==20 and len(maps['val'])==46
        assert not set(schedule.patients)&{v['patient_id'] for v in maps['val']}
        for cohort,records in maps.items():
            folder=root/cohort;folder.mkdir(exist_ok=False)
            for i,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500
                with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:volume=f['image'][...];spacing=json.loads(f.attrs['processed_spacing'])
                result=[]
                for offset in range(0,len(volume),2):
                    x=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda();p=predict_native(model,x,620163+1000*i+offset//2,True)
                    assert torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6;result.append(p.cpu())
                    costs['evaluation_encoder_images']+=len(x);costs['evaluation_g_forward']+=16;costs['evaluation_g_image_forward']+=16*len(x);costs['evaluation_theta']+=4
                save(folder/f'CASE{i:03}.pt',dict(native_probability=torch.cat(result),spacing=spacing));storage(s,reserve=50_000_000)
                write(root/'STATUS.json',dict(status='PREDICTING',cohort=cohort,phase_index=i,costs=costs))
        assert costs['label_phase_reads']==0 and costs['evaluation_encoder_images']==774 and costs['evaluation_g_forward']==6368
        write(root/'PREDICTIONS_SEALED.json',dict(costs=costs,time_unix=time.time(),cache_grid=128,evaluation_grid=384))
        aggregates={};private={};contrasts={};controls=json.loads((factor/'RESULTS.json').read_text())['aggregates']
        for cohort,records in maps.items():
            rows=[]
            for i,row in enumerate(records):
                d=load(root/cohort/f'CASE{i:03}.pt');p=restore_probability(d['native_probability']);costs['reconstruction_phases']+=1
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,d['spacing'])));costs['label_phase_reads']+=1
                write(root/'STATUS.json',dict(status='SCORING',cohort=cohort,phase_index=i,costs=costs))
            aggregates[cohort]=(aggregate_training if cohort=='train' else aggregate)(rows);private[cohort]=rows
            contrasts[cohort]={name:paired_summary(aggregates[cohort],controls[cohort][name]) for name in ('G_DK0','U_D')}
        assert costs['label_phase_reads']==66 and same(model.generator.state_dict(),gsource)
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in meta.items())
        save(root/'PRIVATE_PHASE_RESULTS.pt',private);write(root/'RESULTS.json',dict(status='COMPLETE',aggregates=aggregates,contrasts=contrasts,costs=costs,source_frozen=True,test_access=False,all_predictions_before_labels=True,local_noise_shape_changed=True,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:save(root/'failure_partial.pt',dict(step=active['step'],model=active['model'].state_dict(),optimizer=active['opt'].state_dict(),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
