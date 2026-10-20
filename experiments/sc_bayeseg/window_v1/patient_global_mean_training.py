"""Matched Dmu mean-penalty training with frozen G at weight means."""
import json,os,time,traceback
from pathlib import Path
import torch,numpy as np,h5py
from lcrseg.data import H5LabeledDataset
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.run import same
from torch.nn import functional as F
from sc_bayeseg.window_v1.patient_point_estimate import PointG32,VARIANCE_NAMES,check_variance
from sc_bayeseg.window_v1.patient_global_mean import predict
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.patient_validation import Schedule,phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training
from sc_bayeseg.window_v1.patient_loss_factor import foreground_dice_loss,restore_probability,paired_summary
from sc_bayeseg.window_v1.r_start import write,save,append
from sc_bayeseg.window_v1.common import storage
ARMS={'Mmu':1.}

def global_mean_loss(generator,q,y,step,costs,penalty):
    for layer in generator.anatomy:
        layer.sampled=tuple(getattr(layer,'mean_'+v).detach() for v in ('w','b'))
    logits=generator.spatial_logits(*q)
    ce=F.cross_entropy(logits,y,reduction='none').mean();dice=foreground_dice_loss(logits.softmax(1),y)
    mean_penalty=.5*(q[0].square().flatten(1).sum(1)+q[1].square().flatten(1).sum(1)).mean()/16384
    costs['train_g_forward']+=1;costs['train_g_image_forward']+=len(y)
    return ce+dice+penalty*mean_penalty,dict(CE=float(ce.detach()),Dice_loss=float(dice.detach()),mean_penalty=float(mean_penalty.detach()))

def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();active={}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    costs=dict(optimizer=0,reverse=0,training_encoder_images=0,train_theta=0,train_g_forward=0,train_g_image_forward=0,evaluation_encoder_images=0,evaluation_g_forward=0,evaluation_g_image_forward=0,evaluation_theta=0,label_phase_reads=0,reconstruction_phases=0)
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=550_000_000)
        parent=Path(s['patient_control']);trainroot=Path(s['train_readout_source']);e2e=Path(s['patient_e2e_source'])
        assert json.loads((Path(s['mean_source'])/'AUDIT.json').read_text())['status']=='PASS'
        load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
        assert json.loads((Path(s['supplement_source'])/'AUDIT.json').read_text())['status']=='PASS'
        maps=dict(train=json.loads((trainroot/'PRIVATE_EVALUATION_MAP.json').read_text()),val=json.loads((parent/'PRIVATE_EVALUATION_MAP.json').read_text()))
        assert len(maps['train'])==20 and len(maps['val'])==46
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None);schedule=Schedule(ds)
        assert not set(schedule.patients)&{v['patient_id'] for v in maps['val']}
        previous=Path(s['r32_source']);assert json.loads((previous/'AUDIT.json').read_text())['status']=='PASS'
        shared=load(previous/'S0000.pt');assert shared['step']==0 and shared['residual_grid']==32 and not shared['optimizer']['state']
        gsource=load(s['checkpoint']);assert gsource['cursor']==8675
        sourcefiles=[previous/'S0000.pt',previous/'RESULTS.json',Path(s['factor_source'])/'RESULTS.json',Path(s['checkpoint']),Path(s['mean_source'])/'RESULTS.json',Path(s['point_source'])/'RESULTS.json']
        meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in sourcefiles};models={}
        for stage,penalty in ARMS.items():
            folder=root/stage;folder.mkdir(exist_ok=False)
            model=PointG32(build_model().cuda(),SpatialResidualModel().cuda()).cuda();model.load_state_dict(shared['model'])
            model.backbone.requires_grad_(True);model.backbone.segmentation_head.requires_grad_(False)
            params=tuple(p for p in model.parameters() if p.requires_grad)
            assert sum(p.numel() for p in params)==484576
            assert sum(dict(model.named_parameters())[n][16:].numel() for n in VARIANCE_NAMES)==800
            restore_rng_state(shared['rng'])
            opt=torch.optim.Adam(params,lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False);count=[0];active=dict(stage=stage,model=model,opt=opt,step=0)
            def post(optimizer,args,kwargs):
                count[0]+=1;costs['optimizer']+=1;active['step']=count[0];assert count[0]<=5000 and costs['optimizer']<=5000
                append(root/'physical_updates.jsonl',dict(index=costs['optimizer'],stage=stage,step=count[0]))
            handle=opt.register_step_post_hook(post)
            def snapshot(step):return dict(stage=stage,step=step,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng_state(),costs=costs,lambda_Dice=1,mean_penalty_weight=penalty,global_weight_mode='mean_train_and_inference',residual_grid=32,optimizer_scalar_slots=484576,effective_update_scalars=483776,frozen_variance_scalars=800)
            save(folder/'S0000.pt',snapshot(0))
            for step in range(1,5001):
                assert time.time()-s['original_started_unix']<85500
                x,y,ids,flips=schedule.batch(5000+step);x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
                loss,parts=global_mean_loss(model.generator,model.means(x),y,step,costs,penalty)
                costs['training_encoder_images']+=len(x);assert torch.isfinite(loss);loss.backward();costs['reverse']+=1
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step()
                model.generator.clear_draw();check_variance(model,shared['model'],opt)
                append(root/'steps.jsonl',dict(stage=stage,step=step,absolute_schedule_step=5000+step,indices=ids,flips=flips,loss=float(loss.detach()),gradient_norm_before_clip=norm,**parts))
                if step%100==0:write(root/'STATUS.json',dict(status='TRAINING',stage=stage,step=step,costs=costs))
                if step%1000==0:storage(s,reserve=550_000_000);save(folder/f'S{step:04}.pt',snapshot(step))
            handle.remove();assert count[0]==5000 and all(int(v['step'])==5000 for v in opt.state_dict()['state'].values())
            assert same(model.generator.state_dict(),gsource['model'])
            models[stage]=model.eval();active={};del opt;torch.cuda.empty_cache()
        assert costs['optimizer']==costs['reverse']==5000 and costs['train_g_forward']==5000 and costs['train_theta']==0
        for cohort,records in maps.items():
            for stage,model in models.items():
                folder=root/stage/cohort;folder.mkdir(exist_ok=False)
                for i,row in enumerate(records):
                    assert time.time()-s['original_started_unix']<85500
                    with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:volume=f['image'][...];spacing=json.loads(f.attrs['processed_spacing'])
                    result=[]
                    for offset in range(0,len(volume),2):
                        x=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda();p=predict(model,x)
                        assert torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6;result.append(p.cpu());costs['evaluation_encoder_images']+=len(x)
                        costs['evaluation_g_forward']+=1;costs['evaluation_g_image_forward']+=len(x)
                    save(folder/f'CASE{i:03}.pt',dict(native_probability=torch.cat(result),spacing=spacing));storage(s,reserve=100_000_000)
                    write(root/'STATUS.json',dict(status='PREDICTING',stage=stage,cohort=cohort,phase_index=i,costs=costs))
        assert costs['label_phase_reads']==0 and costs['evaluation_encoder_images']==774 and costs['evaluation_g_forward']==398 and costs['evaluation_theta']==0
        write(root/'PREDICTIONS_SEALED.json',dict(costs=costs,phases_each=dict(train=20,val=46),time_unix=time.time(),cache_grid=128,evaluation_grid=384))
        aggregates={};private={}
        for cohort,records in maps.items():
            aggregates[cohort]={};private[cohort]={}
            for stage in ARMS:
                rows=[]
                for i,row in enumerate(records):
                    d=load(root/stage/cohort/f'CASE{i:03}.pt');p=restore_probability(d['native_probability']);costs['reconstruction_phases']+=1
                    with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                    rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,d['spacing'])));costs['label_phase_reads']+=1
                    write(root/'STATUS.json',dict(status='SCORING',stage=stage,cohort=cohort,phase_index=i,costs=costs))
                aggregates[cohort][stage]=(aggregate_training if cohort=='train' else aggregate)(rows);private[cohort][stage]=rows
        mean=json.loads((Path(s['mean_source'])/'RESULTS.json').read_text())['aggregates'];sampled=json.loads((Path(s['point_source'])/'RESULTS.json').read_text())['aggregates'];matched=json.loads((Path(s['factor_source'])/'RESULTS.json').read_text())['aggregates'];contrasts={}
        for cohort,a in aggregates.items():
            contrasts[cohort]={name:paired_summary(a['Mmu'],control) for name,control in (('Mmu_minus_sampletrained_meanread',mean[cohort]),('Mmu_minus_sampled_Dmu',sampled[cohort]['Dmu']),('Mmu_minus_U_D',matched[cohort]['U_D']))}
        assert costs['label_phase_reads']==66 and all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in meta.items())
        save(root/'PRIVATE_PHASE_RESULTS.pt',private);write(root/'RESULTS.json',dict(status='COMPLETE',aggregates=aggregates,contrasts=contrasts,costs=costs,source_frozen=True,all_predictions_before_labels=True,test_access=False,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:save(root/'failure_partial.pt',dict(stage=active['stage'],step=active['step'],model=active['model'].state_dict(),optimizer=active['opt'].state_dict(),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
