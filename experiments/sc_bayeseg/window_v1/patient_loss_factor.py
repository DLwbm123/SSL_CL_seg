"""Matched segmentation-risk / local-KL factor study with fixed G0."""
import json,os,time,traceback
from pathlib import Path
import torch,numpy as np,h5py
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.patient_validation import Schedule,phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training
from sc_bayeseg.window_v1.patient_e2e_validation import EndToEndDirect,EndToEndG
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.r_start import cpu,write,append,save,draw_g,local_sample,local_kl
from sc_bayeseg.window_v1.common import storage

ARMS={'G_K0':(0,0),'G_D':(1,1),'G_DK0':(1,0),'U_D':(1,None)}


def foreground_dice_loss(p,y):
    target=F.one_hot(y,4).permute(0,3,1,2)[:,1:].to(p.dtype);p=p[:,1:]
    gt=target.sum((2,3));mask=gt>0
    dice=(2*(p*target).sum((2,3))+1e-6)/(p.sum((2,3))+gt+1e-6)
    # GT-absent classes are handled by CE; all-background images have Dice term zero.
    return (((1-dice)*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()


def g_loss(generator,q,y,step,costs,lamb,beta):
    tg=torch.Generator(device=y.device).manual_seed(720163+step);lg=torch.Generator(device=y.device).manual_seed(820163+step)
    ce=[];probs=[]
    for _ in range(2):
        draw_g(generator,tg);costs['train_theta']+=1
        for _ in range(2):
            c,r=local_sample(q,lg);z=generator.spatial_logits(c,r)
            ce.append(F.cross_entropy(z,y,reduction='none').mean((1,2)));probs.append(z.softmax(1))
            costs['train_g_forward']+=1;costs['train_g_image_forward']+=len(y)
    ce=torch.stack(ce).mean();dice=foreground_dice_loss(torch.stack(probs).mean(0),y)
    kc,kr=local_kl(q);kc=kc.mean()/16384;kr=kr.mean()/16384
    return ce+lamb*dice+beta*(kc+kr),dict(CE=float(ce.detach()),Dice_loss=float(dice.detach()),KLc=float(kc.detach()),KLR=float(kr.detach()))


@torch.no_grad()
def predict_native(model,x,seed,is_g):
    if not is_g:return model.logits(x).softmax(1)
    q=model.posterior(x);tg=torch.Generator(device=x.device).manual_seed(seed);lg=torch.Generator(device=x.device).manual_seed(seed+1000000)
    result=torch.zeros(len(x),4,128,128,device=x.device,dtype=x.dtype)
    try:
        for _ in range(4):
            draw_g(model.generator,tg)
            for _ in range(4):
                c,r=local_sample(q,lg);result+=model.generator.spatial_logits(c,r).softmax(1)
        return result/16
    finally:model.generator.clear_draw()


def restore_probability(native):
    p=F.interpolate(native.cuda(),(384,384),mode='bilinear',align_corners=False)
    return (p/p.sum(1,keepdim=True)).cpu()


def paired_summary(values,control):
    a=np.array([p['macro_Dice'] for p in values['patients']]);b=np.array([p['macro_Dice'] for p in control['patients']])
    assert [p['patient'] for p in values['patients']]==[p['patient'] for p in control['patients']]
    g=np.random.default_rng(163);boot=np.mean((a-b)[g.integers(0,len(a),size=(10000,len(a)))],axis=1)
    return dict(mean_Dice_difference=float((a-b).mean()),patient_bootstrap95=np.quantile(boot,[.025,.975]).tolist())


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();active={}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    costs=dict(optimizer=0,reverse=0,training_encoder_images=0,train_theta=0,train_g_forward=0,train_g_image_forward=0,evaluation_encoder_images=0,evaluation_g_forward=0,evaluation_g_image_forward=0,evaluation_theta=0,label_phase_reads=0,reconstruction_phases=0)
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=1_800_000_000)
        parent=Path(s['patient_control']);trainroot=Path(s['train_readout_source']);e2e=Path(s['patient_e2e_source'])
        load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
        assert json.loads((Path(s['supplement_source'])/'AUDIT.json').read_text())['status']=='PASS'
        maps=dict(train=json.loads((trainroot/'PRIVATE_EVALUATION_MAP.json').read_text()),val=json.loads((parent/'PRIVATE_EVALUATION_MAP.json').read_text()))
        assert len(maps['train'])==20 and len(maps['val'])==46
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None);schedule=Schedule(ds)
        assert not set(schedule.patients)&{v['patient_id'] for v in maps['val']}
        shared=load(parent/'PRETRAIN/S5000.pt');gsource=load(s['checkpoint']);hinit=load(s['H_initial'])['model']
        assert gsource['cursor']==8675
        sourcefiles=[parent/'PRETRAIN/S5000.pt',Path(s['checkpoint']),Path(s['H_initial']),e2e/'RESULTS.json',trainroot/'RESULTS.json']
        meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in sourcefiles};models={}
        for stage,(lamb,beta) in ARMS.items():
            folder=root/stage;folder.mkdir(exist_ok=False);encoder=build_model().cuda();encoder.load_state_dict(shared['model'])
            if stage=='U_D':model=EndToEndDirect(encoder).cuda()
            else:
                generator=SpatialResidualModel().cuda();generator.load_state_dict(gsource['model']);model=EndToEndG(encoder,generator).cuda()
                for prefix,module in (('hc.',model.hc),('hr.',model.hr)):module.load_state_dict({k[len(prefix):]:v for k,v in hinit.items() if k.startswith(prefix)})
            model.backbone.requires_grad_(True);model.backbone.segmentation_head.requires_grad_(False)
            params=tuple(p for p in model.parameters() if p.requires_grad)
            assert sum(p.numel() for p in params)==(482500 if stage=='U_D' else 484576)
            restore_rng_state(shared['rng'])
            opt=torch.optim.Adam(params,lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False);count=[0];active=dict(stage=stage,model=model,opt=opt,step=0)
            def post(optimizer,args,kwargs):
                count[0]+=1;costs['optimizer']+=1;active['step']=count[0];assert count[0]<=5000 and costs['optimizer']<=20000
                append(root/'physical_updates.jsonl',dict(index=costs['optimizer'],stage=stage,step=count[0]))
            handle=opt.register_step_post_hook(post)
            def snapshot(step):return dict(stage=stage,step=step,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng_state(),costs=costs,lambda_Dice=lamb,beta_KL=beta)
            save(folder/'S0000.pt',snapshot(0))
            for step in range(1,5001):
                assert time.time()-s['original_started_unix']<85500
                x,y,ids,flips=schedule.batch(5000+step);x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
                if stage=='U_D':
                    z=model.logits(x);ce=F.cross_entropy(z,y,reduction='none').mean();dl=foreground_dice_loss(z.softmax(1),y);loss=ce+dl;parts=dict(CE=float(ce.detach()),Dice_loss=float(dl.detach()),KLc=0.,KLR=0.)
                else:loss,parts=g_loss(model.generator,model.posterior(x),y,step,costs,lamb,beta)
                costs['training_encoder_images']+=len(x);assert torch.isfinite(loss);loss.backward();costs['reverse']+=1
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step()
                if stage!='U_D':model.generator.clear_draw()
                append(root/'steps.jsonl',dict(stage=stage,step=step,absolute_schedule_step=5000+step,indices=ids,flips=flips,loss=float(loss.detach()),gradient_norm_before_clip=norm,**parts))
                if step%100==0:write(root/'STATUS.json',dict(status='TRAINING',stage=stage,step=step,costs=costs))
                if step%1000==0:storage(s,reserve=1_800_000_000);save(folder/f'S{step:04}.pt',snapshot(step))
            handle.remove();assert count[0]==5000 and all(int(v['step'])==5000 for v in opt.state_dict()['state'].values())
            if stage!='U_D':assert same(model.generator.state_dict(),gsource['model'])
            models[stage]=model.eval();active={};del opt;torch.cuda.empty_cache()
        assert costs['optimizer']==costs['reverse']==20000 and costs['train_g_forward']==60000 and costs['train_theta']==30000
        for cohort,records in maps.items():
            for stage,model in models.items():
                folder=root/stage/cohort;folder.mkdir(exist_ok=False)
                for i,row in enumerate(records):
                    assert time.time()-s['original_started_unix']<85500
                    with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:volume=f['image'][...];spacing=json.loads(f.attrs['processed_spacing'])
                    result=[]
                    for offset in range(0,len(volume),2):
                        x=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda();p=predict_native(model,x,620163+1000*i+offset//2,stage!='U_D')
                        assert torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6;result.append(p.cpu());costs['evaluation_encoder_images']+=len(x)
                        if stage!='U_D':costs['evaluation_g_forward']+=16;costs['evaluation_g_image_forward']+=16*len(x);costs['evaluation_theta']+=4
                    save(folder/f'CASE{i:03}.pt',dict(native_probability=torch.cat(result),spacing=spacing));storage(s,reserve=100_000_000)
                    write(root/'STATUS.json',dict(status='PREDICTING',stage=stage,cohort=cohort,phase_index=i,costs=costs))
        assert costs['label_phase_reads']==0 and costs['evaluation_encoder_images']==3096 and costs['evaluation_g_forward']==19104 and costs['evaluation_theta']==4776
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
        controls=dict(train=json.loads((trainroot/'RESULTS.json').read_text())['training'],val=json.loads((e2e/'RESULTS.json').read_text())['aggregates']);contrasts={}
        for cohort in maps:
            a=aggregates[cohort];u=controls[cohort]['E2E_DIRECT' if cohort=='train' else 'DIRECT_E2E'];g=controls[cohort]['E2E_G0' if cohort=='train' else 'CONDITIONAL_G_E2E'];contrasts[cohort]={}
            for stage in ARMS:contrasts[cohort][stage+'_minus_original']=paired_summary(a[stage],u if stage=='U_D' else g)
            for left,right in (('G_D','U_D'),('G_DK0','U_D'),('G_DK0','G_K0'),('G_DK0','G_D')):contrasts[cohort][left+'_minus_'+right]=paired_summary(a[left],a[right])
            vec=lambda v:np.array([p['macro_Dice'] for p in v['patients']])
            for name,d in dict(Dice_KL_interaction=(vec(a['G_DK0'])-vec(a['G_K0']))-(vec(a['G_D'])-vec(g)),matched_Dice_gain_interaction=(vec(a['G_D'])-vec(g))-(vec(a['U_D'])-vec(u))).items():
                rg=np.random.default_rng(163);bt=np.mean(d[rg.integers(0,len(d),size=(10000,len(d)))],axis=1)
                contrasts[cohort][name]=dict(mean_Dice_difference=float(d.mean()),patient_bootstrap95=np.quantile(bt,[.025,.975]).tolist())
        assert costs['label_phase_reads']==264 and all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in meta.items())
        save(root/'PRIVATE_PHASE_RESULTS.pt',private);write(root/'RESULTS.json',dict(status='COMPLETE',aggregates=aggregates,contrasts=contrasts,costs=costs,source_frozen=True,all_predictions_before_labels=True,test_access=False,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:save(root/'failure_partial.pt',dict(stage=active['stage'],step=active['step'],model=active['model'].state_dict(),optimizer=active['opt'].state_dict(),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
