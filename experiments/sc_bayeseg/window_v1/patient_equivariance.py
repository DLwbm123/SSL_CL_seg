"""Four matched two-view supervised arms; original CE and mixture-Dice preserved."""
import json,os,time,traceback
from pathlib import Path
import torch,numpy as np,h5py
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.patient_e2e_validation import EndToEndDirect
from sc_bayeseg.window_v1.patient_point_estimate import PointG32,check_variance
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.patient_validation import Schedule,phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training
from sc_bayeseg.window_v1.patient_loss_factor import foreground_dice_loss,restore_probability
from sc_bayeseg.window_v1.patient_pro_diagnostics import risk,fit_temperature
from sc_bayeseg.window_v1.patient_tta_transitions import transition_summary
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.r_start import draw_g,write,save,append
from sc_bayeseg.window_v1.equivariance_limits import storage
ARMS=('U_2V','U_EQ','G_2V','G_EQ')

def second_view(x,y,step):
    # Nonidentity relative flip, fixed independently of arm and GT.
    g=torch.Generator().manual_seed(930163+step);relative=torch.randint(1,4,(len(x),),generator=g).tolist()
    return torch.stack([flip(v,k) for v,k in zip(x,relative)]),torch.stack([flip(v,k) for v,k in zip(y,relative)]),relative

def class_js(a,b,y):
    m=(a+b)*.5;logm=m.clamp_min(1e-30).log()
    js=.5*((a*(a.clamp_min(1e-30).log()-logm)).sum(1)+(b*(b.clamp_min(1e-30).log()-logm)).sum(1))
    assert float(js.detach().min())>=-2e-6
    cells=[]
    for k in range(4):
        mask=y==k;n=mask.sum((1,2));cells.append(((js*mask).sum((1,2))/n.clamp_min(1),n>0))
    values=torch.stack([v for v,n in cells],1);present=torch.stack([n for v,n in cells],1)
    return ((values*present).sum(1)/present.sum(1)).mean()

def objective(model,xa,ya,xb,yb,rel,step,is_g,costs=None):
    pa=[];pb=[];ca=[];cb=[]
    if is_g:
        qa=model.means(xa)
        if costs is not None:costs['train_encoder_images']+=len(xa)
        qb=model.means(xb)
        if costs is not None:costs['train_encoder_images']+=len(xb)
        tg=torch.Generator(device=xa.device).manual_seed(720163+step)
        for j in range(2):
            draw_g(model.generator,tg)
            if costs is not None:costs['train_Theta']+=1
            za=model.generator.spatial_logits(*qa)
            if costs is not None:costs['train_G_forward']+=1;costs['train_G_image_forward']+=len(xa)
            zb=model.generator.spatial_logits(*qb)
            if costs is not None:costs['train_G_forward']+=1;costs['train_G_image_forward']+=len(xb)
            pa.append(za.softmax(1));pb.append(zb.softmax(1));ca.append(F.cross_entropy(za,ya,reduction='none').mean());cb.append(F.cross_entropy(zb,yb,reduction='none').mean())
        reg=sum(.5*(q[0].square().flatten(1).sum(1)+q[1].square().flatten(1).sum(1)).mean()/16384 for q in (qa,qb))/2
    else:
        za=model.logits(xa)
        if costs is not None:costs['train_encoder_images']+=len(xa);costs['train_U_logits']+=1
        zb=model.logits(xb)
        if costs is not None:costs['train_encoder_images']+=len(xb);costs['train_U_logits']+=1
        pa=[za.softmax(1)];pb=[zb.softmax(1)];ca=[F.cross_entropy(za,ya,reduction='none').mean()];cb=[F.cross_entropy(zb,yb,reduction='none').mean()];reg=za.new_zeros(())
    ce=(torch.stack(ca).mean()+torch.stack(cb).mean())/2
    dice=(foreground_dice_loss(torch.stack(pa).mean(0),ya)+foreground_dice_loss(torch.stack(pb).mean(0),yb))/2
    eq=torch.stack([class_js(a,torch.stack([flip(v,k) for v,k in zip(b,rel)]),ya) for a,b in zip(pa,pb)]).mean()
    return ce+dice+reg,eq,dict(CE=float(ce.detach()),Dice_loss=float(dice.detach()),mean_penalty=float(reg.detach()),JS=float(eq.detach()))

def predict(model,x,is_g):
    if not is_g:return model.logits(x).softmax(1)
    for layer in model.generator.anatomy:layer.sampled=(layer.mean_w.detach(),layer.mean_b.detach())
    try:return model.generator.spatial_logits(*model.means(x)).softmax(1)
    finally:model.generator.clear_draw()

def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();active={};load=lambda p:torch.load(p,map_location='cpu',weights_only=False);read=lambda p:json.loads(Path(p).read_text())
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    costs=dict(optimizer=0,training_reverse=0,diagnostic_reverse_requests=0,train_encoder_images=0,train_U_logits=0,train_G_forward=0,train_G_image_forward=0,train_Theta=0,evaluation_encoder_images=0,evaluation_U_logits=0,evaluation_G_forward=0,evaluation_G_image_forward=0,main_probability_sets=0,aligned_flip_sets=0,raw_restores=0,label_phase_reads=0,temperature_fits=0,temperature_functions=0,calibrated_restores=0,argmax_changes=0)
    def deadline():assert time.time()-s['original_started_unix']<85500
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=4_000_000_000)
        sources=dict(U=Path(s['factor_source'])/'U_D/S0000.pt',G=Path(s['r32_source'])/'S0000.pt');initial={k:load(p) for k,p in sources.items()}
        assert all(v['step']==0 and not v['optimizer']['state'] for v in initial.values())
        maps=dict(train=read(Path(s['train_readout_source'])/'PRIVATE_EVALUATION_MAP.json'),val=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json'));folds=read(Path(s['diagnostic_source'])/'PRIVATE_FOLDS.json')
        assert len(maps['train'])==20 and len(maps['val'])==46 and len(folds)==23
        metadata={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sources.values()};write(root/'PRIVATE_SOURCE_METADATA.json',metadata)
        schedule=Schedule(H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None));assert not set(schedule.patients)&set(folds)
        models={}
        for arm in ARMS:
            deadline();is_g=arm.startswith('G');is_eq=arm.endswith('EQ');folder=root/arm;folder.mkdir(exist_ok=False);source=initial['G' if is_g else 'U']
            model=(PointG32(build_model().cuda(),SpatialResidualModel().cuda()) if is_g else EndToEndDirect(build_model().cuda())).cuda();model.load_state_dict(source['model']);model.backbone.requires_grad_(True);model.backbone.segmentation_head.requires_grad_(False)
            params=tuple(p for p in model.parameters() if p.requires_grad);assert sum(p.numel() for p in params)==(484576 if is_g else 482500)
            opt=torch.optim.Adam(params,lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False);restore_rng_state(source['rng']);active=dict(arm=arm,model=model,opt=opt,step=0);count=[0]
            def post(optimizer,args,kwargs):
                count[0]+=1;costs['optimizer']+=1;active['step']=count[0];append(root/'physical_updates.jsonl',dict(index=costs['optimizer'],arm=arm,step=count[0]))
            handle=opt.register_step_post_hook(post)
            def snapshot(step):return dict(arm=arm,step=step,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng_state(),costs=dict(costs),JS_enabled=is_eq)
            save(folder/'S0000.pt',snapshot(0))
            for step in range(1,5001):
                deadline();xa,ya,ids,flips=schedule.batch(5000+step);xa=xa.cuda();ya=ya.cuda();xb,yb,relative=second_view(xa,ya,step);opt.zero_grad(set_to_none=True)
                base,eq,parts=objective(model,xa,ya,xb,yb,relative,step,is_g,costs);weight=.1*min(step/500,1.) if is_eq else 0.;loss=base+weight*eq;assert torch.isfinite(loss)
                if is_eq and step in (1,1000,2000,3000,4000,5000):
                    gb=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True);costs['diagnostic_reverse_requests']+=1
                    ge=torch.autograd.grad(weight*eq,params,retain_graph=True,allow_unused=True);costs['diagnostic_reverse_requests']+=1
                    nb=sum(float(v.detach().double().square().sum()) for v in gb if v is not None)**.5;ne=sum(float(v.detach().double().square().sum()) for v in ge if v is not None)**.5
                    append(root/'gradient_probes.jsonl',dict(arm=arm,step=step,base_norm=nb,weighted_JS_norm=ne,ratio=ne/nb if nb else None));del gb,ge
                loss.backward();costs['training_reverse']+=1;assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step()
                if is_g:model.generator.clear_draw();check_variance(model,source['model'],opt)
                append(root/'steps.jsonl',dict(arm=arm,step=step,absolute_schedule_step=5000+step,indices=ids,flips=flips,relative_flips=relative,JS_weight=weight,loss=float(loss.detach()),gradient_norm_before_clip=norm,**parts))
                if step%100==0:write(root/'STATUS.json',dict(status='TRAINING',arm=arm,step=step,costs=costs))
                if step%1000==0:storage(s,reserve=4_000_000_000);save(folder/f'S{step:04}.pt',snapshot(step))
            handle.remove();assert count[0]==5000 and all(int(v['step'])==5000 for v in opt.state_dict()['state'].values());models[arm]=model.eval();active={};del opt
        assert (costs['optimizer'],costs['training_reverse'],costs['diagnostic_reverse_requests'],costs['train_encoder_images'],costs['train_U_logits'],costs['train_G_forward'],costs['train_Theta'])==(20000,20000,24,80000,20000,40000,20000)
        equiv={a:[] for a in ARMS}
        for arm,model in models.items():
            is_g=arm.startswith('G')
            for cohort,records in maps.items():
                for i,row in enumerate(records):
                    deadline()
                    with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:volume=f['image'][...];spacing=json.loads(f.attrs['processed_spacing'])
                    count=4 if cohort=='val' else 1;identity=None
                    for k in range(count):
                        output=[]
                        for offset in range(0,len(volume),2):
                            x=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda()
                            with torch.no_grad():p=flip(predict(model,flip(x,k),is_g),k)
                            assert torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6;output.append(p.cpu());costs['evaluation_encoder_images']+=len(x)
                            if is_g:costs['evaluation_G_forward']+=1;costs['evaluation_G_image_forward']+=len(x)
                            else:costs['evaluation_U_logits']+=1
                        native=torch.cat(output);dest=root/arm/cohort/(f'flip{k}' if k else 'identity');dest.mkdir(parents=True,exist_ok=True);save(dest/f'CASE{i:03}.pt',dict(native_probability=native,spacing=spacing))
                        if not k:identity=native;costs['main_probability_sets']+=1
                        else:
                            costs['aligned_flip_sets']+=1;equiv[arm].append(dict(phase=i,flip=k,mean_pixel_TV=float((native.double()-identity.double()).abs().sum())/(2*len(native)*128**2),hard_disagreement=float((native.argmax(1)!=identity.argmax(1)).double().mean())))
                        append(root/'forward_phase_ledger.jsonl',dict(arm=arm,cohort=cohort,phase=i,flip=k,slices=len(volume),label_reads=0,costs=dict(costs)));storage(s,reserve=100_000_000)
                    write(root/'STATUS.json',dict(status='PREDICTING',arm=arm,cohort=cohort,phase=i,costs=costs))
        assert (costs['main_probability_sets'],costs['aligned_flip_sets'],costs['evaluation_encoder_images'],costs['evaluation_U_logits'],costs['evaluation_G_forward'],costs['evaluation_G_image_forward'],costs['label_phase_reads'])==(264,552,9624,2464,2464,4812,0)
        write(root/'PREDICTIONS_SEALED.json',dict(time_unix=time.time(),costs=dict(costs)));write(root/'PRIVATE_EQUIVARIANCE.json',equiv)
        private={c:{a:[] for a in ARMS} for c in maps};risks={a:[] for a in ARMS};quad={a:[] for a in ARMS};transitions={a:[] for a in ('U','G')}
        for cohort,records in maps.items():
            for i,row in enumerate(records):
                deadline()
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                costs['label_phase_reads']+=1;pred={}
                for arm in ARMS:
                    d=load(root/arm/cohort/'identity'/f'CASE{i:03}.pt');p=restore_probability(d['native_probability']);costs['raw_restores']+=1;pred[arm]=p.argmax(1).numpy()
                    private[cohort][arm].append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,d['spacing'])))
                    if cohort=='val':
                        risks[arm].append(dict(patient=row['patient_id'],fold=folds[row['patient_id']],risk=risk(p,y)))
                        quad[arm].append(dict(patient=row['patient_id'],fold=folds[row['patient_id']],logp=p.permute(0,2,3,1).reshape(-1,4)[::64].double().clamp_min(1e-30).log(),target=torch.from_numpy(np.asarray(y,dtype=np.int64).reshape(-1)[::64].copy())))
                if cohort=='val':
                    for kind in ('U','G'):
                        m=np.bincount((np.asarray(y,dtype=np.int64)*16+pred[kind+'_2V']*4+pred[kind+'_EQ']).ravel(),minlength=64).reshape(4,4,4);classes=[transition_summary(m,k) for k in (1,2,3)]
                        transitions[kind].append(dict(patient=row['patient_id'],phase=i,counts=m.tolist(),classes=classes,new_RV_real_loss=classes[2]['identity_TP']>0 and classes[2]['TTA_TP']==0))
                write(root/'STATUS.json',dict(status='RAW_SCORING',cohort=cohort,phase=i,costs=costs))
        fits={}
        for arm in ARMS:
            fits[arm]={}
            for fold in range(5):
                deadline();selected=[v for v in quad[arm] if v['fold']!=fold];lp=torch.cat([v['logp'] for v in selected]).cuda();target=torch.cat([v['target'] for v in selected]).cuda();weights=torch.cat([torch.full((len(v['target']),),1/(len(selected)*len(v['target'])),dtype=torch.float64) for v in selected]).cuda()
                fit=fit_temperature(lp,target,weights);fit.update(fit_patients=sorted({v['patient'] for v in selected}),heldout_patients=sorted(p for p in folds if folds[p]==fold));fits[arm][str(fold)]=fit;costs['temperature_fits']+=1;costs['temperature_functions']+=fit['function_evaluations'];del lp,target,weights
        save(root/'PRIVATE_TEMPERATURES.pt',fits);write(root/'TEMPERATURES_SEALED.json',dict(time_unix=time.time(),costs=dict(costs)))
        for i,row in enumerate(maps['val']):
            zs={}
            for arm in ARMS:
                deadline();p=restore_probability(load(root/arm/'val/identity'/f'CASE{i:03}.pt')['native_probability']);costs['calibrated_restores']+=1;fit=fits[arm][str(folds[row['patient_id']])];assert row['patient_id'] not in fit['fit_patients'];z=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);changes=int((z.argmax(1)!=p.argmax(1)).sum());assert changes==0;costs['argmax_changes']+=changes;zs[arm]=z
                append(root/'temperature_output_manifest.jsonl',dict(arm=arm,phase=i,fold=folds[row['patient_id']],T=fit['T'],sealed_before_label=True,argmax_changes=changes))
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            costs['label_phase_reads']+=1
            for arm,z in zs.items():risks[arm][i]['temperature_risk']=risk(z,y)
        assert (costs['raw_restores'],costs['calibrated_restores'],costs['label_phase_reads'],costs['temperature_fits'],costs['argmax_changes'])==(264,184,112,20,0) and costs['temperature_functions']<=1280
        aggregates={c:{a:(aggregate_training if c=='train' else aggregate)(rows) for a,rows in arms.items()} for c,arms in private.items()}
        for arm,model in models.items():
            source=initial['G' if arm.startswith('G') else 'U']['model'];assert same({k:v for k,v in model.state_dict().items() if k.startswith('backbone.segmentation_head.')},{k:v for k,v in source.items() if k.startswith('backbone.segmentation_head.')})
            if arm.startswith('G'):assert same(model.generator.state_dict(),{k.removeprefix('generator.'):v for k,v in source.items() if k.startswith('generator.')});check_variance(model,source)
        assert all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==(v['size'],v['mtime_ns']) for p,v in metadata.items())
        save(root/'PRIVATE_PHASE_RESULTS.pt',private);save(root/'PRIVATE_RISKS.pt',risks);write(root/'PRIVATE_TRANSITIONS.json',transitions)
        write(root/'RESULTS.json',dict(status='COMPLETE',aggregates=aggregates,costs=costs,source_frozen=True,predictions_sealed_before_labels=True,test_access=False,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:save(root/'failure_partial.pt',dict(arm=active['arm'],step=active['step'],model=active['model'].state_dict(),optimizer=active['opt'].state_dict(),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
