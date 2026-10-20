"""Frozen-prediction error decomposition and bounded posterior/MC readout."""
import json,os,time,traceback
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,h5py
from torch.nn import functional as F
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.patient_image_interface import PatientImageInterface
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.r_start import write,save,draw_g,local_sample,cpu
from sc_bayeseg.window_v1.common import storage


def error_metrics(p,y):
    y=torch.as_tensor(y,dtype=torch.long);pred=p.argmax(1);cm=torch.bincount((4*y+pred).flatten(),minlength=16).reshape(4,4)
    loss=-p.double().clamp_min(1e-30).log().gather(1,y[:,None])[:,0];rows=[]
    for k in range(4):
        n=int((y==k).sum());pn=int(cm[:,k].sum());tp=int(cm[k,k]);total=float(loss[y==k].sum())
        rows.append(dict(GT=n,predicted=pn,TP=tp,precision=tp/pn if pn else None,recall=tp/n if n else None,volume_ratio=pn/n if n else None,NLL=total/n if n else None,NLL_contribution=total/y.numel()))
    return dict(confusion_GT_rows=cm.tolist(),classes=rows,NLL=float(loss.mean()),foreground_equal_NLL=float(np.mean([v['NLL'] for v in rows[1:] if v['NLL'] is not None])) if any(v['NLL'] is not None for v in rows[1:]) else None)


def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['patient']].append(row['metrics'])
    out=[]
    for patient,phases in sorted(groups.items()):
        assert len(phases)==2
        classes=[]
        for k in range(4):
            v={}
            for key in ('precision','recall','volume_ratio','NLL','NLL_contribution'):
                values=[p['classes'][k][key] for p in phases if p['classes'][k][key] is not None];v[key]=float(np.mean(values)) if values else None
            classes.append(v)
        out.append(dict(patient=patient,classes=classes,NLL=float(np.mean([p['NLL'] for p in phases])),foreground_equal_NLL=float(np.mean([p['foreground_equal_NLL'] for p in phases if p['foreground_equal_NLL'] is not None]))))
    cohort=[]
    for k in range(4):
        cohort.append({key:float(np.mean([p['classes'][k][key] for p in out if p['classes'][k][key] is not None])) if any(p['classes'][k][key] is not None for p in out) else None for key in classes[k]})
    return dict(patients=out,classes=cohort,NLL=float(np.mean([p['NLL'] for p in out])),foreground_equal_NLL=float(np.mean([p['foreground_equal_NLL'] for p in out])))


def posterior_stats(q):
    result={}
    for prefix,mu,ls in (('c',q[0],q[1]),('R',q[2],q[3])):
        result[prefix]=dict(mean_KL_over_N=float(.5*mu.double().square().flatten(1).sum(1).mean()/16384),scale_KL_over_N=float(.5*((2*ls.double()).exp()-1-2*ls.double()).flatten(1).sum(1).mean()/16384),mean_RMS=float(mu.double().square().mean().sqrt()),logsd_mean=float(ls.double().mean()),logsd_min=float(ls.min()),logsd_max=float(ls.max()),logsd_quantiles=torch.quantile(ls.double().flatten(),torch.tensor([.1,.5,.9],dtype=torch.float64)).tolist(),near_lower_fraction=float((ls<=-6.99).double().mean()),near_upper_fraction=float((ls>=-1.01).double().mean()),mapping_derivative_below_01=float((1-((ls+4)/3).square()<.01).double().mean()))
    return result


@torch.no_grad()
def native_readout(model,x,seed):
    q=model.posterior(x);tg=torch.Generator(device=x.device).manual_seed(seed);lg=torch.Generator(device=x.device).manual_seed(seed+1000000)
    p=torch.zeros(len(x),4,128,128,device=x.device);lp=torch.zeros_like(p)
    try:
        for _ in range(4):
            draw_g(model.generator,tg)
            for _ in range(4):
                c,r=local_sample(q,lg);logp=model.generator.spatial_logits(c,r).log_softmax(1);p+=logp.exp();lp+=logp
        return p/16,lp/16,q
    finally:model.generator.clear_draw()


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;rng=capture_rng_state()
    costs=dict(optimizer=0,reverse=0,encoder_images=0,G_forward=0,Theta=0,G_image_forward=0,sealed_phase_reads=0,native_label_phase_reads=0,new_validation_forwards=0)
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=600_000_000)
        parent=Path(s['patient_control']);e2e=Path(s['patient_e2e_source']);training=Path(s['train_readout_source'])
        load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
        trainmap=json.loads((training/'PRIVATE_EVALUATION_MAP.json').read_text());valmap=json.loads((parent/'PRIVATE_EVALUATION_MAP.json').read_text());assert len(trainmap)==20 and len(valmap)==46
        write(root/'PRIVATE_TRAIN_MAP.json',trainmap);write(root/'PRIVATE_VAL_MAP.json',valmap)
        specs=dict(FROZEN_DIRECT=(parent/'DIRECT/S5000.pt',parent/'DIRECT'),FROZEN_G0=(parent/'CONDITIONAL_G/S5000.pt',parent/'CONDITIONAL_G'),E2E_DIRECT=(e2e/'DIRECT_E2E/S5000.pt',e2e/'DIRECT_E2E'),E2E_G0=(e2e/'CONDITIONAL_G_E2E/S5000.pt',e2e/'CONDITIONAL_G_E2E'))
        meta={v[0]:(v[0].stat().st_size,v[0].stat().st_mtime_ns) for v in specs.values()};errors={};rawerrors={}
        for cohort,records in (('train',trainmap),('val',valmap)):
            errors[cohort]={};rawerrors[cohort]={}
            for name,(_,vfolder) in specs.items():
                folder=training/name if cohort=='train' else vfolder;rows=[]
                for i,row in enumerate(records):
                    assert time.time()-s['original_started_unix']<85500
                    p=load(folder/'predictions'/f'CASE{i:03}.pt')['probability']
                    with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                    rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=error_metrics(p,y)));costs['sealed_phase_reads']+=1
                rawerrors[cohort][name]=rows;errors[cohort][name]=aggregate(rows)
                write(root/'STATUS.json',dict(status='ERROR_DECOMPOSITION',cohort=cohort,stage=name,costs=costs))
        risk={};post={}
        for name in ('FROZEN_G0','E2E_G0'):
            payload=load(specs[name][0]);model=PatientImageInterface(build_model().cuda(),SpatialResidualModel().cuda()).cuda();model.load_state_dict(payload['model']);model.eval().requires_grad_(False)
            folder=root/name;folder.mkdir(exist_ok=False)
            for i,row in enumerate(trainmap):
                with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:volume=f['image'][...]
                ps=[];lps=[];qs=[[],[],[],[]]
                for offset in range(0,len(volume),2):
                    x=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda();p,lp,q=native_readout(model,x,620163+1000*i+offset//2)
                    ps.append(p.cpu());lps.append(lp.cpu())
                    for k in range(4):qs[k].append(q[k].cpu())
                    costs['encoder_images']+=len(x);costs['G_forward']+=16;costs['Theta']+=4;costs['G_image_forward']+=16*len(x)
                save(folder/f'CASE{i:03}.pt',dict(probability=torch.cat(ps),meanlog=torch.cat(lps),posterior=tuple(torch.cat(v) for v in qs)))
                write(root/'STATUS.json',dict(status='NATIVE_REPLAY',stage=name,phase_index=i,costs=costs));storage(s,reserve=50_000_000)
            assert same(model.state_dict(),payload['model']);del model,payload;torch.cuda.empty_cache()
        write(root/'NATIVE_SEALED.json',dict(costs=costs,time_unix=time.time()))
        for name in ('FROZEN_G0','E2E_G0'):
            risk[name]=[];post[name]=[]
            for i,row in enumerate(trainmap):
                d=load(root/name/f'CASE{i:03}.pt');p=d['probability'];lp=d['meanlog']
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=torch.from_numpy(f['label'][...].astype(np.int64))
                y=F.interpolate(y[:,None].float(),(128,128),mode='nearest')[:,0].long();ec=-lp.double().gather(1,y[:,None])[:,0];mn=-p.double().clamp_min(1e-30).log().gather(1,y[:,None])[:,0]
                assert float((ec-mn).min())>=-2e-6
                risk[name].append(dict(patient=row['patient_id'],case=row['case_id'],expected_CE128=float(ec.mean()),mixture_NLL128=float(mn.mean()),Jensen_gap128=float((ec-mn).mean())))
                post[name].append(dict(patient=row['patient_id'],case=row['case_id'],statistics=posterior_stats(d['posterior'])));costs['native_label_phase_reads']+=1
        assert (costs['encoder_images'],costs['G_forward'],costs['Theta'],costs['G_image_forward'],costs['sealed_phase_reads'],costs['native_label_phase_reads'])==(460,3840,960,7360,264,40)
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in meta.items());restore_rng_state(rng)
        save(root/'PRIVATE_ERROR_ROWS.pt',rawerrors)
        write(root/'RESULTS.json',dict(status='COMPLETE',errors=errors,native_risk=risk,posterior=post,costs=costs,source_frozen=True,RNG_restored=True,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        restore_rng_state(rng);write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
