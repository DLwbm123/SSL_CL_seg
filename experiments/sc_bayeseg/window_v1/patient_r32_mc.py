"""Fixed-terminal R32 local-KL comparison across four independent MC streams."""
import json,os,time,traceback
from pathlib import Path
import torch,numpy as np,h5py
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.patient_r32 import EndToEndG32
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.patient_validation import phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_loss_factor import predict_native,restore_probability,paired_summary
from sc_bayeseg.window_v1.r_start import write,save,append
from sc_bayeseg.window_v1.common import storage


def class_nll(p,y):
    target=torch.from_numpy(np.asarray(y,dtype=np.int64));q=p.double().clamp_min(1e-30)
    loss=-q.log().gather(1,target[:,None]).squeeze(1);cells=[]
    for k in range(4):
        mask=target==k;n=int(mask.sum());total=float(loss[mask].sum())
        cells.append(dict(class_index=k,pixels=n,NLL_sum=total,conditional_NLL=total/n if n else None,full_image_contribution=total/target.numel()))
    assert abs(sum(c['full_image_contribution'] for c in cells)-float(loss.mean()))<1e-7
    return cells


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    costs=dict(optimizer=0,reverse=0,encoder_images=0,theta=0,g_forward=0,g_image_forward=0,label_phases=0,reconstruction_phases=0)
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    read=lambda p:json.loads(Path(p).read_text())
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=800_000_000)
        previous=Path(s['kl_source']);assert read(previous/'AUDIT.json')['status']=='PASS' and read(previous/'EXIT.json')['exit_code']==0
        records=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json');assert len(records)==46
        sources=dict(K1=previous/'R32_K1/S5000.pt',K0=Path(s['r32_source'])/'S5000.pt')
        meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in sources.values()};models={};states={}
        for arm,p in sources.items():
            state=load(p);assert state['step']==5000 and state['residual_grid']==32
            m=EndToEndG32(build_model().cuda(),SpatialResidualModel().cuda()).cuda();m.load_state_dict(state['model']);m.eval().requires_grad_(False)
            states[arm]=state['model'];models[arm]=m
        for stream in range(4):
            for arm,m in models.items():
                folder=root/f'FLOW{stream}'/arm;folder.mkdir(parents=True,exist_ok=False)
                for i,row in enumerate(records):
                    assert time.time()-s['original_started_unix']<85500
                    with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:volume=f['image'][...];spacing=json.loads(f.attrs['processed_spacing'])
                    parts=[]
                    for offset in range(0,len(volume),2):
                        x=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda()
                        p=predict_native(m,x,1620163+10000000*stream+1000*i+offset//2,True)
                        assert torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6;parts.append(p.cpu())
                        costs['encoder_images']+=len(x);costs['theta']+=4;costs['g_forward']+=16;costs['g_image_forward']+=16*len(x)
                    save(folder/f'CASE{i:03}.pt',dict(native_probability=torch.cat(parts),spacing=spacing));append(root/'predictions.jsonl',dict(stream=stream,arm=arm,phase=i,costs=dict(costs)))
                    write(root/'STATUS.json',dict(status='PREDICTING',stream=stream,arm=arm,phase=i,costs=costs));storage(s,reserve=100_000_000)
        assert (costs['encoder_images'],costs['theta'],costs['g_forward'],costs['g_image_forward'],costs['label_phases'])==(4352,8896,35584,69632,0)
        write(root/'PREDICTIONS_SEALED.json',dict(time_unix=time.time(),sets=8,phases=368,costs=dict(costs)))
        control=read(Path(s['factor_source'])/'RESULTS.json')['aggregates']['val']['U_D'];aggregates={};private={};contrasts={}
        for stream in range(4):
            key=f'FLOW{stream}';aggregates[key]={};private[key]={};contrasts[key]={}
            for arm in models:
                rows=[]
                for i,row in enumerate(records):
                    assert time.time()-s['original_started_unix']<85500
                    d=load(root/key/arm/f'CASE{i:03}.pt');p=restore_probability(d['native_probability']);costs['reconstruction_phases']+=1
                    with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                    metrics=phase_metrics(p,y,d['spacing']);cells=class_nll(p,y)
                    assert abs(sum(c['full_image_contribution'] for c in cells)-metrics['NLL'])<1e-7
                    rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=metrics,class_NLL=cells));costs['label_phases']+=1
                    write(root/'STATUS.json',dict(status='SCORING',stream=stream,arm=arm,phase=i,costs=costs))
                a=aggregate(rows);aggregates[key][arm]=a;private[key][arm]=rows
                contrasts[key][arm+'_minus_U_D']=paired_summary(a,control)
            contrasts[key]['K1_minus_K0']=paired_summary(aggregates[key]['K1'],aggregates[key]['K0'])
        assert costs['label_phases']==costs['reconstruction_phases']==368
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in meta.items()) and all(same(m.state_dict(),states[arm]) for arm,m in models.items())
        save(root/'PRIVATE_PHASE_RESULTS.pt',private)
        write(root/'RESULTS.json',dict(status='COMPLETE',aggregates=aggregates,contrasts=contrasts,costs=costs,source_frozen=True,all_predictions_before_labels=True,test_access=False,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
