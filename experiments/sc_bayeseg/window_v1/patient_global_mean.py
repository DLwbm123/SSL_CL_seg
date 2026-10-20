"""Frozen Dmu readout at generator weight means; no optimization or stochastic draws."""
import json,os,time,traceback
from pathlib import Path
import torch,numpy as np,h5py
from sc_bayeseg.window_v1.patient_point_estimate import PointG32
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.patient_validation import phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability,paired_summary
from sc_bayeseg.window_v1.r_start import write,save
from sc_bayeseg.window_v1.common import storage
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.run import same

@torch.no_grad()
def predict(model,x):
    # Explicit detached means, preserving all saved distribution/prior parameters.
    for layer in model.generator.anatomy:
        layer.sampled=tuple(getattr(layer,'mean_'+v).detach() for v in ('w','b'))
    try:return model.generator.spatial_logits(*model.means(x)).softmax(1)
    finally:model.generator.clear_draw()

def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    costs=dict(optimizer=0,reverse=0,encoder_images=0,G_forward=0,G_image_forward=0,Theta_draws=0,local_draws=0,label_phase_reads=0,reconstruction_phases=0)
    rng=capture_rng_state()
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=300_000_000)
        read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
        assert read(Path(s['case_source'])/'AUDIT.json')['status']=='PASS'
        prior=Path(s['point_source']);assert read(prior/'AUDIT.json')['status']=='PASS'
        maps=dict(train=read(Path(s['train_readout_source'])/'PRIVATE_EVALUATION_MAP.json'),val=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json'))
        assert len(maps['train'])==20 and len(maps['val'])==46
        source=prior/'Dmu/S5000.pt';d=load(source);assert d['step']==5000 and d['mean_penalty_weight']==1
        meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in (source,prior/'RESULTS.json',Path(s['checkpoint']))}
        model=PointG32(build_model().cuda(),SpatialResidualModel().cuda()).cuda();model.load_state_dict(d['model']);model.requires_grad_(False);model.eval()
        for cohort,records in maps.items():
            folder=root/cohort;folder.mkdir(exist_ok=False)
            for i,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500
                with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:volume=f['image'][...];spacing=json.loads(f.attrs['processed_spacing'])
                results=[]
                for offset in range(0,len(volume),2):
                    x=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda();p=predict(model,x)
                    assert torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6
                    results.append(p.cpu());costs['encoder_images']+=len(x);costs['G_forward']+=1;costs['G_image_forward']+=len(x)
                save(folder/f'CASE{i:03}.pt',dict(native_probability=torch.cat(results),spacing=spacing));storage(s,reserve=50_000_000)
                write(root/'STATUS.json',dict(status='PREDICTING',cohort=cohort,phase=i,costs=costs))
        assert costs['encoder_images']==774 and costs['G_forward']==398 and costs['label_phase_reads']==0
        write(root/'PREDICTIONS_SEALED.json',dict(time_unix=time.time(),costs=costs,phases=66))
        aggregates={};private={};contrasts={};control=read(prior/'RESULTS.json')['aggregates'];matched=read(Path(s['factor_source'])/'RESULTS.json')['aggregates']
        for cohort,records in maps.items():
            rows=[]
            for i,row in enumerate(records):
                out=load(root/cohort/f'CASE{i:03}.pt');p=restore_probability(out['native_probability']);costs['reconstruction_phases']+=1
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,out['spacing'])));costs['label_phase_reads']+=1
                write(root/'STATUS.json',dict(status='SCORING',cohort=cohort,phase=i,costs=costs))
            aggregates[cohort]=(aggregate_training if cohort=='train' else aggregate)(rows);private[cohort]=rows
            contrasts[cohort]=dict(mean_minus_sampled_Dmu=paired_summary(aggregates[cohort],control[cohort]['Dmu']),mean_minus_U_D=paired_summary(aggregates[cohort],matched[cohort]['U_D']))
        assert costs['label_phase_reads']==costs['reconstruction_phases']==66 and same(model.state_dict(),d['model'])
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in meta.items());restore_rng_state(rng);assert same(capture_rng_state(),rng)
        save(root/'PRIVATE_PHASE_RESULTS.pt',private)
        write(root/'RESULTS.json',dict(status='COMPLETE',aggregates=aggregates,contrasts=contrasts,costs=costs,source_frozen=True,RNG_restored=True,predictions_sealed_before_labels=True,test_access=False,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        restore_rng_state(rng);write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
