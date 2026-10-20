"""Independent trajectory and saved-native audit for matched mean-weight training."""
import json,os
from pathlib import Path
import torch,numpy as np,h5py
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.patient_point_estimate import PointG32,VARIANCE_NAMES
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.patient_validation import phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability,paired_summary
from sc_bayeseg.window_v1.r_start import write

def paired(a,b,metric):
    assert [p['patient'] for p in a]==[p['patient'] for p in b]
    d=np.array([x[metric]-y[metric] for x,y in zip(a,b)]);g=np.random.default_rng(163)
    return dict(mean_difference=float(d.mean()),patient_bootstrap95=np.quantile(d[g.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975]).tolist())

def audit(s):
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    c=r['costs'];assert (c['optimizer'],c['reverse'],c['training_encoder_images'],c['train_theta'],c['train_g_forward'],c['train_g_image_forward'])==(5000,5000,10000,0,5000,10000)
    assert (c['evaluation_encoder_images'],c['evaluation_theta'],c['evaluation_g_forward'],c['evaluation_g_image_forward'],c['label_phase_reads'],c['reconstruction_phases'])==(774,0,398,774,66,66)
    steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()];updates=[json.loads(x) for x in (root/'physical_updates.jsonl').read_text().splitlines()];old=[json.loads(x) for x in (Path(s['point_source'])/'steps.jsonl').read_text().splitlines() if json.loads(x)['stage']=='Dmu']
    assert len(steps)==len(updates)==len(old)==5000
    for i,(a,b,u) in enumerate(zip(steps,old,updates),1):
        assert a['stage']==u['stage']=='Mmu' and a['step']==u['step']==u['index']==i
        assert all(a[k]==b[k] for k in ('step','absolute_schedule_step','indices','flips'))
        assert abs(a['loss']-a['CE']-a['Dice_loss']-a['mean_penalty'])<5e-6
    initial=load(Path(s['r32_source'])/'S0000.pt');gsource={k:v for k,v in initial['model'].items() if k.startswith('generator.')}
    template=PointG32(build_model(),SpatialResidualModel());template.backbone.requires_grad_(True);template.backbone.segmentation_head.requires_grad_(False);names=[n for n,p in template.named_parameters() if p.requires_grad]
    assert sum(p.numel() for p in template.parameters() if p.requires_grad)==484576
    for step in range(0,5001,1000):
        q=load(root/'Mmu'/f'S{step:04}.pt');assert q['step']==step and q['lambda_Dice']==q['mean_penalty_weight']==1 and q['global_weight_mode']=='mean_train_and_inference' and q['residual_grid']==32 and q['effective_update_scalars']==483776 and q['frozen_variance_scalars']==800
        states=q['optimizer']['state'];assert (not states) if step==0 else all(int(v['step'])==step for v in states.values())
        if step==0:assert same(q['model'],initial['model']) and same(q['rng'],initial['rng'])
        for name in VARIANCE_NAMES:
            assert torch.equal(q['model'][name][16:],initial['model'][name][16:])
            if step:
                v=states[q['optimizer']['param_groups'][0]['params'][names.index(name)]];assert not torch.count_nonzero(v['exp_avg'][16:]) and not torch.count_nonzero(v['exp_avg_sq'][16:])
        for prefix in ('generator.','backbone.segmentation_head.'):
            assert same({k:v for k,v in q['model'].items() if k.startswith(prefix)},{k:v for k,v in initial['model'].items() if k.startswith(prefix)})
    assert read(root/'PREDICTIONS_SEALED.json')['costs']['label_phase_reads']==0
    private=load(root/'PRIVATE_PHASE_RESULTS.pt');mean=read(Path(s['mean_source'])/'RESULTS.json')['aggregates'];sampled=read(Path(s['point_source'])/'RESULTS.json')['aggregates'];matched=read(Path(s['factor_source'])/'RESULTS.json')['aggregates'];err=[0.];risk={}
    def eq(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:eq(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):eq(x,y)
        elif isinstance(a,float):
            e=abs(a-b);assert e<1e-10;err[0]=max(err[0],e)
        else:assert a==b
    for cohort,source in (('train',s['train_readout_source']),('val',s['patient_control'])):
        records=read(Path(source)/'PRIVATE_EVALUATION_MAP.json');rows=[];slices=0
        for i,row in enumerate(records):
            d=load(root/'Mmu'/cohort/f'CASE{i:03}.pt');p=d['native_probability'];assert p.dtype==torch.float32 and p.shape[1:]==(4,128,128) and torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6;slices+=len(p);p=restore_probability(p)
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,d['spacing'])))
        assert slices==(230 if cohort=='train' else 544);eq(rows,private[cohort]['Mmu']);a=(aggregate_training if cohort=='train' else aggregate)(rows);eq(a,r['aggregates'][cohort]['Mmu']);risk[cohort]={}
        for name,b in (('Mmu_minus_sampletrained_meanread',mean[cohort]),('Mmu_minus_sampled_Dmu',sampled[cohort]['Dmu']),('Mmu_minus_U_D',matched[cohort]['U_D'])):
            eq(paired_summary(a,b),r['contrasts'][cohort][name]);risk[cohort][name]={m:paired(a['patients'],b['patients'],m) for m in ('macro_Dice','NLL','Brier')}
        print('recomputed',cohort,flush=True)
    assert r['source_frozen'] and r['all_predictions_before_labels'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    result=dict(status='PASS',physical_updates=5000,full_states_checked=6,schedule_pairs=5000,initial_model_RNG_fresh_Adam_and_objective_checked=True,G_and_unused_head_frozen=True,variance_parameter_and_Adam_rows_exact=True,frozen_variance_scalars=800,native_prediction_sets_recomputed=66,maximum_metric_error=err[0],audit_reconstruction_phases=66,audit_label_reads=66,new_network_forward=0,new_reverse=0,new_optimizer=0)
    write(root/'AUDIT.json',result);r.pop('space');r.update(status='COMPLETE_AUDIT_PASS',audit=result,probability_contrasts=risk,scope='Matched mean-weight training; repeated development val; not full official BayeSeg')
    for cohort,arms in r['aggregates'].items():
        for a in arms.values():
            for i,p in enumerate(a['patients']):p['patient']=f'{cohort.upper()}{i:03}'
    write(root/'PUBLIC_RESULTS.json',r);print(json.dumps(result))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
