"""Audit matched R32 KL trajectories and exact saved-native patient evaluation."""
import json,os
from pathlib import Path
import torch,h5py,numpy as np
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.patient_r32 import EndToEndG32
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.patient_validation import phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability,paired_summary
from sc_bayeseg.window_v1.r_start import write


def audit(s):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(s['output']);factor=Path(s['factor_source']);read=lambda p:json.loads(p.read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    c=r['costs'];assert (c['optimizer'],c['reverse'],c['training_encoder_images'],c['train_theta'],c['train_g_forward'],c['train_g_image_forward'])==(10000,10000,20000,20000,40000,80000)
    assert (c['evaluation_encoder_images'],c['evaluation_theta'],c['evaluation_g_forward'],c['evaluation_g_image_forward'],c['label_phase_reads'],c['reconstruction_phases'])==(1548,3184,12736,24768,132,132)
    steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()];updates=[json.loads(x) for x in (root/'physical_updates.jsonl').read_text().splitlines()];old=[json.loads(x) for x in (Path(s['r32_source'])/'steps.jsonl').read_text().splitlines()]
    assert len(steps)==len(updates)==10000 and len(old)==5000
    initial=load(Path(s['r32_source'])/'S0000.pt');gsource={k:v for k,v in initial['model'].items() if k.startswith('generator.')}
    for j,(stage,rho) in enumerate((('R32_K1',1.),('R32_KQ',.25))):
        for i,(a,b,u) in enumerate(zip(steps[j*5000:(j+1)*5000],old,updates[j*5000:(j+1)*5000]),1):
            assert a['stage']==u['stage']==stage and a['step']==u['step']==i and u['index']==j*5000+i
            assert all(a[k]==b[k] for k in ('step','absolute_schedule_step','indices','flips'))
            assert abs(a['loss']-a['CE']-a['Dice_loss']-a['KLc']-rho*a['KLR'])<5e-6
        for step in range(0,5001,1000):
            q=load(root/stage/f'S{step:04}.pt');assert q['step']==step and q['residual_grid']==32 and q['lambda_Dice']==1 and q['KLc_weight']==1 and q['KLR_weight']==rho
            states=q['optimizer']['state'];assert (not states) if step==0 else all(int(v['step'])==step for v in states.values())
            if step==0:assert same(q['model'],initial['model']) and same(q['rng'],initial['rng'])
            assert same({k:v for k,v in q['model'].items() if k.startswith('generator.')},gsource)
            assert same({k:v for k,v in q['model'].items() if k.startswith('backbone.segmentation_head.')},{k:v for k,v in initial['model'].items() if k.startswith('backbone.segmentation_head.')})
    # Verify the actual deployed posterior grid without a G draw or parameter update.
    model=EndToEndG32(build_model(),SpatialResidualModel());model.load_state_dict(load(root/'R32_K1/S5000.pt')['model']);model.eval().requires_grad_(False)
    with torch.no_grad():shape=list(model.posterior(torch.zeros(1,1,128,128))[2].shape)
    assert shape==[1,16,32,32]
    seal=read(root/'PREDICTIONS_SEALED.json');assert seal['costs']['label_phase_reads']==0 and seal['cache_grid']==128 and seal['evaluation_grid']==384
    maps=dict(train=read(Path(s['train_readout_source'])/'PRIVATE_EVALUATION_MAP.json'),val=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json'));private=load(root/'PRIVATE_PHASE_RESULTS.pt');err=[0.]
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):
            e=abs(a-b);assert e<1e-10;err[0]=max(err[0],e)
        else:assert a==b
    controls=read(factor/'RESULTS.json')['aggregates'];prior=read(Path(s['r32_source'])/'RESULTS.json')['aggregates']
    for cohort,records in maps.items():
        recomputed={}
        for stage in ('R32_K1','R32_KQ'):
            rows=[];slices=0
            for i,row in enumerate(records):
                d=load(root/stage/cohort/f'CASE{i:03}.pt');p=d['native_probability'];slices+=len(p)
                assert p.dtype==torch.float32 and p.shape[1:]==(4,128,128) and torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6
                p=restore_probability(p)
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,d['spacing'])))
            assert slices==(230 if cohort=='train' else 544)
            compare(private[cohort][stage],rows);a=(aggregate_training if cohort=='train' else aggregate)(rows);compare(r['aggregates'][cohort][stage],a);recomputed[stage]=a
            for name,control in (('R32_K0',prior[cohort]),('U_D',controls[cohort]['U_D']),('R16_K1',controls[cohort]['G_D'])):
                compare(r['contrasts'][cohort][stage+'_minus_'+name],paired_summary(a,control))
            print('recomputed',cohort,stage,flush=True)
        compare(r['contrasts'][cohort]['R32_KQ_minus_R32_K1'],paired_summary(recomputed['R32_KQ'],recomputed['R32_K1']))
    assert r['source_frozen'] and not r['test_access'] and r['all_predictions_before_labels'] and r['original_elapsed_seconds']<86400
    result=dict(status='PASS',physical_updates=10000,full_states_checked=12,schedule_pairs=10000,initial_model_RNG_fresh_Adam_and_KLc_KLR_weights_checked=True,G_frozen=True,actual_R_shape=shape,native_prediction_sets_recomputed=132,maximum_metric_error=err[0],all_patient_contrasts_recomputed=True,audit_synthetic_encoder_images=1,audit_G_forward=0,new_parameter_updates=0,audit_reconstruction_phases=132,metric_formula='same frozen evaluator, independent saved-native-array recomputation')
    write(root/'AUDIT.json',result);print(json.dumps(result))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
