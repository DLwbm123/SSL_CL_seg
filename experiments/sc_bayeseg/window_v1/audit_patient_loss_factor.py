"""Audit factorial source pairing, physical updates and sealed native predictions."""
import json,os
from pathlib import Path
import torch,numpy as np,h5py
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.patient_validation import phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training
from sc_bayeseg.window_v1.patient_loss_factor import ARMS,restore_probability,paired_summary
from sc_bayeseg.window_v1.r_start import write


def audit(s):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;root=Path(s['output']);parent=Path(s['patient_control']);e2e=Path(s['patient_e2e_source']);train=Path(s['train_readout_source'])
    read=lambda p:json.loads(p.read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    c=r['costs'];assert (c['optimizer'],c['reverse'],c['training_encoder_images'],c['train_theta'],c['train_g_forward'],c['train_g_image_forward'])==(20000,20000,40000,30000,60000,120000)
    assert (c['evaluation_encoder_images'],c['evaluation_g_forward'],c['evaluation_theta'],c['evaluation_g_image_forward'],c['label_phase_reads'],c['reconstruction_phases'])==(3096,19104,4776,37152,264,264)
    updates=[json.loads(x) for x in (root/'physical_updates.jsonl').read_text().splitlines()];steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()]
    old=[json.loads(x) for x in (e2e/'steps.jsonl').read_text().splitlines() if json.loads(x)['stage']=='DIRECT_E2E'];assert len(updates)==len(steps)==20000 and len(old)==5000
    gsource=load(s['checkpoint'])['model'];rng=load(parent/'PRETRAIN/S5000.pt')['rng'];loss_error=0.
    for j,(name,(lamb,beta)) in enumerate(ARMS.items()):
        initial=load(e2e/('DIRECT_E2E' if name=='U_D' else 'CONDITIONAL_G_E2E')/'S0000.pt')['model']
        for i in range(5000):
            a=steps[j*5000+i];u=updates[j*5000+i];assert a['stage']==u['stage']==name and a['step']==u['step']==i+1 and u['index']==j*5000+i+1
            assert all(a[k]==old[i][k] for k in ('step','absolute_schedule_step','indices','flips'))
            expected=a['CE']+lamb*a['Dice_loss']+(0 if beta is None else beta)*(a['KLc']+a['KLR']);loss_error=max(loss_error,abs(expected-a['loss']));assert abs(expected-a['loss'])<5e-6
        for step in range(0,5001,1000):
            q=load(root/name/f'S{step:04}.pt');assert q['step']==step and q['stage']==name and q['lambda_Dice']==lamb and q['beta_KL']==beta
            opt=q['optimizer']['state'];assert (not opt) if step==0 else all(int(v['step'])==step for v in opt.values())
            if step==0:assert same(q['model'],initial) and same(q['rng'],rng)
            assert same({k:v for k,v in q['model'].items() if k.startswith('backbone.segmentation_head.')},{k:v for k,v in initial.items() if k.startswith('backbone.segmentation_head.')})
            if name!='U_D':assert same({k[10:]:v for k,v in q['model'].items() if k.startswith('generator.')},gsource)
            if step:assert not same(q['model'],initial)
    seal=read(root/'PREDICTIONS_SEALED.json');assert seal['costs']['label_phase_reads']==0 and seal['cache_grid']==128 and seal['evaluation_grid']==384
    maps=dict(train=read(train/'PRIVATE_EVALUATION_MAP.json'),val=read(parent/'PRIVATE_EVALUATION_MAP.json'));private=load(root/'PRIVATE_PHASE_RESULTS.pt');err=[0.];aggregates={}
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
    for cohort,records in maps.items():
        aggregates[cohort]={}
        for name in ARMS:
            rows=[];slices=0
            for i,row in enumerate(records):
                d=load(root/name/cohort/f'CASE{i:03}.pt');p=d['native_probability'];slices+=len(p)
                assert p.dtype==torch.float32 and p.shape[1:]==(4,128,128) and torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6
                p=restore_probability(p)
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,d['spacing'])))
            assert slices==(230 if cohort=='train' else 544)
            compare(private[cohort][name],rows);aggregates[cohort][name]=(aggregate_training if cohort=='train' else aggregate)(rows);compare(r['aggregates'][cohort][name],aggregates[cohort][name]);print('recomputed',cohort,name,flush=True)
    controls=dict(train=read(train/'RESULTS.json')['training'],val=read(e2e/'RESULTS.json')['aggregates'])
    for cohort,a in aggregates.items():
        u=controls[cohort]['E2E_DIRECT' if cohort=='train' else 'DIRECT_E2E'];g=controls[cohort]['E2E_G0' if cohort=='train' else 'CONDITIONAL_G_E2E'];expected={}
        for name in ARMS:expected[name+'_minus_original']=paired_summary(a[name],u if name=='U_D' else g)
        for left,right in (('G_D','U_D'),('G_DK0','U_D'),('G_DK0','G_K0'),('G_DK0','G_D')):expected[left+'_minus_'+right]=paired_summary(a[left],a[right])
        vec=lambda a:np.array([v['macro_Dice'] for v in a['patients']])
        for name,d in dict(Dice_KL_interaction=(vec(a['G_DK0'])-vec(a['G_K0']))-(vec(a['G_D'])-vec(g)),matched_Dice_gain_interaction=(vec(a['G_D'])-vec(g))-(vec(a['U_D'])-vec(u))).items():
            rg=np.random.default_rng(163);boot=np.mean(d[rg.integers(0,len(d),size=(10000,len(d)))],axis=1);expected[name]=dict(mean_Dice_difference=float(d.mean()),patient_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
        compare(r['contrasts'][cohort],expected)
    assert r['source_frozen'] and r['all_predictions_before_labels'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    out=dict(status='PASS',physical_updates=20000,full_states_checked=24,schedule_pairs_with_original_control=20000,initial_models_RNG_and_fresh_Adam_checked=True,G_frozen_all_snapshots=True,actual_lambda_beta_and_objective_checked=True,maximum_logged_loss_rounding_error=loss_error,native_prediction_sets_reconstructed_and_recomputed=264,maximum_metric_error=err[0],all_paired_and_factor_interactions_recomputed=True,audit_reconstruction_phases=264,new_network_forwards=0,new_parameter_updates=0,metric_formula='same frozen evaluator, independent sealed-native-array recomputation')
    write(root/'AUDIT.json',out);print(json.dumps(out))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
