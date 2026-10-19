"""One completion audit; uses existing metrics/checkpoints, no new model forwards."""
import json
from pathlib import Path
import torch
from ..gyro_source_repair_v11.summarize import rows
from ..semantic_condition_v1.summarize import finite_tree
from ..main_style_ssl_v1.data import Accessor
from ..siemens_lonly_v1.run import aggregate,compare
from ..main_style_ssl_v1.runtime import write


def main(root):
    root=Path(root);cfg=json.loads((root/'SETTINGS_GYRO_low.json').read_text())
    out=root/'jobs/GYRO_low';q=root/'qualification/GYRO_low';ref=Path(cfg['CE_low_reference'])
    status=json.loads((root/'STATUS.json').read_text())['status']
    assert status in ['COMPLETE_SOURCE_GATES_PASS_NO_TARGET','SCIENTIFIC_STOP_SOURCE_GATE']
    assert json.loads((out/'EXIT.json').read_text())['exit_code']==0
    result=json.loads((out/'RESULTS.json').read_text());control=json.loads((ref/'RESULTS.json').read_text())
    physical=rows(out/'physical_updates.jsonl');engineering=rows(q/'engineering_updates.jsonl')
    assert len(physical)==69 and all(v['physical_call']==v['iteration']==i+1 for i,v in enumerate(physical))
    assert len(engineering)==32 and [v['physical_call'] for v in engineering]==list(range(1,33))
    events=rows(root/'budget_events.jsonl');tokens={k:[v for v in events if v['kind']==k] for k in ['reserved','executed']}
    assert all(len(v)==101 and sorted(x['token'] for x in v)==list(range(1,102)) for v in tokens.values())
    counts={}
    for v in tokens['executed']:counts[v['key']]=counts.get(v['key'],0)+1
    assert counts=={'engineering:GYRO_low':32,'formal:GYRO_low':69}
    assert json.loads((root/'budget_counter.json').read_text())['total']==101
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    latest=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    baseline=torch.load(ref/'initial.pt',map_location='cpu',weights_only=False)
    assert finite_tree(initial) and finite_tree(final) and finite_tree(latest)
    assert initial['optimizer']['state']=={} and baseline['optimizer']['state']=={}
    assert all(torch.equal(v,baseline['model'][k]) for k,v in initial['model'].items())
    assert all(x['step']==x['physical_updates']==69 for x in [final,latest])
    for key in ['model','ema']:
        assert all(torch.equal(v,latest[key][k]) for k,v in final[key].items())
    assert final['torch_rng'].equal(latest['torch_rng'])
    assert all(a.equal(b) for a,b in zip(final['cuda_rng'],latest['cuda_rng']))
    for k,d in final['optimizer']['state'].items():
        for name,v in d.items():assert torch.equal(v,latest['optimizer']['state'][k][name])
    assert final['reads']==latest['reads']
    assert torch.equal(initial['model']['frames'],final['model']['frames'])
    for name in ['parent.native.decoder.conv_logit.sigma.weight','parent.native.decoder.conv_logit.grad_update']:
        assert torch.equal(initial['model'][name],final['model'][name])
    assert any(not torch.equal(v,final['model'][k]) for k,v in initial['model'].items() if k.startswith('parent.'))
    assert any(not torch.equal(v,final['model'][k]) for k,v in initial['model'].items() if k.startswith('projector.'))
    pre=json.loads((out/'MATCHED_ENTRY.json').read_text());assert pre['status']=='PASS' and pre['both_momenta_empty']
    eng=json.loads((q/'ENGINEERING.json').read_text());assert eng['status']=='PASS' and eng['actual_calls']==32
    ev={}
    for role,name,field,n,v in [('val','private_student_val.jsonl','Siemens',23,46),('train_labeled','private_student_train.jsonl','source_train',10,20),('val','private_ema_val.jsonl','EMA_diagnostic',23,46)]:
        rr=rows(out/name);keys={(x['patient_id'],x['phase']) for x in Accessor(cfg['data'],'mnms','Siemens',role).rows}
        assert len(rr)==len(keys)==v and {(x['patient'],x['phase']) for x in rr}==keys
        a=aggregate(rr);assert a=={k:x for k,x in result[field].items() if k!='forward_slices'}
        assert a['patients']==n;ev[field]=rr
    ce=rows(ref/'private_student_val.jsonl');original=rows(Path(cfg['source_val_rows']))
    keys=lambda rr:{(x['patient'],x['phase']) for x in rr}
    assert keys(ce)==keys(original)==keys(ev['Siemens'])
    paired={}
    for label,base in [('CE_low',ce),('original',original)]:
        paired['GYRO_minus_'+label]=dict(overall=compare(base,ev['Siemens']),phases={p:compare([x for x in base if x['phase']==p],[x for x in ev['Siemens'] if x['phase']==p]) for p in ['ED','ES']})
    updates=rows(out/'private_update_diagnostics.jsonl');assert len(updates)==69
    assert [x['actual_call'] for x in updates]==list(range(1,70))
    assert all(lr==(.03 if k.startswith('projector') else .001) for x in updates for k,lr in x['lrs'].items())
    bystep={x['step']:x for x in rows(out/'private_loss_diagnostics.jsonl')};assert set(bystep)==set(range(69))
    ratios=[bystep[i]['encoder_weighted_ratio'] for i in range(69)]
    old=json.loads((ref.parent/'GYRO_low/FAILURE.json').read_text());assert old['physical_updates']==2
    oldsteps={x['step']:x for x in rows(ref.parent/'GYRO_low/private_loss_diagnostics.jsonl')}
    assert bystep[2]['encoder_weighted_ratio']==oldsteps[2]['encoder_weighted_ratio']
    assert max(ratios)>.25 and ratios[2]>.25
    for a in [root/'CONTROLLER_ACTIVE.json',*root.glob('ACTIVE_*.json')]:
        v=json.loads(a.read_text());p=Path('/proc')/str(v['pid'])/'stat'
        assert not p.exists() or p.read_text().split()[21]!=v['birth']
    public=dict(status='AUDIT_PASS',run_status=status,actual_optimizer_calls=101,formal_optimizer_calls=69,engineering_optimizer_calls=32,cap=101,unused=0,pending_reservations=0,budget_counts=counts,
        GYRO=result,CE_low=control,original_source=aggregate(original),paired=paired,engineering=eng,
        gradient_diagnostics=dict(above_old_threshold_steps=sum(x>.25 for x in ratios),max_weighted_ratio=max(ratios),old_third_step_reproduced=True,third_step_ratio=ratios[2],final_ratio=ratios[-1],
            native_update_path_length=sum(x['native_update_norm'] for x in updates),native_first_update=updates[0]['native_update_norm'],native_last_update=updates[-1]['native_update_norm']),
        audit=dict(latest_equals_final_including_optimizer_RNG=True,finite_states=True,matched_initial_exact=True,both_momenta_initially_empty=True,frame_exact=True,native_and_projector_changed=True,original_constants_unchanged=True,patient_EDES_sets_exact=True,all_workers_retired=True),
        validation_only=True,CE_low_reused_not_contemporaneous=True,BayeSeg_independent=True,CARE_independent=True,target_started=False,additional_patient_forwards=0,additional_optimizer_calls=0,old_budgets_separate=True)
    write(root/'PUBLIC_SUMMARY.json',public)
    print(json.dumps(dict(status='AUDIT_PASS',calls=101,dice=100*result['Siemens']['primary_dice'],delta_CE_low_pp=result['delta_from_CE_low_pp'],gradient=public['gradient_diagnostics'])))
