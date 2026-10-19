import json
from pathlib import Path
import torch
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import write
from ..siemens_lonly_v1.run import aggregate,compare
from ..semantic_condition_v1.summarize import finite_tree
from .core import ARMS


def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]


def main(root):
    root=Path(root);status=json.loads((root/'STATUS.json').read_text())
    assert status['status']=='BLOCKED'
    failure=json.loads((root/'jobs/GYRO_low/FAILURE.json').read_text())
    assert failure['step']==failure['physical_updates']==2 and 'preset encoder gradient ratio gate' in failure['error']
    cfg=json.loads((root/'SETTINGS_calibration.json').read_text())
    results={};initials={};audits={};ev={};expected={}
    for arm in ARMS:
        out=root/'jobs'/arm;q=root/'qualification'/arm
        partial=arm=='GYRO_low'
        nsteps=2 if partial else 69
        assert json.loads((out/'EXIT.json').read_text())['exit_code']==(1 if partial else 0)
        ledger=rows(out/'physical_updates.jsonl')
        assert len(ledger)==nsteps and all(v['physical_call']==v['iteration']==i+1 for i,v in enumerate(ledger))
        eng=json.loads((q/'ENGINEERING.json').read_text());assert eng['status']=='PASS' and eng['actual_calls']==32
        assert len(rows(q/'engineering_updates.jsonl'))==32
        expected['engineering:'+arm]=32;expected['formal:'+arm]=nsteps
        initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
        final=torch.load(out/'final.pt',map_location='cpu',weights_only=False) if not partial else None
        assert initial['optimizer']['state']=={} and finite_tree(initial)
        if not partial:
            assert final['step']==final['physical_updates']==69
            assert finite_tree(final) and torch.equal(initial['model']['frames'],final['model']['frames'])
        else:
            assert not (out/'final.pt').exists() and not (out/'RESULTS.json').exists()
        initials[arm]=initial['model']
        if arm=='calibration':
            assert all(torch.equal(v,final['model'][k]) for k,v in initial['model'].items() if k.startswith('parent.'))
            assert any(not torch.equal(v,final['model'][k]) for k,v in initial['model'].items() if k.startswith('projector.'))
            calibrated=final['model']
        else:
            assert all(torch.equal(v,calibrated[k]) for k,v in initial['model'].items())
        result=json.loads((out/'RESULTS.json').read_text()) if not partial else dict(status='STOPPED_GRADIENT_RATIO_BEFORE_THIRD_CALL',arm=arm,step=2,physical_updates=2,fixed_endpoint_dice_available=False,partial_weight_checkpoint_available=False)
        results[arm]=result
        assert result['step']==result['physical_updates']==nsteps
        pre=json.loads((out/'PREFLIGHT.json').read_text());assert pre['status']=='PASS' and pre['momentum_empty']
        dd=rows(out/'private_update_diagnostics.jsonl');assert len(dd)==nsteps
        assert all(v['actual_call']==i+1 for i,v in enumerate(dd))
        native_lr=.03 if arm=='CE_high' else .001
        for v in dd:
            for name,lr in v['lrs'].items():assert lr==(.03 if name.startswith('projector') else native_lr)
        if arm=='calibration':assert all(v['native_update_norm']==0 for v in dd)
        if arm in ['CE_high','CE_low']:assert all(v['projector_update_norm']==0 for v in dd)
        ld=rows(out/'private_loss_diagnostics.jsonl')
        ratios=[v['encoder_weighted_ratio'] for v in ld]
        if partial:
            bystep={v['step']:v for v in ld}
            assert all(0<bystep[i]['encoder_weighted_ratio']<=.25 for i in [0,1])
            assert bystep[2]['encoder_weighted_ratio']>.25
            rejected_step={k:v for k,v in bystep[2].items() if k!='arm'}
        audits[arm]=dict(formal=nsteps,engineering=32,exit_code=1 if partial else 0,finite_checkpoint=True if not partial else None,initial_momentum_empty=True,
            frames_exact=True if not partial else None,common_entry_exact=True,engineering_receipt=eng,
            initial_encoder_ratio=pre['encoder_weighted_ratio'],max_encoder_weighted_ratio=max(ratios),
            native_update_norm_first=dd[0]['native_update_norm'],native_update_norm_last=dd[-1]['native_update_norm'],
            native_update_path_length=sum(v['native_update_norm'] for v in dd),
            projector_update_norm_first=dd[0]['projector_update_norm'],group_lrs_exact=True)
        if arm not in ['calibration','GYRO_low']:
            rr={}
            for role,name,field,n,p in [('val','private_student_val.jsonl','Siemens',23,46),('train_labeled','private_student_train.jsonl','source_train',10,20),('val','private_ema_val.jsonl','EMA_diagnostic',23,46)]:
                r=rows(out/name);keys={(v['patient_id'],v['phase']) for v in Accessor(cfg['data'],'mnms','Siemens',role).rows}
                assert len(r)==len(keys) and {(v['patient'],v['phase']) for v in r}==keys
                a=aggregate(r);assert a=={k:v for k,v in result[field].items() if k!='forward_slices'}
                assert a['patients']==n and a['volumes']==p
                rr[field]=r
            ev[arm]=rr
    original=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=False)
    assert original['step']==original['physical_updates']==43600
    assert all(torch.equal(v,initials['calibration'][k]) for k,v in original['ema'].items())
    events=rows(root/'budget_events.jsonl');reserved=[v for v in events if v['kind']=='reserved'];executed=[v for v in events if v['kind']=='executed']
    assert len(reserved)==len(executed)==337
    assert sorted(v['token'] for v in reserved)==sorted(v['token'] for v in executed)==list(range(1,338))
    count={}
    for v in executed:count[v['key']]=count.get(v['key'],0)+1
    assert count==expected and json.loads((root/'budget_counter.json').read_text())['total']==337
    for active in [root/'CONTROLLER_ACTIVE.json',*root.glob('ACTIVE_*.json')]:
        a=json.loads(active.read_text());p=Path('/proc')/str(a['pid'])/'stat'
        assert not p.exists() or p.read_text().split()[21]!=a['birth']
    baseline=rows(Path(cfg['source_val_rows']));base=aggregate(baseline)
    paired={}
    for a,b in [('original','CE_high'),('original','CE_low'),('CE_high','CE_low')]:
        x=baseline if a=='original' else ev[a]['Siemens'];y=ev[b]['Siemens']
        paired[b+'_minus_'+a]=dict(overall=compare(x,y),phases={phase:compare([r for r in x if r['phase']==phase],[r for r in y if r['phase']==phase]) for phase in ['ED','ES']})
    public=dict(status='AUDIT_PASS',run_status='SCIENTIFIC_STOP_GRADIENT_RATIO',controller_status_raw=status['status'],arms=results,audits=audits,paired=paired,original_source=base,
        actual_optimizer_calls=337,formal_optimizer_calls=209,engineering_optimizer_calls=128,cap=404,unused=67,
        reservations_unconfirmed=0,budget_ledger_counts=count,all_workers_retired=True,patient_phase_sets_exact=True,
        stage_boundary_all_momenta_reset=True,calibrated_weights_and_BN_retained=True,native_calibration_exact=True,
        student_is_primary=True,EMA_only_diagnostic=True,all_source_gates_passed=None,GYRO_vs_CE_low_gate_evaluable=False,rejected_step_diagnostic=rejected_step,gradient_guard_triggered=True,partial_terminal_checkpoint_missing=True,
        target_training_started=False,target_start_authorized=False,source_historical_updates_separate=43600,
        old_GYRO170_separate=True,BayeSeg_independent=True,CARE_independent=True,
        validation_only=True,test_access=False,U_hidden_GT_access=False,additional_patient_forwards=0,additional_optimizer_calls=0)
    write(root/'PUBLIC_SUMMARY.json',public)
    print(json.dumps(public,ensure_ascii=False,indent=2))
