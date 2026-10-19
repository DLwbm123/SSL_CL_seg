"""Read private completion receipts once; export anonymous aggregates only."""
import json
from pathlib import Path
import torch
from .core import ARMS, STEPS
from ..siemens_lonly_v1.run import aggregate, compare
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import write


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def finite_tree(value):
    if isinstance(value,torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value,dict): return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)): return all(finite_tree(v) for v in value)
    return True


def main(root):
    root=Path(root)
    assert json.loads((root/'STATUS.json').read_text())['status']=='COMPLETE'
    results={};audits={};evaluations={};cfgs=[];initials={}
    expected_keys={}
    for arm in ARMS:
        out=root/'jobs'/arm
        cfg=json.loads((root/('V2_SETTINGS_'+arm+'.json')).read_text());cfgs.append(cfg)
        result=json.loads((out/'RESULTS.json').read_text())
        assert json.loads((out/'EXIT.json').read_text())['exit_code']==0
        assert result['status']=='COMPLETE' and result['step']==result['physical_updates']==STEPS
        ledger=rows(out/'physical_updates.jsonl')
        assert len(ledger)==STEPS
        assert all(v['physical_call']==v['iteration']==i+1 and v['optimizer_executed'] for i,v in enumerate(ledger))
        expected_keys['formal:'+arm]=STEPS
        state=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
        initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
        assert state['step']==state['physical_updates']==STEPS and state['arm']==arm
        assert finite_tree(state)
        frozen=None
        if arm!='dense_fixmatch':
            frozen=[k for k in initial['model'] if k.startswith('parent.') or k=='conditioner.prototypes']
            assert all(torch.equal(initial['model'][k],state['model'][k]) and
                       torch.equal(initial['ema'][k],state['ema'][k]) for k in frozen)
            frozen=True
        initials[arm]=initial['model']
        pre=json.loads((out/'PREFLIGHT.json').read_text())
        assert pre['status']=='PASS' and pre['source_entry_max_abs_error']==0
        for term in ['L']+([] if arm.endswith('_lonly') else ['U']):
            assert pre['gradient_norms'][term]>0
        qcfg=json.loads((root/('V2_QUAL_'+arm+'.json')).read_text())
        q=Path(qcfg['output'])
        eng=json.loads((q/'ENGINEERING.json').read_text())
        assert eng['status']=='PASS' and eng['physical_optimizer_updates']==32
        assert len(rows(q/'engineering_updates.jsonl'))==32
        expected_keys[qcfg['phase']+':'+arm]=32
        eval_rows={}
        for domain,role,name,field,patients,volumes in [
            ('Philips','val','private_Philips_val.jsonl','Philips',31,62),
            ('Siemens','val','private_Siemens_val.jsonl','Siemens',23,46),
            ('Philips','train_labeled','private_train.jsonl','train',13,26),
            ('Siemens','train_labeled','private_Siemens_train.jsonl','source_train',10,20)]:
            rr=rows(out/name); summary=aggregate(rr)
            expected={(r['patient_id'],r['phase']) for r in Accessor(cfg['data'],'mnms',domain,role).rows}
            assert len(rr)==len(expected) and {(r['patient'],r['phase']) for r in rr}==expected
            assert summary['patients']==patients and summary['volumes']==volumes
            reported={k:v for k,v in result[field].items() if k!='forward_slices'}
            assert summary==reported
            eval_rows[field]=rr
        evaluations[arm]=eval_rows
        costs=json.loads((out/'COSTS.json').read_text())
        # This pilot reused the original non-continuation engine snapshot;
        # its COSTS schema predates explicit prior/new split fields.
        assert not cfg.get('resume_checkpoint')
        assert costs['physical_optimizer_updates']==costs.get('new_optimizer_updates',STEPS)==STEPS
        assert costs.get('prior_optimizer_updates',0)==0 and costs['L_batch_reads']==STEPS
        assert costs['U_batch_reads']==(0 if arm.endswith('_lonly') else 7*STEPS)
        assert not result['independent_test_access']
        results[arm]=result
        audits[arm]=dict(exit_code=0,formal_optimizer_calls=STEPS,
            final_checkpoint_readable_finite=True,immutable_native_head_prototypes_exact=frozen,
            preflight=pre,engineering=eng,costs=costs,patient_phase_sets_exact=True)
        del state,initial
    for cfg in cfgs[1:]:
        assert {k:v for k,v in cfg.items() if k not in ['arm','output']}=={
            k:v for k,v in cfgs[0].items() if k not in ['arm','output']}
    native=[k for k in initials['dense_fixmatch'] if k.startswith('parent.native.')]
    for arm in ARMS[1:]:
        assert all(torch.equal(initials[arm][k],initials['dense_fixmatch'][k]) for k in native)
        assert all(torch.equal(initials[arm][k],initials['aligned_lonly'][k])
                   for k in initials[arm] if k.startswith('conditioner.'))
    source_rows=rows(Path(cfgs[0]['source_val_rows']))
    source=aggregate(source_rows)
    assert source['patients']==23 and source['volumes']==46
    assert abs(source['primary_dice']-.8036740663770262)<1e-12
    saved_source=torch.load(cfgs[0]['source_checkpoint'],map_location='cpu',weights_only=False)
    assert saved_source['step']==saved_source['physical_updates']==43600 and saved_source['method']=='fixmatch'
    assert all(torch.equal(saved_source['ema'][k],initials['dense_fixmatch'][k]) for k in native)
    for arm in ARMS:
        assert compare(source_rows,evaluations[arm]['Siemens'])==results[arm]['paired_old']
    stat_path=Path(cfgs[0]['statistics'])
    stats=torch.load(stat_path,map_location='cpu',weights_only=False)
    assert stats['source_slices']==230 and stats['source_patients']==10 and stats['optimizer_updates']==0
    assert stats['prototypes'].shape==(4,3,144) and finite_tree(stats)
    p=torch.nn.functional.normalize(stats['prototypes'],dim=-1)
    assert torch.equal(initials['aligned_lonly']['conditioner.prototypes'],p)
    # Include the failed, discarded v1 qualification and all restored-step replays.
    failed=root/'qualification/aligned_lonly'
    failure=json.loads((failed/'ENGINEERING_FAILURE.json').read_text())
    assert failure['physical_optimizer_updates']==32 and len(rows(failed/'engineering_updates.jsonl'))==32
    expected_keys['engineering:aligned_lonly']=32
    events=rows(root/'budget_events.jsonl')
    reserved=[v for v in events if v['kind']=='reserved'];executed=[v for v in events if v['kind']=='executed']
    assert len(reserved)==len(executed)==17692
    assert sorted(v['token'] for v in reserved)==list(range(1,17693))
    assert sorted(v['token'] for v in executed)==list(range(1,17693))
    count={}
    for v in executed: count[v['key']]=count.get(v['key'],0)+1
    assert count==expected_keys
    process_checks={}
    for active in [root/'CONTROLLER_ACTIVE_V2.json',*(root/'unit_slope_v2').glob('ACTIVE_*.json')]:
        identity=json.loads(active.read_text()); path=Path('/proc')/str(identity['pid'])/'stat'
        same=path.exists() and path.read_text().split()[21]==identity['birth']
        assert not same, 'worker/controller has not retired'
        process_checks['controller' if active.name=='CONTROLLER_ACTIVE_V2.json' else active.stem]=True
    pairs={}
    for a,b in [('aligned_lonly','aligned_ssl'),('shuffled_lonly','shuffled_ssl'),
                ('shuffled_ssl','aligned_ssl'),('shuffled_lonly','aligned_lonly'),('dense_fixmatch','aligned_ssl')]:
        key=b+'_minus_'+a
        pairs[key]={field:dict(overall=compare(evaluations[a][field],evaluations[b][field]),
            phases={phase:compare([r for r in evaluations[a][field] if r['phase']==phase],
                                  [r for r in evaluations[b][field] if r['phase']==phase]) for phase in ['ED','ES']})
                    for field in ['Siemens','Philips','train','source_train']}
    old_interaction=100*((results['aligned_ssl']['Siemens']['primary_dice']-results['aligned_lonly']['Siemens']['primary_dice'])-
                         (results['shuffled_ssl']['Siemens']['primary_dice']-results['shuffled_lonly']['Siemens']['primary_dice']))
    ssl_gain=100*(results['aligned_ssl']['Philips']['primary_dice']-results['aligned_lonly']['Philips']['primary_dice'])
    old_semantic=100*(results['aligned_ssl']['Siemens']['primary_dice']-results['shuffled_ssl']['Siemens']['primary_dice'])
    new_semantic=100*(results['aligned_ssl']['Philips']['primary_dice']-results['shuffled_ssl']['Philips']['primary_dice'])
    target_dense=100*(results['aligned_ssl']['Philips']['primary_dice']-results['dense_fixmatch']['Philips']['primary_dice'])
    gates=dict(target_ssl_gain_pp=dict(value=ssl_gain,threshold=.5,passed=ssl_gain>=.5),
        old_semantic_gain_pp=dict(value=old_semantic,threshold=.5,passed=old_semantic>=.5),
        new_semantic_nonnegative_pp=dict(value=new_semantic,threshold=0,passed=new_semantic>=0),
        old_U_interaction_pp=dict(value=old_interaction,threshold=.5,passed=old_interaction>=.5),
        target_vs_dense_pp=dict(value=target_dense,threshold=-.5,passed=target_dense>=-.5))
    public=dict(status='AUDIT_PASS',method_revision='unit_slope_bounded_v2',source=source,arms=results,
        audits=audits,paired=pairs,scientific_gates=gates,all_joint_gates_passed=all(v['passed'] for v in gates.values()),
        formal_optimizer_updates=17500,engineering_optimizer_updates=192,total_new_optimizer_calls=17692,
        unused_authorized_updates=2308,budget_cap=20000,budget_ledger_counts=count,
        source_artifact_updates=43600,source_updates_separate=True,
        validation_only=True,independent_test_access=False,additional_patient_forwards=0,
        additional_optimizer_calls=0,source_zero_shot_Philips_not_evaluated=True,
        source_preparation={k:v for k,v in stats.items() if k!='prototypes'},
        engineering_failed_v1=failure,all_workers_and_controller_retired=True)
    write(root/'PUBLIC_SUMMARY.json',public)
    print(json.dumps(dict(status='AUDIT_PASS',actual_calls=17692,scientific_gates=gates),indent=2))
