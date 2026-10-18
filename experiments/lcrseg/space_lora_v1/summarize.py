"""Audit private artifacts on the owning server; export anonymous aggregates only."""
import json
from pathlib import Path
import torch
from .core import ARMS, STEPS
from ..siemens_lonly_v1.run import aggregate, compare
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import write


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(root):
    root=Path(root); results={}; audits={}; evaluations={}; cfgs=[]
    source_rows=None; engineering=0
    continued=json.loads((root/'SETTINGS_dense_fixmatch.json').read_text()).get('protocol')=='space_lora_continuation_v1'
    endpoint=10000 if continued else STEPS
    start=4000 if continued else 0
    for arm in ARMS:
        out=root/'jobs'/arm
        cfg=json.loads((root/('SETTINGS_'+arm+'.json')).read_text()); cfgs.append(cfg)
        result=json.loads((out/'RESULTS.json').read_text())
        assert json.loads((out/'EXIT.json').read_text())['exit_code']==0
        assert result['status']=='COMPLETE' and result['step']==result['physical_updates']==endpoint
        ledger=rows(out/'physical_updates.jsonl')
        assert len(ledger)==endpoint-start
        assert all(v['physical_call']==start+i+1 and v['optimizer_executed'] for i,v in enumerate(ledger))
        state=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
        initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
        assert state['step']==state['physical_updates']==endpoint and state['arm']==arm
        assert all(torch.isfinite(v).all() for key in ['model','ema'] for v in state[key].values() if v.is_floating_point())
        preflight=json.loads((out/'PREFLIGHT.json').read_text())
        assert preflight['status']=='PASS' and preflight['source_entry_max_abs_error']==0
        base_frozen=None; residual=None
        if arm!='dense_fixmatch':
            frozen=[k for k in initial['model'] if not k.endswith(('.A','.B'))]
            base_frozen=all(torch.equal(initial['model'][k],state['model'][k]) for k in frozen)
            assert base_frozen
            residual=0.
            for key,a in state['model'].items():
                if not key.endswith('.A'): continue
                path=key[:-2]
                from ..lctx_weight_memory_v0_1.core import LAYERS
                index=LAYERS.index(path.removeprefix('parent.native.')+'.weight')
                v=state['model']['parent.space_basis_'+str(index)].double()
                residual=max(residual,float((a.double()@v).norm())/(float(a.norm())+1e-12))
            assert residual<2e-5
        q=root/'qualification'/arm
        if continued:
            eng=json.loads((q/'RESUME_PREFLIGHT.json').read_text())
            assert eng['status']=='PASS' and eng['optimizer_updates']==0
        else:
            eng=json.loads((q/'ENGINEERING.json').read_text())
            assert eng['status']=='PASS' and eng['physical_optimizer_updates']==32
            assert len(rows(q/'engineering_updates.jsonl'))==32
            engineering+=32
        eval_rows={}
        for domain,role,name,field,patients,volumes in [
            ('Philips','val','private_Philips_val.jsonl','Philips',31,62),
            ('Siemens','val','private_Siemens_val.jsonl','Siemens',23,46),
            ('Philips','train_labeled','private_train.jsonl','train',13,26),
            ('Siemens','train_labeled','private_Siemens_train.jsonl','source_train',10,20)]:
            rr=rows(out/name); summary=aggregate(rr)
            expected={(r['patient_id'],r['phase']) for r in Accessor(cfg['data'],'mnms',domain,role).rows}
            assert {(r['patient'],r['phase']) for r in rr}==expected
            assert summary['patients']==patients and summary['volumes']==volumes
            assert abs(summary['primary_dice']-result[field]['primary_dice'])<1e-12
            eval_rows[field]=rr
        evaluations[arm]=eval_rows
        costs=json.loads((out/'COSTS.json').read_text())
        assert costs['physical_optimizer_updates']==endpoint
        if continued:
            assert costs['new_optimizer_updates']==6000 and costs['prior_optimizer_updates']==4000
        assert costs['L_batch_reads']==endpoint and costs['U_batch_reads']==7*endpoint
        assert not result['independent_test_access']
        results[arm]=result
        audits[arm]=dict(exit_code=0,formal_optimizer_calls=len(ledger),engineering=eng,
            final_checkpoint_readable_finite=True,base_frozen=base_frozen,
            projection_relative_residual=residual,preflight=preflight,costs=costs,
            validation_and_training_patient_phase_sets_exact=True)
        if source_rows is None: source_rows=rows(Path(cfg['source_val_rows']))
    for cfg in cfgs[1:]:
        assert {k:v for k,v in cfg.items() if k not in ['arm','output','resume_checkpoint']}=={
            k:v for k,v in cfgs[0].items() if k not in ['arm','output','resume_checkpoint']}
        if continued:
            assert Path(cfg['resume_checkpoint']).parents[2]==Path(cfgs[0]['resume_checkpoint']).parents[2]
    source=aggregate(source_rows)
    assert source['patients']==23 and source['volumes']==46
    assert abs(source['primary_dice']-.8036740663770262)<1e-12
    pairs={}
    for reference in ['dense_fixmatch','free_lora']:
        pairs['space_minus_'+reference]={}
        for domain in ['Philips','Siemens','train','source_train']:
            a=evaluations[reference][domain]; b=evaluations['space_lora'][domain]
            pairs['space_minus_'+reference][domain]=dict(overall=compare(a,b),phases={
                phase:compare([r for r in a if r['phase']==phase],[r for r in b if r['phase']==phase])
                for phase in ['ED','ES']})
    for arm in ARMS:
        assert compare(source_rows,evaluations[arm]['Siemens'])==results[arm]['paired_old']
    stats=json.loads(Path(cfgs[0]['statistics']).with_suffix('.json').read_text())
    assert stats['source_patients']==10 and stats['source_slices']==230 and stats['optimizer_updates']==0
    public=dict(source=source,arms=results,paired=pairs,allocation=stats,
        audit=audits,formal_optimizer_updates=3*(endpoint-start),engineering_optimizer_updates=engineering,
        total_new_optimizer_calls=3*(endpoint-start)+engineering,independent_test_access=False,
        source_zero_shot_Philips_not_evaluated=True,
        source_artifact_updates=43600,source_and_target_training_groups_distinct=True)
    if continued:
        prior=Path(cfgs[0]['resume_checkpoint']).parents[2]
        changes={}
        for arm in ARMS:
            changes[arm]={}
            for field,name in [('Philips','private_Philips_val.jsonl'),('Siemens','private_Siemens_val.jsonl'),
                               ('train','private_train.jsonl'),('source_train','private_Siemens_train.jsonl')]:
                a=rows(prior/'jobs'/arm/name);b=evaluations[arm][field]
                changes[arm][field]=dict(overall=compare(a,b),phases={phase:compare(
                    [r for r in a if r['phase']==phase],[r for r in b if r['phase']==phase]) for phase in ['ED','ES']})
        public.update(change_4k_to_10k=changes,prior_endpoint=json.loads((prior/'PUBLIC_SUMMARY.json').read_text())['arms'],
                      prior_formal_calls=12000,prior_engineering_calls=96,cumulative_optimizer_calls=30096)
    write(root/'PUBLIC_SUMMARY.json',public)
    print(json.dumps(dict(status='AUDIT_PASS',formal_updates=3*(endpoint-start),engineering_updates=engineering,
          paired=pairs),indent=2))
