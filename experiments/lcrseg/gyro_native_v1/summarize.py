"""One completion audit, no new image forward or optimizer call; public aggregates only."""
import json
from pathlib import Path
import torch
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import write
from ..siemens_lonly_v1.run import aggregate,compare
from ..semantic_condition_v1.summarize import finite_tree


def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def main(root):
    root=Path(root)
    assert json.loads((root/'STATUS.json').read_text())['status']=='SCIENTIFIC_STOP_SOURCE_GATE'
    cfg=json.loads((root/'SETTINGS_source_prepare.json').read_text())
    qcfg=json.loads((root/'QUAL_source_prepare.json').read_text())
    out=root/'source';q=Path(qcfg['output'])
    assert json.loads((out/'EXIT.json').read_text())['exit_code']==0
    result=json.loads((out/'RESULTS.json').read_text())
    assert result['physical_updates']==result['step']==138 and not result['source_gate_passed']
    ledger=rows(out/'physical_updates.jsonl')
    assert len(ledger)==138 and all(v['physical_call']==v['iteration']==i+1 for i,v in enumerate(ledger))
    eng=json.loads((q/'ENGINEERING.json').read_text())
    assert eng['status']=='PASS' and eng['physical_optimizer_updates']==32
    assert len(rows(q/'engineering_updates.jsonl'))==32
    first_failure=json.loads((root/'qualification/source_prepare/FAILURE.json').read_text())
    assert first_failure['physical_updates']==0
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    state=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    original=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=False)
    assert finite_tree(state) and state['step']==state['physical_updates']==138
    assert original['step']==original['physical_updates']==43600 and original['method']=='fixmatch'
    assert all(torch.equal(original['ema'][k],initial['model'][k]) for k in original['ema'])
    assert all(torch.equal(v,state['ema'][k]) for k,v in state['model'].items())
    assert torch.equal(initial['model']['frames'],state['model']['frames'])
    evaluations={}
    for role,name,field,patients,volumes in [
        ('val','private_Siemens_val.jsonl','Siemens',23,46),
        ('train_labeled','private_Siemens_train.jsonl','source_train',10,20)]:
        rr=rows(out/name);expected={(r['patient_id'],r['phase']) for r in Accessor(cfg['data'],'mnms','Siemens',role).rows}
        assert len(rr)==len(expected) and {(r['patient'],r['phase']) for r in rr}==expected
        summary=aggregate(rr);reported={k:v for k,v in result[field].items() if k!='forward_slices'}
        assert summary==reported and summary['patients']==patients and summary['volumes']==volumes
        evaluations[field]=rr
    baseline=rows(Path(cfg['source_val_rows']));base=aggregate(baseline)
    assert abs(base['primary_dice']-.8036740663770262)<1e-12
    assert {(r['patient'],r['phase']) for r in baseline}=={(r['patient'],r['phase']) for r in evaluations['Siemens']}
    pairs=dict(overall=compare(baseline,evaluations['Siemens']),phases={phase:compare([r for r in baseline if r['phase']==phase],[r for r in evaluations['Siemens'] if r['phase']==phase]) for phase in ['ED','ES']})
    events=rows(root/'budget_events.jsonl')
    reserved=[r for r in events if r['kind']=='reserved'];executed=[r for r in events if r['kind']=='executed']
    assert len(reserved)==len(executed)==170
    assert sorted(r['token'] for r in reserved)==sorted(r['token'] for r in executed)==list(range(1,171))
    counts={}
    for r in executed:counts[r['key']]=counts.get(r['key'],0)+1
    assert counts=={'engineering:source_prepare':32,'formal:source_prepare':138}
    assert json.loads((root/'budget_counter.json').read_text())['total']==170
    assert not Path(cfg['bank']).exists() and not (root/'jobs').exists()
    for active in [root/'CONTROLLER_ACTIVE.json',root/'CONTROLLER_ACTIVE_CONTRACT_FIX.json',*root.glob('ACTIVE_*.json'),*(root/'contract_fix_01').glob('ACTIVE_*.json')]:
        r=json.loads(active.read_text());p=Path('/proc')/str(r['pid'])/'stat'
        assert not p.exists() or p.read_text().split()[21]!=r['birth']
    public=dict(status='AUDIT_PASS_SCIENTIFIC_STOP_SOURCE_GATE',source_final=result,original_source=base,
        paired_source=pairs,engineering=eng,source_preflight=json.loads((out/'PREFLIGHT.json').read_text()),
        actual_optimizer_calls=170,formal_optimizer_calls=138,engineering_optimizer_calls=32,
        cap_actual_optimizer_calls=10330,unused_authorized_calls=10160,pending_reservations=0,
        budget_ledger_counts=counts,source_historical_updates_separate=43600,
        original_source_immutable_entry_verified=True,final_student_equals_source_ema=True,
        final_checkpoint_finite=True,source_frames_exact=True,patient_phase_sets_exact=True,
        all_workers_retired=True,formal_target_arms_started=0,source_reservoir_generated=False,
        author_code_executed=False,readout_binding=False,pseudo_class_references=False,
        CARE_independent=True,validation_only=True,test_access=False,U_hidden_GT_access=False,
        additional_patient_forwards=0,additional_optimizer_calls=0,
        engineering_initial_contract_failure_updates=0)
    write(root/'PUBLIC_SUMMARY.json',public)
    print(json.dumps(public,ensure_ascii=False,indent=2))
