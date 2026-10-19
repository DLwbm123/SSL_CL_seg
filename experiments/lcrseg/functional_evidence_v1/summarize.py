"""Audit the actual plasticity stop; no new evaluation or optimizer calls."""
import json
from pathlib import Path
import torch
from .core import ARMS,STEPS
from ..siemens_lonly_v1.run import aggregate,compare
from ..main_style_ssl_v1.data import Accessor


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def finite(x):
    if isinstance(x,torch.Tensor): return not x.is_floating_point() or bool(torch.isfinite(x).all())
    if isinstance(x,dict): return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)): return all(finite(v) for v in x)
    return True


def main(cfg):
    root=Path(cfg['root'])
    assert json.loads((root/'STATUS.json').read_text())['status']=='SCIENTIFIC_STOP_PLASTICITY'
    output=root/'PUBLIC_SUMMARY.json'
    if output.exists(): raise FileExistsError('completed audit already exists')
    results={}; audited={}; evaluated={}; engineering={}; configs={}
    for arm in ARMS:
        q=root/'qualification'/arm
        eng=json.loads((q/'ENGINEERING.json').read_text())
        ledger=rows(q/'engineering_updates.jsonl')
        assert eng['status']=='PASS' and eng['physical_optimizer_updates']==len(ledger)==32
        assert all(r['physical_call']==i+1 and r['optimizer_executed'] for i,r in enumerate(ledger))
        engineering[arm]=eng
    for arm in ARMS[:2]:
        out=root/'jobs'/arm
        config=json.loads((root/('SETTINGS_'+arm+'.json')).read_text());configs[arm]=config
        result=json.loads((out/'RESULTS.json').read_text())
        assert json.loads((out/'EXIT.json').read_text())['exit_code']==0
        ledger=rows(out/'physical_updates.jsonl')
        assert len(ledger)==result['step']==result['physical_updates']==STEPS
        assert all(r['physical_call']==r['iteration']==i+1 and r['optimizer_executed'] for i,r in enumerate(ledger))
        final=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
        initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
        assert final['arm']==arm and final['step']==final['physical_updates']==STEPS and finite(final)
        entry=json.loads((out/'PREFLIGHT.json').read_text())
        assert entry['status']=='PASS' and entry['source_entry_max_abs_error']==0
        enc1=[k for k in initial['model'] if k.startswith('parent.native.enc1.')]
        frozen=all(torch.equal(initial['model'][k],final['model'][k]) and
                   torch.equal(initial['ema'][k],final['ema'][k]) for k in enc1)
        if arm=='native_lonly': assert frozen
        source=torch.load(config['source_checkpoint'],map_location='cpu',weights_only=False)
        assert source['step']==source['physical_updates']==43600
        assert all(torch.equal(v,source['ema'][k]) for k,v in initial['model'].items())
        evaluated[arm]={}
        for domain,role,filename,field,patients,volumes in [
            ('Siemens','val','private_Siemens_val.jsonl','Siemens',23,46),
            ('Philips','val','private_Philips_val.jsonl','Philips',31,62),
            ('Siemens','train_labeled','private_Siemens_train.jsonl','source_train',10,20),
            ('Philips','train_labeled','private_train.jsonl','train',13,26)]:
            rr=rows(out/filename); summary=aggregate(rr)
            expected={(r['patient_id'],r['phase']) for r in Accessor(config['data'],'mnms',domain,role).rows}
            assert len(rr)==len(expected)==volumes
            assert {(r['patient'],r['phase']) for r in rr}==expected
            assert summary['patients']==patients
            assert summary=={k:v for k,v in result[field].items() if k!='forward_slices'}
            evaluated[arm][field]=rr
        assert not result['independent_test_access']
        costs=json.loads((out/'COSTS.json').read_text())
        assert costs['physical_optimizer_updates']==400 and costs['L_batch_reads']==400 and costs['U_batch_reads']==0
        results[arm]=result
        audited[arm]=dict(exit_code=0,final_checkpoint_finite=True,formal_actual_calls=400,
            patient_phase_sets_exact=True,source_entry_exact=True,enc1_final_unchanged=frozen,
            preflight=entry,costs=costs)
        del final,initial,source
    for arm in ARMS[2:]: assert not (root/'jobs'/arm).exists()
    cfg0=configs['dense_lonly']
    events=rows(Path(cfg0['budget_root'])/'budget_events.jsonl')
    reserved=[r for r in events if r['kind']=='reserved'];executed=[r for r in events if r['kind']=='executed']
    assert len(reserved)==len(executed)==18652
    assert sorted(r['token'] for r in reserved)==sorted(r['token'] for r in executed)==list(range(1,18653))
    counts={}
    for r in executed:
        if r['key'].startswith(('engineering_functional:','formal_functional:')):
            counts[r['key']]=counts.get(r['key'],0)+1
    expected={'engineering_functional:'+arm:32 for arm in ARMS}
    expected.update({'formal_functional:'+arm:400 for arm in ARMS[:2]})
    assert counts==expected and sum(counts.values())==960
    retired=True
    for p in [root/'CONTROLLER_ACTIVE_GRID.json',*(root/'grid_recovery').glob('ACTIVE_*.json')]:
        receipt=json.loads(p.read_text());stat=Path('/proc')/str(receipt['pid'])/'stat'
        assert not (stat.exists() and stat.read_text().split()[21]==receipt['birth'])
    paired={field:dict(overall=compare(evaluated['dense_lonly'][field],evaluated['native_lonly'][field]),
        phases={phase:compare([r for r in evaluated['dense_lonly'][field] if r['phase']==phase],
                              [r for r in evaluated['native_lonly'][field] if r['phase']==phase]) for phase in ('ED','ES')})
        for field in ('Siemens','Philips','source_train','train')}
    gate=json.loads((root/'PLASTICITY_GATE.json').read_text())
    assert gate['status']=='FAIL'
    for field in ('train','Philips'):
        actual=100*(results['native_lonly'][field]['primary_dice']-results['dense_lonly'][field]['primary_dice'])
        assert abs(actual-gate['native_minus_dense_pp'][field])<1e-12
    bank=json.loads(Path(cfg0['bank']).with_suffix('.json').read_text())
    assert bank['status']=='PASS' and not bank['U_label_fields_read'] and bank['cached_continuation_max_error']==0
    public=dict(status='AUDIT_PASS_SCIENTIFIC_STOP_PLASTICITY',arms=results,audits=audited,
        engineering=engineering,paired_native_minus_dense=paired,plasticity_gate=gate,memory_preparation=bank,
        formal_optimizer_calls=800,engineering_optimizer_calls=160,total_new_optimizer_calls=960,
        historical_round_optimizer_calls=17692,round_total_optimizer_calls=18652,round_cap=20000,
        remaining_authorized_optimizer_calls=1348,unused_allocated_formal_calls=1200,
        calls_by_phase=counts,source_historical_updates_separate=43600,
        non_optimizer_preparation_failure=dict(grid_mismatch_fixed=True,optimizer_calls=0,preserved=True),
        unstarted_formal_arms=list(ARMS[2:]),all_workers_retired=retired,
        validation_only=True,independent_test_access=False,hidden_U_gt_access=False,
        additional_patient_forwards=0,additional_optimizer_calls=0,
        unified_SSL_CL_benefit_tested=False,TMI_novelty_established=False)
    output.write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=public['status'],new_calls=960,round_total=18652,
                         remaining=1348,gate=gate),ensure_ascii=False))
