"""Single completion audit of existing checkpoints and metrics; no extra inference."""
import json
from pathlib import Path
import numpy as np
import torch
ARMS=('G_U','M_U')
from .decision import decide
from ..gyro_source_repair_v11.summarize import rows
from ..semantic_condition_v1.summarize import finite_tree
from ..main_style_ssl_v1.data import Accessor
from ..siemens_lonly_v1.run import aggregate,compare
from ..main_style_ssl_v1.runtime import write


def main(root):
    root=Path(root)
    if (root/'PUBLIC_SUMMARY.json').exists():
        print('Existing completion audit retained; no repeat checkpoint reads.');return
    assert json.loads((root/'STATUS.json').read_text())['status']=='COMPLETE'
    for name in ARMS:
        assert json.loads((root/'jobs'/name/'EXIT.json').read_text())['exit_code']==0
    cfg=json.loads((root/'CONTROLLER.json').read_text())
    entry=torch.load(cfg['prepared'],map_location='cpu',weights_only=False)
    assert entry['step']==entry['physical_updates']==69
    caldir=root/'jobs/calibration'
    ci=torch.load(caldir/'initial.pt',map_location='cpu',weights_only=False)
    cf=torch.load(caldir/'final.pt',map_location='cpu',weights_only=False)
    cl=torch.load(caldir/'latest.pt',map_location='cpu',weights_only=False)
    assert json.loads((caldir/'EXIT.json').read_text())['exit_code']==0
    assert finite_tree(ci) and finite_tree(cf) and finite_tree(cl)
    assert ci['optimizer']['state']=={} and ci['step']==ci['physical_updates']==0
    assert all(torch.equal(v,entry['model'][k]) for k,v in ci['model'].items())
    assert cf['step']==cf['physical_updates']==cl['step']==cl['physical_updates']==1000
    assert cf['reads']==cl['reads']==[0,0]
    for field in ['model','ema']:
        assert all(torch.equal(v,cl[field][k]) for k,v in cf[field].items())
    assert all(torch.equal(v,cf['ema'][k]) for k,v in cf['model'].items())
    assert all(torch.equal(v,ci['model'][k]) for k,v in cf['model'].items() if k.startswith('parent.') or k=='frames')
    assert any(not torch.equal(v,ci['model'][k]) for k,v in cf['model'].items() if k.startswith('projector.'))
    assert cf['torch_rng'].equal(cl['torch_rng']) and all(a.equal(b) for a,b in zip(cf['cuda_rng'],cl['cuda_rng']))
    assert cf['optimizer']['param_groups']==cl['optimizer']['param_groups']
    for key,value in cf['optimizer']['state'].items():
        for n,v in value.items():assert torch.equal(v,cl['optimizer']['state'][key][n])
    calls=rows(caldir/'physical_updates.jsonl');assert len(calls)==1000 and all(x['physical_call']==x['iteration']==i+1 for i,x in enumerate(calls))
    eng=json.loads((root/'qualification/calibration/ENGINEERING.json').read_text());assert eng['status']=='PASS' and eng['actual_calls']==32
    assert [x['physical_call'] for x in rows(root/'qualification/calibration/engineering_updates.jsonl')]==list(range(1,33))
    updates=rows(caldir/'private_update_diagnostics.jsonl');assert len(updates)==1000
    assert all(x['native_update_norm']==0 and all(k.startswith('projector') and v==.03 for k,v in x['lrs'].items()) for x in updates)
    bystep={x['step']:x for x in rows(caldir/'private_loss_diagnostics.jsonl')};assert set(bystep)==set(range(1000))
    assert all(not x['labels_read'] and x['anchor_beta']==1/3 and x['source_images']==16 and x['gradient_groups']['encoder_all']['weighted_G_norm']==0 and x['gradient_groups']['projector']['weighted_G_norm']>0 for x in bystep.values())
    cal_audit=dict(formal_actual=1000,engineering_actual=32,native_frame_exact=True,projector_changed=True,labels_read=False,source_native_evaluation_not_repeated=True,finite=True,latest_equals_final=True,momentum_reset_before_target=True)
    # Native/frame stay S†; target entry uses only the new source-prepared phi/BN.
    entry['model']={k:(cf['model'][k] if k.startswith('projector.') else v) for k,v in entry['model'].items()}
    del ci,cf,cl
    base_old=rows(Path(cfg['source_val_rows']))
    base_new=rows(root/'preparation/private_Philips_val.jsonl')
    baseline={'Siemens':aggregate(base_old),'Philips':aggregate(base_new)}
    assert abs(baseline['Siemens']['primary_dice']-.7943371427121955)<1e-12
    results={};metrics={};audits={};diag={};initial_rng=None
    expected={'engineering:calibration':32,'formal:calibration':1000}
    for arm in ARMS:
        out=root/'jobs'/arm;q=root/'qualification'/arm
        calls=rows(out/'physical_updates.jsonl');assert len(calls)==2000
        assert all(x['physical_call']==x['iteration']==i+1 for i,x in enumerate(calls))
        eng=json.loads((q/'ENGINEERING.json').read_text());assert eng['status']=='PASS' and eng['actual_calls']==32
        assert [x['physical_call'] for x in rows(q/'engineering_updates.jsonl')]==list(range(1,33))
        expected['engineering:'+arm]=32;expected['formal:'+arm]=2000
        initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
        final=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
        latest=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert finite_tree(initial) and finite_tree(final) and finite_tree(latest)
        assert initial['optimizer']['state']=={}
        assert all(torch.equal(x,entry['model'][k]) for k,x in initial['model'].items())
        assert initial['step']==initial['physical_updates']==0
        if initial_rng is None:initial_rng=(initial['torch_rng'],initial['cuda_rng'])
        else:
            assert initial_rng[0].equal(initial['torch_rng'])
            assert all(a.equal(b) for a,b in zip(initial_rng[1],initial['cuda_rng']))
        assert final['step']==final['physical_updates']==latest['step']==latest['physical_updates']==2000
        for key in ['model','ema']:
            assert all(torch.equal(v,latest[key][k]) for k,v in final[key].items())
        assert final['reads']==latest['reads'] and final['torch_rng'].equal(latest['torch_rng'])
        assert all(a.equal(b) for a,b in zip(final['cuda_rng'],latest['cuda_rng']))
        for key,value in final['optimizer']['state'].items():
            for n,v in value.items():assert torch.equal(v,latest['optimizer']['state'][key][n])
        assert final['optimizer']['param_groups']==latest['optimizer']['param_groups']
        for n in ['frames','parent.native.decoder.conv_logit.sigma.weight','parent.native.decoder.conv_logit.grad_update']:
            assert torch.equal(initial['model'][n],final['model'][n])
        assert any(not torch.equal(v,final['model'][k]) for k,v in initial['model'].items() if k.startswith('parent.'))
        assert all(torch.equal(v,final['ema'][k]) for k,v in initial['model'].items() if k.startswith('projector.'))
        projector_changed=any(not torch.equal(v,final['model'][k]) for k,v in initial['model'].items() if k.startswith('projector.'))
        assert not projector_changed
        projector_parameters_changed=any(not torch.equal(v,final['model'][k]) for k,v in initial['model'].items() if k.startswith('projector.') and k.endswith(('.weight','.bias')))
        assert not projector_parameters_changed
        r=json.loads((out/'RESULTS.json').read_text());assert r['step']==r['physical_updates']==2000
        rr={}
        roles=[('Siemens','val','private_Siemens_val.jsonl','Siemens',23,46),('Philips','val','private_Philips_val.jsonl','Philips',31,62),
          ('Siemens','train_labeled','private_source_train.jsonl','source_train',10,20),('Philips','train_labeled','private_target_train.jsonl','target_train',13,26),
          ('Siemens','val','private_ema_Siemens_val.jsonl','ema_Siemens',23,46),('Philips','val','private_ema_Philips_val.jsonl','ema_Philips',31,62)]
        for domain,role,file,field,npats,nvol in roles:
            v=rows(out/file);keys={(x['patient_id'],x['phase']) for x in Accessor(cfg['data'],'mnms',domain,role).rows}
            assert len(v)==len(keys)==nvol and {(x['patient'],x['phase']) for x in v}==keys
            a=aggregate(v);assert a['patients']==npats
            reported=r['EMA_diagnostic'][domain] if field.startswith('ema_') else r[field]
            assert a=={k:x for k,x in reported.items() if k!='forward_slices'}
            rr[field]=v
        results[arm]=r;metrics[arm]=rr
        updates=rows(out/'private_update_diagnostics.jsonl');assert len(updates)==2000
        assert [x['actual_call'] for x in updates]==list(range(1,2001))
        assert all(all(not k.startswith('projector') and lr==.001 for k,lr in x['lrs'].items()) for x in updates)
        assert all(x['projector_update_norm']==0 for x in updates)
        bystep={x['step']:x for x in rows(out/'private_loss_diagnostics.jsonl')};assert set(bystep)==set(range(2000))
        assert all(x['fixed_projector'] and x['gradient_groups']['projector']['weighted_G_norm']==0 for x in bystep.values())
        assert all(x['spectral_weight']==(.45 if arm=='G_U' else .3) for x in bystep.values())
        def quant(field):
            values=[field(bystep[i]) for i in range(2000)]
            return dict(min=float(min(values)),median=float(np.median(values)),max=float(max(values)))
        diag[arm]=dict(weighted_spectral_to_CE_ratio=quant(lambda x:x['encoder_weighted_ratio']),
            CE_encoder_norm=quant(lambda x:x['gradient_groups']['encoder_all']['CE_norm']),
            weighted_U_encoder_norm=quant(lambda x:x['gradient_groups']['encoder_all']['weighted_G_norm']),
            native_update_path_length=sum(x['native_update_norm'] for x in updates),
            projector_update_path_length=sum(x['projector_update_norm'] for x in updates),
            geometric_images_per_step=0 if arm=='dense_fixmatch' else 24,replay_images_per_step=0 if arm=='dense_fixmatch' else 8)
        audits[arm]=dict(formal_actual=2000,engineering_actual=32,exit_code=0,finite=True,latest_equals_final=True,
            common_entry_exact=True,initial_momenta_empty=True,frames_constants_unchanged=True,projector_buffers_changed=projector_changed,projector_parameters_changed=projector_parameters_changed,
            patients_and_EDES_exact=True,engineering=eng)
        del initial,final,latest
    events=rows(root/'budget_events.jsonl');reserved=[x for x in events if x['kind']=='reserved'];executed=[x for x in events if x['kind']=='executed']
    assert len(reserved)==len(executed)==5096
    assert sorted(x['token'] for x in reserved)==sorted(x['token'] for x in executed)==list(range(1,5097))
    counts={}
    for x in executed:counts[x['key']]=counts.get(x['key'],0)+1
    assert counts==expected and json.loads((root/'budget_counter.json').read_text())['total']==5096
    for active in [root/'CONTROLLER_ACTIVE.json',*root.glob('ACTIVE_*.json')]:
        a=json.loads(active.read_text());p=Path('/proc')/str(a['pid'])/'stat'
        assert not p.exists() or p.read_text().split()[21]!=a['birth']
    paired={}
    for a,b in [('entry',x) for x in ARMS]+[('M_U','G_U')]:
        pair={}
        for domain in ['Siemens','Philips']:
            x=(base_old if domain=='Siemens' else base_new) if a=='entry' else metrics[a][domain];y=metrics[b][domain]
            assert {(v['patient'],v['phase']) for v in x}=={(v['patient'],v['phase']) for v in y}
            pair[domain]=dict(overall=compare(x,y),phases={p:compare([v for v in x if v['phase']==p],[v for v in y if v['phase']==p]) for p in ['ED','ES']})
        paired[b+'_minus_'+a]=pair
    references={};coordinate_paired={}
    for arm,reference in [('G_U','G_U_reference'),('M_U','M_U_reference')]:
        file=Path(cfg[reference]);references[arm]=json.loads(file.read_text())
        coordinate_paired[arm]={}
        for domain in ['Siemens','Philips']:
            prior=rows(file.parent/('private_'+domain+'_val.jsonl'));current=metrics[arm][domain]
            assert {(v['patient'],v['phase']) for v in prior}=={(v['patient'],v['phase']) for v in current}
            coordinate_paired[arm][domain]=dict(overall=compare(prior,current),phases={p:compare([v for v in prior if v['phase']==p],[v for v in current if v['phase']==p]) for p in ['ED','ES']})
    bank=json.loads((Path(cfg['bank']).parent/'BANK_PREPARED.json').read_text());assert bank['status']=='PASS' and bank['bank_size']==256 and not bank['labels_read']
    decision=decide(root);assert decision==json.loads((root/'DECISION.json').read_text())
    write(root/'PUBLIC_SUMMARY.json',dict(status='AUDIT_PASS',decision=decision,arms=results,baseline=baseline,audits=audits,paired=paired,diagnostics=diag,coordinate_paired=coordinate_paired,bank=bank,coordinate_references=references,dense_reference=json.loads(Path(cfg['dense_reference']).read_text()),
        calibration_audit=cal_audit,actual_optimizer_calls=5096,formal_optimizer_calls=5000,engineering_optimizer_calls=96,cap=5096,unused=0,pending_reservations=0,
        all_workers_retired=True,patient_phase_sets_exact=True,validation_only=True,BayeSeg_independent=True,CARE_independent=True,
        additional_patient_forwards=0,additional_optimizer_calls=0,prior_source_costs_separate=True))
    print(json.dumps(dict(status='AUDIT_PASS',decision=decision,diagnostics=diag)))
