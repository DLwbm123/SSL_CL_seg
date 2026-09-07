"""Create-only CPU stages with separate target, fit, deployment and evaluation processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from care_hr_v0_7_1.execute_r1 import (ROOT, DATA, FORMAL, original_inputs, check_original,
    require, source_state, write_csv, REFERENCE, BASE_POLICIES)
from care_hr_v0_7_1.io_r1 import (read_json,write_json,digest,verify_files,project_training,validate_population)
from care_hr_v0_7_1.io_r2 import load_label,counters,asset_preflight
from care_hr_v0_7_1.scoring_r1 import case_metrics,gains
from care_hr_v0_7_1.summary_r1 import DOMAINS,FIELDS,legacy_aggregate
from shor_jascl_v0_3.core import shor_routes
from .features import extract,choose,predict
from .learning import train_nested,select_threshold,inner_folds

DOC=ROOT/'experiments/lcrseg/docs/shor_uv_v0_8'
PARENT=Path('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/care_hr_v0_7_1_r1_20260907_01/run_01')
R2=Path('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/care_hr_v0_7_1_r2_20260907_01/run_01')
PARENT_HASH='9b186f8c4e80236987700fcb74cf90a5b41071b63df128a5b1e28142df7141a4'
R2_HASH='6bf0a8d6957a023590af224ea725e7f13757a1aa9a4a0c4587d2605f05a46aa0'
POLICIES=('SHOR_CONFIDENCE_VETO','SHOR_UV_CF','SHOR_UV_ROUTED_ONLY','SHOR_UV_GAIN_ONLY')
CONTROL_NAMES=dict(C0='CURRENT',C1='RIDGE_HARD_FROZEN',C3='SHOR_FROZEN',C6='PPC_FROZEN')


def seal(output,name,files,**extra):
    write_json(output/name,dict(source_commit=source_state(),
        files={str(p.relative_to(output)):digest(p) for p in files},**extra))


def verify(output,name):
    s=read_json(output/name)
    require(s['source_commit']==source_state(),'source differs from seal')
    verify_files(output,s['files'])
    return s


def probabilities(meta):
    return {(s,h):np.load(FORMAL/f'expert_probability_cache/seed{s}_expert{h}.npy',
                         mmap_mode='r',allow_pickle=False) for s in range(3) for h in range(3)}


def validate_folds(rows):
    patients={}
    for r in rows:
        require(r['fold'] in range(5),'invalid outer fold')
        require(patients.setdefault(r['patient_id'],r['fold'])==r['fold'],'patient crosses outer folds')
    require(set(patients.values())==set(range(5)),'missing outer fold')
    return patients


def prepare(output,qualification):
    evidence=read_json(qualification)
    require(evidence['source_commit']==source_state() and evidence['local_pass'] and evidence['server_pass'],
            'exact-source local/server qualification absent')
    output.mkdir(parents=True,exist_ok=False)
    (output/'public').mkdir()
    write_json(output/'QUALIFICATION.json',evidence)
    verify_files(ROOT,read_json(DOC/'HISTORY_PROTECTION.json'))
    require(digest(PARENT/'ACTION_SPACE_SEAL.json')==PARENT_HASH,'parent seal mismatch')
    parent=read_json(PARENT/'ACTION_SPACE_SEAL.json')
    verify_files(PARENT,parent['files'])
    require(digest(R2/'ACTION_SPACE_SEAL.json')==R2_HASH,'R2 seal mismatch')
    r2_manifest=read_json(ROOT/'experiments/lcrseg/docs/care_hr_v0_7_1/continuation_r2/PRIVATE_ARTIFACT_MANIFEST_SUMMARY.json')
    old_r2={k:v['sha256'] for k,v in r2_manifest['artifacts'].items()}
    verify_files(R2,old_r2)
    expected,entries=original_inputs()
    # Exact listed immutable inputs only; no checkpoint tensors or image assets opened.
    verify_files(FORMAL,parent['original_files'])
    meta=read_json(PARENT/'input_metadata.json')
    rows=read_json(PARENT/'blind_rows.json')
    projected={}
    for spec in expected['seed_manifests']:
        s=spec['seed']; path=DATA/'manifests/training'/f'lcrseg_v1_seed{s}.csv'
        require(digest(path)==spec['sha256'],'manifest changed')
        projected[s]=project_training(path)
    population=validate_population(rows,projected)
    mapping=validate_folds(rows)
    base=read_json(ROOT/'experiments/lcrseg/docs/ppc_shor_v0_6a/PPC_SHOR_V0_6A_PREREGISTRATION.json')
    threshold_path=Path(base['inputs']['oof_root'])/expected['frozen_shor_threshold_manifest']['path']
    require(digest(threshold_path)==expected['frozen_shor_threshold_manifest']['sha256'],'frozen SHOR threshold mismatch')
    thresholds=read_json(threshold_path)['formal']
    routes=np.full(198,2,dtype=np.int64)
    for s in range(3):
        ix=[i for i,r in enumerate(rows) if r['seed']==s]
        t={r['historical_domain']:float(r['threshold']) for r in thresholds
           if r['seed']==s and r['stage_index']==2 and r['feasible']}
        routes[ix]=shor_routes(np.asarray([rows[i]['alpha'] for i in ix]),stage=2,thresholds=t)
    prob=probabilities(meta)
    x=np.empty((198,2,18),dtype=np.float64)
    for i,r in enumerate(rows):
        s=r['seed']; pos=meta['probability_positions'][str(i)]
        for h in (0,1): x[i,h]=extract(prob[s,2][pos],prob[s,h][pos],r['alpha'],h)
    c6_routes=np.load(PARENT/'routes.npy',allow_pickle=False)
    for f in range(5):
        ix=meta['fold_indices'][str(f)]
        for key,expert in (('C0',None),('C1',None),('C3',None),('C6',None)):
            hard=np.load(FORMAL/meta['control_paths'][f'{f}:{key}'],mmap_mode='r',allow_pickle=False)
            for j,i in enumerate(ix):
                r=rows[i]; s=r['seed']; pos=meta['probability_positions'][str(i)]
                h=2 if key=='C0' else int(np.argmax(r['alpha'])) if key=='C1' else int(routes[i]) if key=='C3' else int(c6_routes[i])
                require(np.array_equal(hard[j],prob[s,h][pos].argmax(0)),'route/cache hard mask parity failed')
    count=counters(); allowed,errors=asset_preflight(DATA,rows,count)
    require(not errors and len(allowed)==198,'all198 metadata admission failed')
    write_json(output/'rows.json',rows);write_json(output/'meta.json',meta)
    write_json(output/'assets.json',allowed);np.save(output/'features.npy',x,allow_pickle=False)
    np.save(output/'SHOR_routes.npy',routes,allow_pickle=False)
    fold_manifest={}
    for f in range(5):
        train=sorted(p for p,v in mapping.items() if v!=f)
        inner=inner_folds(np.array(train))
        fold_manifest[str(f)]={'outer_train_patients':train,
            'outer_test_patients':sorted(p for p,v in mapping.items() if v==f),
            'inner_assignment':dict(zip(train,map(int,inner)))}
    write_json(output/'folds.json',fold_manifest)
    write_json(output/'history_private.json',{'parent_files':parent['files'],'R2_files':old_r2,
                'R1_seal':PARENT_HASH,'R2_seal':R2_HASH})
    parent_terminal_files={n:digest(PARENT/n) for n in ('EVALUATOR_ACCESS_RESERVATION.json','public/CAPACITY_STATUS.json','public/RUNTIME_COUNTERS.json')}
    write_json(output/'parent_terminal_hashes.json',parent_terminal_files)
    files=[output/n for n in ('rows.json','meta.json','assets.json','features.npy','SHOR_routes.npy','folds.json','QUALIFICATION.json','history_private.json','parent_terminal_hashes.json')]
    seal(output,'INPUT_SEAL.json',files,protocol_sha256=digest(DOC/'PROTOCOL.md'),
         feature_contract_sha256=digest(DOC/'FEATURE_CONTRACT.json'),
         source_scope='new heads only; upstream alpha/SHOR are frozen and not cross-fit anew',
         original_files=parent['original_files'],seed_manifests=expected['seed_manifests'],
         threshold_manifest_sha256=digest(threshold_path),GT_payload_reads=0,forward_count=0)
    write_json(output/'public/FOLD_ASSIGNMENT_MANIFEST.json',dict(rows=198,patients=177,
        patient_manifest_sha256=digest(output/'folds.json'),patient_cross_fold_violations=0,
        folds={f:{'train_patients':len(v['outer_train_patients']),'test_patients':len(v['outer_test_patients']),
                  'test_rows':sum(r['fold']==int(f) for r in rows)} for f,v in fold_manifest.items()},
        inner_salt='shor-uv-v0.8-inner',upstream_independent=False))
    write_json(output/'public/INPUT_AND_EXPOSURE_LINEAGE.json',dict(population=population,
        input_seal_sha256=digest(output/'INPUT_SEAL.json'),feature_sha256=digest(output/'features.npy'),
        SHOR_route_sha256=digest(output/'SHOR_routes.npy'),SHOR_historical_routes=int((routes<2).sum()),
        reused_probability_files=9,reused_sample_expert_outputs=594,new_images_read=0,
        new_network_forwards=0,new_checkpoint_tensor_loads=0,metadata_rows_checked=count['asset_metadata_rows_checked'],
        prior_R1_domain_materialized=66,prior_R1_failed_GT_attempts=1,prior_R2_validated_rows=198,
        prior_terminal='FAIL_FROZEN_ACTION_SPACE_CAPACITY',new_GT_payload_reads_before_input_seal=0))


def targets(output):
    verify(output,'INPUT_SEAL.json')
    write_json(output/'TARGET_ACCESS_RESERVATION.json',dict(source_commit=source_state(),
        input_seal_sha256=digest(output/'INPUT_SEAL.json'),created_unix=time.time(),
        role='target materialization and baseline service; no fitting or deployment decisions'))
    rows=read_json(output/'rows.json');meta=read_json(output/'meta.json');assets=read_json(output/'assets.json')
    routes=np.load(output/'SHOR_routes.npy',allow_pickle=False);prob=probabilities(meta)
    count=counters();count.update(new_image_reads=0,network_model_constructions=0,checkpoint_tensor_loads=0)
    physical=set(); records=[]; errors=0.;baseline=[];originals=[];lookup={}
    expected,entries=original_inputs()
    check_original(entries,'case_metrics.jsonl')
    with (FORMAL/'case_metrics.jsonl').open() as f:
        for line in f:
            old=json.loads(line)
            if old['policy'] in BASE_POLICIES: lookup[old['seed'],old['case_id'],old['policy']]=old
    domain_rows={}
    for s in range(3):
        selected={r['case_id'] for r in rows if r['seed']==s}
        rs=project_training(DATA/'manifests/training'/f'lcrseg_v1_seed{s}.csv',
              ('case_id','patient_id','primary_20pct_split','site_or_vendor','label_h5_relpath','label_sha256'),selected)
        count['true_domain_records_materialized']+=len(rs)
        domain_rows.update({(s,r['case_id']):r for r in rs})
    try:
        for i,r in enumerate(rows):
            a=assets[i];full=domain_rows[r['seed'],r['case_id']]
            require(all(a[k]==full[k] for k in ('patient_id','primary_20pct_split','label_h5_relpath','label_sha256')),'target role/binding mismatch')
            require(full['site_or_vendor'] in DOMAINS,'unknown domain')
            gt,_=load_label(DATA,full,count,physical);count['true_domain_rows_processed']+=1
            p=meta['probability_positions'][str(i)];s=r['seed']
            expert=[case_metrics(prob[s,h][p].argmax(0),gt) for h in range(3)]
            before=expert[2]
            target=np.array([[expert[h]['rim_dice']-before['rim_dice'],expert[h]['cup_dice']-before['cup_dice'],0] for h in range(2)])
            target[:,2]=np.maximum(0,-target[:,:2].min(1))
            record={**r,'domain':full['site_or_vendor'],'domain_index':DOMAINS.index(full['site_or_vendor']),
                    'expert_metrics':expert,'targets':target.tolist(),'has_evaluable_gt':before['has_evaluable_gt']}
            records.append(record)
        # Preserve inherited original ordering and grouping for zero-error parity.
        for f in range(5):
            ix=meta['fold_indices'][str(f)]
            hard={p:np.load(FORMAL/meta['control_paths'][f'{f}:{p}'],mmap_mode='r',allow_pickle=False) for p in BASE_POLICIES}
            for j,i in enumerate(ix):
                r=records[i];s=r['seed'];p=meta['probability_positions'][str(i)]
                for policy,name in BASE_POLICIES.items():
                    old=lookup[s,r['case_id'],policy]
                    h=old['route']
                    require(h in (0,1,2) and old['domain']==r['domain'],'baseline route/domain mismatch')
                    if policy=='C3': require(h==int(routes[i]),'SHOR route sidecar mismatch')
                    require(np.array_equal(hard[policy][j],prob[s,h][p].argmax(0)),'control reference mismatch')
                    m=r['expert_metrics'][h]
                    error=max(abs(m[k]-old[k]) for k in FIELDS);errors=max(errors,error)
                    require(error<=1e-12,'BLOCKED_BASELINE_MISMATCH')
                    baseline.append({**r,**m,**gains(r['expert_metrics'][2],m),'policy':name})
                    originals.append({**old,'policy':name})
                    count['baseline_case_policy_comparisons_completed']+=1
                    count['baseline_scalar_comparisons_completed']+=4
        oldagg,newagg=legacy_aggregate(originals),legacy_aggregate(baseline)
        require(len(oldagg)==len(newagg),'baseline group count')
        for old,new in zip(oldagg,newagg):
            require(all(old[k]==new[k] for k in ('level','key','policy','cases')),'group ordering')
            error=max(abs(old[k]-new[k]) for k in FIELDS);errors=max(errors,error)
            require(error<=1e-12,'BLOCKED_BASELINE_GROUP_MISMATCH')
            if new['level']=='overall':require(abs(new['foreground_dice']-REFERENCE[new['policy']])<=1e-12,'frozen reference')
        # These are evaluator-owned records. Trainers open only their own scoped package.
        write_json(output/'evaluation_targets.json',records)
        write_json(output/'baseline_records.json',baseline)
        for f in range(5):
            train=[r for r in records if r['fold']!=f]
            require(not ({r['patient_id'] for r in train}&{r['patient_id'] for r in records if r['fold']==f}),'outer leak')
            package=[{k:r[k] for k in ('row_index','patient_id','seed','domain_index','targets','has_evaluable_gt')} for r in train]
            write_json(output/f'train_targets_fold{f}.json',package)
        files=[output/'evaluation_targets.json',output/'baseline_records.json']+[output/f'train_targets_fold{f}.json' for f in range(5)]
        seal(output,'TARGET_SEAL.json',files,outer_policy_metrics_viewed=False,
             role='target service; labels never sent to deployment; each fit reads one scoped training file')
        write_json(output/'public/BASELINE_PARITY.json',dict(status='PASS_BASELINE_PARITY',case_policy_comparisons=792,
            scalar_comparisons=3168,grouped_rows=len(newagg),maximum_absolute_difference=errors,rtol=0,atol=1e-12))
        write_json(output/'public/TARGET_ACCESS_COUNTERS.json',count)
    finally:
        write_json(output/'TARGET_COUNTERS.json',count)


def train_fold(output,fold):
    verify(output,'INPUT_SEAL.json')
    # Do not call generic TARGET_SEAL verification: that would read/hash outer targets.
    ts=read_json(output/'TARGET_SEAL.json')
    require(ts['source_commit']==source_state(),'target source mismatch')
    name=f'train_targets_fold{fold}.json'
    require(digest(output/name)==ts['files'][name],'training package changed')
    data=read_json(output/name)
    mapping=read_json(output/'folds.json')[str(fold)]
    allowed=set(mapping['outer_train_patients']); excluded=set(mapping['outer_test_patients'])
    require({r['patient_id'] for r in data}==allowed and not allowed&excluded,'outer patient boundary')
    x=np.load(output/'features.npy',allow_pickle=False)[[r['row_index'] for r in data]]
    routes=np.load(output/'SHOR_routes.npy',allow_pickle=False)[[r['row_index'] for r in data]]
    y=np.array([r['targets'] for r in data])
    patients=np.array([r['patient_id'] for r in data]); seeds=np.array([r['seed'] for r in data])
    domains=np.array([r['domain_index'] for r in data]);evaluable=np.array([r['has_evaluable_gt'] for r in data])
    d=output/f'fold{fold}';d.mkdir()
    with (d/'FIT_EVENTS.jsonl').open('x') as log:
        def sink(variant):
            def emit(event,record):
                log.write(json.dumps(dict(event=event,variant=variant,outer_fold=fold,record=record),allow_nan=False)+'\n')
                log.flush()
            return emit
        cf=train_nested(x,y,patients,routes,seeds,domains,evaluable,ledger_sink=sink('cf'))
        routed=train_nested(x,y,patients,routes,seeds,domains,evaluable,True,ledger_sink=sink('routed_only'))
    confidence,candidates=select_threshold(None,x,y,routes,seeds,domains,evaluable,'confidence')
    bundles={'SHOR_UV_CF':dict(model=cf['model'],candidate=cf['choices']['cf']),
             'SHOR_UV_GAIN_ONLY':dict(model=cf['model'],candidate=cf['choices'].get('gain')),
             'SHOR_UV_ROUTED_ONLY':dict(model=routed['model'],candidate=routed['choices']['cf']),
             'SHOR_CONFIDENCE_VETO':dict(model=None,candidate=confidence)}
    write_json(d/'models.json',bundles)
    write_json(d/'INNER_SELECTION.json',dict(cf=cf,routed_only=routed,confidence_candidates=candidates,
        selected_confidence=confidence,training_patients=sorted(allowed),excluded_patients=sorted(excluded),
        outer_GT_target_access=0,read_training_package=name))
    seal(output,f'fold{fold}/MODEL_SEAL.json',[d/'models.json'],
         training_evidence_sha256={n:digest(d/n) for n in ('INNER_SELECTION.json','FIT_EVENTS.jsonl')},
         outer_fold=fold,target_package_sha256=ts['files'][name],outer_GT_target_access=0)


def deploy_fold(output,fold):
    verify(output,'INPUT_SEAL.json');verify(output,f'fold{fold}/MODEL_SEAL.json')
    rows=read_json(output/'rows.json');meta=read_json(output/'meta.json')
    ix=meta['fold_indices'][str(fold)]
    x=np.load(output/'features.npy',allow_pickle=False);routes=np.load(output/'SHOR_routes.npy',allow_pickle=False)
    bundle=read_json(output/f'fold{fold}/models.json'); prob=probabilities(meta)
    d=output/f'fold{fold}';files=[];allchoices={};pair_predictions={}
    for policy,b in bundle.items():
        decisions=[];hard=[]
        for i in ix:
            h=int(routes[i])
            selected=choose(h,x[i,h] if h<2 else np.zeros(18),b['model'],b['candidate'])
            require(selected in (2,h),'illegal new expert selection')
            r=rows[i];pos=meta['probability_positions'][str(i)]
            hard.append(prob[r['seed'],selected][pos].argmax(0).astype(np.uint8))
            decisions.append(selected)
        name=d/(policy+'.npy');np.save(name,np.stack(hard),allow_pickle=False);files.append(name)
        allchoices[policy]=decisions
        pair_predictions[policy]=None if b['model'] is None else predict(b['model'],x[ix]).tolist()
    write_json(d/'decisions.json',allchoices);write_json(d/'pair_predictions.json',pair_predictions)
    files.extend([d/'decisions.json',d/'pair_predictions.json'])
    seal(output,f'fold{fold}/PREDICTION_SEAL.json',files,outer_fold=fold,rows=len(ix),
         model_seal_sha256=digest(d/'MODEL_SEAL.json'),GT_reads=0,domain_reads=0,
         hard_output='exact frozen probability argmax; route 2 always current')


def seal_all(output):
    files=[]
    for f in range(5):
        verify(output,f'fold{f}/PREDICTION_SEAL.json')
        files.append(output/f'fold{f}/PREDICTION_SEAL.json')
    seal(output,'ALL_OUTER_PREDICTIONS_SEALED.json',files,rows=198,policies=list(POLICIES),
         all_five_folds_complete=True,outer_policy_metrics_viewed=False,created_unix=time.time())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('prepare','targets','train','deploy','seal','evaluate'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--qualification',type=Path)
    p.add_argument('--fold',type=int,choices=range(5))
    a=p.parse_args(); out=a.output.resolve()
    require(str(out).startswith('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/shor_uv_v0_8_'),
            'outputs require new registered NAS root')
    try:
        if a.stage=='prepare':prepare(out,a.qualification)
        elif a.stage=='targets':targets(out)
        elif a.stage=='train':train_fold(out,a.fold)
        elif a.stage=='deploy':deploy_fold(out,a.fold)
        elif a.stage=='seal':seal_all(out)
        else:
            from .report import evaluate
            evaluate(out)
        print(json.dumps({'stage':a.stage,'fold':a.fold,'status':'COMPLETED'}),flush=True)
    except Exception as exc:
        if out.exists():
            name='failure_'+a.stage+('_'+str(a.fold) if a.fold is not None else '')
            trace=out/(name+'.traceback.txt')
            with trace.open('x') as f:f.write(traceback.format_exc())
            write_json(out/(name+'.json'),dict(status='INCOMPLETE_ENGINEERING',exception=type(exc).__name__,
                traceback_sha256=digest(trace),outer_summary_viewed=(out/'OUTER_EVALUATION_RESERVATION.json').exists()))
        raise


if __name__=='__main__':main()
