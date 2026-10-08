"""Fixed training-only state probes; unchanged native extractor and V102 predictors."""
import fcntl
import gc
import importlib.util
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


def load(path):
    spec=importlib.util.spec_from_file_location('diagnostic',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def probe(t,j,c):
    before=c.snapshot(t);seed=t.provider.seed
    try:
        t.provider.seed=168;t.cursor=j
        random.seed(861030+j);np.random.seed(861030+j);torch.manual_seed(861030+j)
        within=c.snapshot(t);value=t.extract()
        assert c.e.same(within,c.snapshot(t)), 'extract mutated probe snapshot'
        return value
    finally:
        t.provider.seed=seed;c.restore(t,before)
        assert t.provider.seed==seed and c.e.same(before,c.snapshot(t)), 'probe did not restore native state/RNG'


def main(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c,stage_b as b
    d=load(cfg['diagnostic_entry']);torch.set_num_threads(2);torch.cuda.set_device(0)
    d.SOLVE_LEDGER=root/'LINEAR_SOLVE_LEDGER.jsonl'
    if cfg.get('recovery_source'):
        previous=Path(cfg['recovery_source']);events=[json.loads(v) for v in (previous/'LINEAR_SOLVE_LEDGER.jsonl').read_text().splitlines()]
        assert Counter((v['event'],v['phase']) for v in events)==Counter({('attempt','synthetic'):2,('success','synthetic'):2})
        assert not (previous/'EXTRACTION_LEDGER.jsonl').exists() and d.read(previous/'SYNTHETIC_QUALIFICATION.json')['status']=='PASS'
        d.SOLVES['synthetic']=2
    else:d.selfcheck()
    d.PHASE='diagnostic'
    d.write(root/'SYNTHETIC_QUALIFICATION.json',dict(status='PASS',synthetic_ridge_solves=2,scope='V102 fold predictor checks; actual probe restoration assertions run on every real extraction.'))
    roles=c.split_roles(cfg['data']);assert roles==d.read(Path(cfg['action_root'])/'ROLES.private.json')
    keys,original,reward,gain,old=d.dataset(cfg,'V101');contexts=b.contexts();ledger=b.JobLedger(root,{})
    counts=Counter();unique={'train_labeled':set(),'train_unlabeled':set()};sample=c.e.primitive.CurrentData.__getitem__
    def counted_item(ds,i):
        case=ds.rows[i]['case_id'];assert ds.role in unique
        allowed=roles['A_fit'] if ds.role=='train_labeled' else roles['U_adapt'];assert case in allowed
        counts['dataset_item_attempts_'+ds.role]+=1
        value=sample(ds,i);counts['dataset_item_success_'+ds.role]+=1;unique[ds.role].add(case)
        return value
    c.e.primitive.CurrentData.__getitem__=counted_item
    def hook(name):
        def count(module,args):counts[name+'_image_forwards']+=len(args[0])
        return count
    def create(i,stream):
        random.seed(168);np.random.seed(168);torch.manual_seed(168)
        t=b.make(cfg,roles,ledger,contexts[i],300);t.provider.seed=168+10000*stream
        for name,model in [('student',t.model),('ema',t.ema),('memory',t.memory)]:model.register_forward_pre_hook(hook(name))
        assert set(r['case_id'] for r in t.provider._l.rows)==set(roles['A_fit'][:contexts[i][1]])
        return t
    def restore(t,i,stream,step):
        base=cfg['v92'] if step==100 else cfg['v101'];name='ENTRY100_STREAM.private.pt' if step==100 else 'ON_POLICY_ENTRY200.private.pt'
        value=torch.load(Path(base)/f'jobs/collect{i}_{stream}'/name,map_location='cpu',weights_only=False)
        assert value['step']==step;c.restore(t,value)
    def extract(t,key,kind,j=None):
        ordinal=counts['extractions']+1;assert ordinal<=160
        record=dict(context=key[0],stream=key[1],step=key[2],kind=kind,probe=j,ordinal=ordinal)
        counts['extractions']+=1;c.e.append(root/'EXTRACTION_LEDGER.jsonl',dict(event='attempt',**record))
        try:
            before=c.snapshot(t);seed=t.provider.seed
            value=t.extract() if j is None else probe(t,j,c)
            assert c.e.same(before,c.snapshot(t)) and seed==t.provider.seed
        except BaseException:
            c.e.append(root/'EXTRACTION_LEDGER.jsonl',dict(event='failure',**record));raise
        c.e.append(root/'EXTRACTION_LEDGER.jsonl',dict(event='success',**record));return value.numpy().astype(np.float64)
    # Every original replay qualifies before any new probe features are generated.
    for phase in ('qualification','probe'):
        features={};audits=[]
        for i,ctx in enumerate(contexts):
            for stream in (1,2):
                t=create(i,stream)
                for step in (100,200):
                    key=(ctx[3],stream,step);index=keys.index(key);restore(t,i,stream,step)
                    if phase=='qualification':
                        value=extract(t,key,phase);assert np.array_equal(value,original[index]), 'original state replay mismatch'
                        audits.append(dict(context=key[0],stream=stream,step=step,state_exact=True,snapshot_and_rng_exact=True))
                    else:features[index]=np.stack([extract(t,key,phase,j) for j in range(4)])
                del t;gc.collect();torch.cuda.empty_cache()
                d.write(root/'STATUS.json',dict(status='RUNNING',phase=phase,contexts_processed=2*i+stream,extractions=counts['extractions'])) if not (root/'STATUS.json').exists() else (root/'STATUS.json').write_text(json.dumps(dict(status='RUNNING',phase=phase,contexts_processed=2*i+stream,extractions=counts['extractions'])))
        if phase=='qualification':
            assert len(audits)==32 and counts['extractions']==32
            d.write(root/'REPLAY_QUALIFICATION.json',dict(status='PASS',rows=audits,extractions=32,native_updates=0,query_evaluations=0))
        else:probes=np.stack([features[i] for i in range(32)])
    assert probes.shape==(32,4,24) and counts['extractions']==160 and not ledger.count
    np.savez(root/'FEATURES.private.npz',original=original,probes=probes)
    variants={'ORIGINAL':original,'FIXED_SINGLE':probes[:,0],'FIXED_MEAN4':probes.mean(1)}
    rows=[];stability=[]
    for name,x in variants.items():
        rows+=d.crossvalidate(name,(keys,x,reward,gain,old))
        for context in sorted({k[0] for k in keys}):
            for step in (100,200):
                a,z=[keys.index((context,s,step)) for s in (1,2)]
                stability.append(dict(representation=name,context=context,step=step,crossstream_rms=float(np.sqrt(np.mean((x[a]-x[z])**2))),state_exact=bool(np.array_equal(x[a],x[z]))))
    variance=[dict(context=k[0],stream=k[1],step=k[2],within_probe_rms=float(np.sqrt(np.mean(np.var(probes[i],axis=0))))) for i,k in enumerate(keys)]
    folds=d.aggregate(rows,['dataset','protocol','fold','method'],['dense','gain','old_change','regret'])
    summary=d.aggregate(rows,['dataset','protocol','method'],['dense','gain','old_change','regret'])
    comparison=[]
    for representation in variants:
        for protocol,fold in sorted({(r['protocol'],r['fold']) for r in folds}):
            block={r['method']:r for r in folds if r['dataset']==representation and r['protocol']==protocol and r['fold']==fold}
            orig={r['method']:r for r in folds if r['dataset']=='ORIGINAL' and r['protocol']==protocol and r['fold']==fold}
            single={r['method']:r for r in folds if r['dataset']=='FIXED_SINGLE' and r['protocol']==protocol and r['fold']==fold}
            for method in ('STATE_RIDGE','STATE_1NN'):
                comparison.append(dict(representation=representation,protocol=protocol,fold=fold,method=method,vs_global=block[method]['dense']-block['TRAIN_GLOBAL']['dense'],vs_time=block[method]['dense']-block['TRAIN_TIME']['dense'],vs_original=block[method]['dense']-orig[method]['dense'],vs_single=block[method]['dense']-single[method]['dense']))
    historical=[r for r in d.read(Path(cfg['diagnostic_entry']).parent/'HELDOUT_RESULTS.json') if r['dataset']=='V101']
    current=[dict(r,dataset='V101') for r in rows if r['dataset']=='ORIGINAL'];assert historical==current, 'original predictor replay changed'
    assert len(rows)==960 and d.SOLVES=={'synthetic':2,'diagnostic':30}
    for name,values in [('HELDOUT_RESULTS',rows),('HELDOUT_FOLDS',folds),('COMPARISONS',comparison),('STATE_STABILITY',stability),('WITHIN_PROBE',variance)]:d.write(root/(name+'.json'),values);d.table(root/(name+'.csv'),values)
    d.write(root/'DIAGNOSTIC.json',dict(status='STATE_PROBE_DIAGNOSTIC_COMPLETE_NO_DEPLOYMENT_CLAIM',summary=summary,stability_summary=d.aggregate(stability,['representation','step'],['crossstream_rms','state_exact']),variance_summary=d.aggregate(variance,['step'],['within_probe_rms']),original_V102_prediction_replay='EXACT_ALL320_ROWS',independent_source_validation='NA_SHARED_SOURCE_QUERY_ROLES'))
    d.write(root/'COSTS.json',dict(native_updates=0,actor_optimizer_updates=0,training_query_image_evaluations=0,development_query_image_evaluations=0,ridge_solves=30,synthetic_ridge_solves=2,nearest_neighbor_reference_sets=30,counts=dict(counts),unique_existing_training_images={k:len(v) for k,v in unique.items()},new_annotation_cases=0))
    d.write(root/'FINAL.json',dict(status='COMPLETE',decision='STATE_PROBE_DIAGNOSTIC_COMPLETE_NO_DEPLOYMENT_CLAIM',time=time.time()))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN']);d=load(cfg['diagnostic_entry'])
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    d.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
    try:main(root,cfg)
    except BaseException as exc:d.write(root/'FAILURE.json',dict(error=repr(exc),time=time.time()));raise
