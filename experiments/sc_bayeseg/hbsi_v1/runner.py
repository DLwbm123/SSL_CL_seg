import copy
import csv
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from .data_adapter import CurrentTrainingData, AccessGate, confined, restore_probability
from .diagnostics import parameter_audit
from .local_inference import Refiner
from .objective import SequentialObjective
from .predictive import Predictor, InferenceBudget
from .stream_state import StreamState, digest
from .synthetic import to_json


def protocol(args):
    if args.protocol is None or not args.protocol.is_file():
        raise ValueError('a concrete protocol file is required')
    p=json.loads(args.protocol.read_text())
    if p.get('schema')!=1 or p.get('example',False):
        raise ValueError('example protocols are not executable')
    return p


def storage_check(output):
    root=os.environ.get('SSLCL_STORAGE_ROOT')
    if not root or not output.resolve().is_relative_to(Path(root).resolve()):
        raise PermissionError('real operations require the NAS storage wrapper and NAS output')


def train(args,cfg,ledger,counter,p):
    storage_check(args.output)
    steps=p['steps']
    if type(steps) is not int or not 1<=steps<=1000 or not 1<=p.get('batch_size',1)<=4:
        raise ValueError('invalid declared steps or batch size')
    if not 0<args.budget_seconds<=3600:
        raise ValueError('this implementation entry retains its 3600s hard cap')
    if args.command=='train-stream' and p.get('source_qualification')!='PASS':
        raise PermissionError('stream training requires a separately qualified source protocol')
    state=StreamState.load(p['checkpoint']) if 'checkpoint' in p else StreamState(cfg)
    if state.config!=cfg or state.active is not None:
        raise ValueError('config mismatch or active checkpoint; explicit resume protocol required')
    if args.command=='train-source' and state.receipts:
        raise ValueError('source cannot absorb a sealed stage again')
    if args.command=='train-stream' and not state.receipts:
        raise ValueError('stream requires source posterior')
    with ledger.attempt(args.run_id+'_train',seconds=args.budget_seconds,updates=steps) as out:
        data=CurrentTrainingData(p['data_root'],p['dataset'],p['domain'],p['stage'],cfg,p.get('split_seed',0))
        state.begin_task(p['stage'],data.manifest_hash)
        objective=SequentialObjective(state,Refiner(state.model,state.phi,counter))
        (out/'PARAMETER_AUDIT.json').write_text(json.dumps(parameter_audit(state),indent=2))
        with (out/'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
            for i in range(steps):
                ledger.check()
                L=data.batch('L',p.get('batch_size',1),state.generator)
                U=data.batch('U',p.get('batch_size',1),state.generator)
                loss,parts=objective.estimate(L,U,data.counts,state.generator)
                state.optimizer.zero_grad();loss.backward()
                if any(v.grad is not None and not torch.isfinite(v.grad).all() for group in state.optimizer.param_groups for v in group['params']):
                    raise FloatingPointError('outer gradients')
                state.optimizer.step();state.steps+=1
                for k in ('global_updates','domain_updates','phi_updates','outer_backward_calls'):counter.add(k)
                n=sum(len(b['x']) for b in (L,U) if b is not None)
                counter.add('real_train_images',n);counter.add('current_train_images',n)
                log.write(json.dumps({'step':i+1,**to_json(parts)})+'\n');log.flush()
        state.finalize_task(counter.export());state.save(out/'PRIVATE_SEALED_STATE.pt')
        (out/'DATA_BOUNDARY_AUDIT.json').write_text(json.dumps({'events':data.gate.events,'historical_train_access':0,'val_test_access':0},indent=2))
    return 0


def real_smoke(args,cfg,ledger,counter,p):
    storage_check(args.output)
    if cfg.inner_steps>3 or cfg.global_mc!=1 or not 1<=p.get('batches',2)<=2 or not 1<=p.get('batch_size',1)<=2:
        raise ValueError('zero-update smoke budget exceeded')
    state=StreamState(cfg)
    with ledger.attempt(args.run_id+'_real_smoke',seconds=min(300.,args.budget_seconds or 300.),updates=0) as out:
        data=CurrentTrainingData(p['data_root'],p['dataset'],p['domain'],p['stage'],cfg,p.get('split_seed',0))
        state.begin_task(p['stage'],data.manifest_hash)
        objective=SequentialObjective(state,Refiner(state.model,state.phi,counter))
        before=digest(state.q.state_dict());rows=[]
        for _ in range(p.get('batches',2)):
            L=data.batch('L',p.get('batch_size',1),state.generator);U=data.batch('U',p.get('batch_size',1),state.generator)
            loss,parts=objective.estimate(L,U,data.counts,state.generator)
            state.optimizer.zero_grad();loss.backward();counter.add('outer_backward_calls')
            counter.add('real_train_images',sum(len(b['x']) for b in (L,U) if b is not None))
            # Temporary untrained memory permits an API smoke; it is never saved as a trained posterior.
            temporary=copy.deepcopy(state);temporary.finalize_task()
            prediction=Predictor(counter).predict_image_only(L['x'],temporary,InferenceBudget(S=2,L=2,K=0,repeats=1))
            restored=[restore_probability(prediction['probability'][j:j+1],shape) for j,shape in enumerate(L['native_shapes'])]
            rows.append({'parts':to_json(parts),'prediction_status':prediction['status'],'restored_shapes':[list(v.shape) for v in restored]})
        assert digest(state.q.state_dict())==before
        (out/'REAL_SMOKE_RESULTS.json').write_text(json.dumps({'outer_updates':0,'rows':rows,'data_access':data.gate.events},indent=2))
    return 0


def evaluate(args,cfg,ledger,counter,p):
    import h5py
    import numpy as np
    from .metrics_adapter import phase_metrics, patient_aggregate
    storage_check(args.output)
    if p.get('role')!='val' or p['dataset'] not in ('mnms','prostate'):
        raise PermissionError('only separately authorized frozen grayscale validation supported; test sealed')
    state=StreamState.load(p['checkpoint'])
    if state.config!=cfg or state.active is not None:
        raise ValueError('evaluation requires matching sealed state')
    root=Path(p['data_root']);gate=AccessGate('evaluation','evaluation')
    rows=[]
    with ledger.attempt(args.run_id+'_evaluate',seconds=min(300.,args.budget_seconds or 300.)) as out:
        with (root/'manifests/training'/f"lcrseg_v1_seed{p.get('split_seed',0)}.csv").open() as f:
            records=[r for r in csv.DictReader(f) if r['dataset']==p['dataset'] and r['primary_20pct_split']=='val' and r['site_or_vendor'] in p['domains']]
        if not records:raise ValueError('no validation records')
        for i,r in enumerate(records):
            capability={'stage':r['site_or_vendor'],'role':'val'}
            gate.check(capability,'image')
            with h5py.File(confined(root/'h5/v1',r['image_h5_relpath']),'r') as f:
                volume=np.array(f['image'][...]);spacing=json.loads(f.attrs['processed_spacing'])
            x=torch.from_numpy(volume.copy())[:,None].to(getattr(torch,cfg.dtype))
            small=F.interpolate(x,(cfg.height,cfg.width),mode='bilinear',align_corners=False)
            prediction=Predictor(counter).predict_image_only(small,state,InferenceBudget(**p.get('inference',{})))
            prob=restore_probability(prediction.pop('probability'),x.shape[-2:])
            torch.save({'probability':prob,'diagnostics':to_json(prediction)},out/f'PRIVATE_PREDICTION_{i}.pt')
            gate.check(capability,'label',prediction_sealed=True)
            with h5py.File(confined(root/'h5/v1',r['label_h5_relpath']),'r') as f:y=np.array(f['label'][...])
            rows.append({'patient':r['patient_id'],'case':r['case_id'],'domain':r['site_or_vendor'],
                         'metrics':phase_metrics(prob,y,spacing,cfg.classes),'inference_status':prediction['status']})
            counter.add('validation_images',len(x));ledger.check()
        results={d:patient_aggregate([r for r in rows if r['domain']==d],cfg.classes,1 if p['dataset']=='prostate' else 2) for d in p['domains']}
        (out/'PRIVATE_PATIENT_METRICS.json').write_text(json.dumps({'domains':results,'cases':rows},indent=2))
        (out/'DATA_BOUNDARY_AUDIT.json').write_text(json.dumps({'events':gate.events,'prediction_before_GT':True,'test_access':0},indent=2))
    return 0


def dispatch(args,cfg,ledger,counter):
    if args.command=='synthetic-qualification':
        tests=sorted((p for p in ledger.root.glob('*_TEST_RESULTS.json') if 'PARTIAL_TEST_RESULTS' not in p.name),key=lambda p:p.stat().st_mtime)
        complete=[] if not tests else json.loads(tests[-1].read_text())
        partials=[] if not tests else [p for p in ledger.root.glob('*_PARTIAL_TEST_RESULTS.json') if p.stat().st_mtime>tests[-1].stat().st_mtime]
        if {t['id'] for t in complete}!={f'T{i:02d}' for i in range(1,19)} or any(t['status']!='PASS' for t in complete) or any(t['status']!='PASS' for p in partials for t in json.loads(p.read_text())):
            raise PermissionError('latest unit test suite must pass before synthetic qualification')
        from .synthetic import run
        result=run(cfg,ledger,counter,args.run_id)
        print(json.dumps({'status':result['status'],'optimization_status':result['optimization_status']}))
        return 0 if result['optimization_status']=='PASS' and result['status']=='SYNTHETIC_EXECUTION_COMPLETE_INFERENCE_PASS' else 2
    if args.command=='real-smoke' and args.protocol is None:
        with ledger.attempt(args.run_id+'_real_smoke_not_run',seconds=1) as out:
            (out/'REAL_SMOKE_RESULTS.json').write_text(json.dumps({'status':'NOT_RUN','reason':'no current data protocol supplied; no real data opened','outer_updates':0}))
        print('NOT_RUN: no current real-data protocol supplied')
        return 0
    p=protocol(args)
    if args.command in ('train-source','train-stream'):return train(args,cfg,ledger,counter,p)
    if args.command=='real-smoke':return real_smoke(args,cfg,ledger,counter,p)
    if args.command=='evaluate':return evaluate(args,cfg,ledger,counter,p)
    raise ValueError('unknown command')
